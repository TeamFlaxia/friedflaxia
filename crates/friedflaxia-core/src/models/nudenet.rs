//! NudeNet YOLOv8n (320) inference and class-agnostic NMS.
//!
//! The ONNX graph is YOLOv8n with folded BatchNorm. We implement the fixed
//! 320x320 architecture directly and load its convolution weights from GGUF.

use super::ModelError;
use crate::gguf::Gguf;
use crate::ops::silu;
use crate::tensor::tensor;
use std::collections::HashMap;

pub const NUDENET_SIZE: usize = 320;
pub const NUDENET_CLASSES: usize = 18;
const STRIDES: [usize; 3] = [8, 16, 32];

pub const LABELS: [&str; NUDENET_CLASSES] = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
];

#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub label: &'static str,
    pub score: f32,
    pub box_xyxy: [f32; 4],
}

pub struct NudeNet {
    weights: HashMap<String, Vec<f32>>,
}

impl NudeNet {
    pub fn load(gguf: &Gguf, file: &[u8]) -> Result<Self, ModelError> {
        let mut weights = HashMap::with_capacity(gguf.tensor_count());
        for info in gguf.tensors() {
            let values = tensor(gguf, file, &info.name)?.vec()?;
            weights.insert(info.name.clone(), values);
        }
        Ok(Self { weights })
    }

    fn weight(&self, name: &str) -> Result<&[f32], ModelError> {
        self.weights
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| ModelError::Shape(format!("missing NudeNet tensor {name}")))
    }

    /// Debug helper: run only the NMS stage on a raw anchor-major output.
    #[doc(hidden)]
    pub fn debug_nms(
        raw: &[f32],
        top_k: usize,
        iou_threshold: f32,
        score_threshold: f32,
    ) -> Vec<Detection> {
        nms(raw, top_k, iou_threshold, score_threshold)
    }

    /// Run the detector and return NMS-selected detections in input-pixel space.
    pub fn detect(
        &self,
        input: &[f32],
        top_k: usize,
        iou_threshold: f32,
        score_threshold: f32,
    ) -> Result<Vec<Detection>, ModelError> {
        let raw = self.raw_output(input)?;
        Ok(nms(&raw, top_k, iou_threshold, score_threshold))
    }

    /// Raw `[22, 2100]` YOLO output (channel-major per anchor column).
    pub fn raw_output(&self, input: &[f32]) -> Result<Vec<f32>, ModelError> {
        if input.len() != 3 * NUDENET_SIZE * NUDENET_SIZE {
            return Err(ModelError::Invalid("NudeNet input must be 3x320x320"));
        }
        let mut x = conv(self, input, 3, 320, 320, "model.0.conv", 16, 3, 2, 1)?;
        silu_inplace(&mut x);
        x = conv(self, &x, 16, 160, 160, "model.1.conv", 32, 3, 2, 1)?;
        silu_inplace(&mut x);
        x = c2f(self, &x, 32, 80, 80, "model.2", 32, 1, true)?;
        x = conv(self, &x, 32, 80, 80, "model.3.conv", 64, 3, 2, 1)?;
        silu_inplace(&mut x);
        let p2 = c2f(self, &x, 64, 40, 40, "model.4", 64, 2, true)?;
        x = conv(self, &p2, 64, 40, 40, "model.5.conv", 128, 3, 2, 1)?;
        silu_inplace(&mut x);
        let p3 = c2f(self, &x, 128, 20, 20, "model.6", 128, 2, true)?;
        x = conv(self, &p3, 128, 20, 20, "model.7.conv", 256, 3, 2, 1)?;
        silu_inplace(&mut x);
        let p4 = c2f(self, &x, 256, 10, 10, "model.8", 256, 1, true)?;
        let p5 = sppf(self, &p4, 256, 10, 10, "model.9", 256)?;

        let up5 = upsample_nearest(&p5, 256, 10, 10);
        let cat4 = concat(&[up5, p3], 256, 128, 20, 20);
        let n4 = c2f(self, &cat4, 384, 20, 20, "model.12", 128, 1, false)?;
        let up4 = upsample_nearest(&n4, 128, 20, 20);
        let cat3 = concat(&[up4, p2], 128, 64, 40, 40);
        let n3 = c2f(self, &cat3, 192, 40, 40, "model.15", 64, 1, false)?;

        let down3 = conv(self, &n3, 64, 40, 40, "model.16.conv", 64, 3, 2, 1)?;
        let down3 = silu_tensor(down3);
        let cat4b = concat(&[down3, n4], 64, 128, 20, 20);
        let n4b = c2f(self, &cat4b, 192, 20, 20, "model.18", 128, 1, false)?;
        let down4 = conv(self, &n4b, 128, 20, 20, "model.19.conv", 128, 3, 2, 1)?;
        let down4 = silu_tensor(down4);
        let cat5 = concat(&[down4, p5], 128, 256, 10, 10);
        let n5 = c2f(self, &cat5, 384, 10, 10, "model.21", 256, 1, false)?;

        let mut raw = Vec::with_capacity(22 * 2100);
        for (scale, (feature, channels, size)) in
            [(n3, 64usize, 40usize), (n4b, 128, 20), (n5, 256, 10)]
                .into_iter()
                .enumerate()
        {
            let head = detect_scale(self, &feature, channels, size, scale)?;
            raw.extend(head);
        }
        Ok(raw)
    }
}

#[allow(clippy::too_many_arguments)]
fn conv(
    net: &NudeNet,
    input: &[f32],
    in_c: usize,
    h: usize,
    w: usize,
    prefix: &str,
    out_c: usize,
    k: usize,
    stride: usize,
    pad: usize,
) -> Result<Vec<f32>, ModelError> {
    let weights = net.weight(&format!("{prefix}.weight"))?;
    let bias = net.weight(&format!("{prefix}.bias"))?;
    let expected = out_c * in_c * k * k;
    if weights.len() != expected || bias.len() != out_c {
        return Err(ModelError::Shape(format!("{prefix} weight shape")));
    }
    let out_h = (h + 2 * pad - k) / stride + 1;
    let out_w = (w + 2 * pad - k) / stride + 1;
    let mut out = vec![0.0f32; out_c * out_h * out_w];
    for o in 0..out_c {
        let b = bias[o];
        for y in 0..out_h {
            for x in 0..out_w {
                let mut sum = 0.0f32;
                for c in 0..in_c {
                    for ky in 0..k {
                        let iy = y * stride + ky;
                        if iy < pad {
                            continue;
                        }
                        let iy = iy - pad;
                        if iy >= h {
                            continue;
                        }
                        for kx in 0..k {
                            let ix = x * stride + kx;
                            if ix < pad {
                                continue;
                            }
                            let ix = ix - pad;
                            if ix >= w {
                                continue;
                            }
                            let weight = weights[(((o * in_c + c) * k + ky) * k) + kx];
                            sum += weight * input[c * h * w + iy * w + ix];
                        }
                    }
                }
                out[o * out_h * out_w + y * out_w + x] = sum + b;
            }
        }
    }
    Ok(out)
}

fn silu_tensor(mut values: Vec<f32>) -> Vec<f32> {
    silu_inplace(&mut values);
    values
}

fn silu_inplace(values: &mut [f32]) {
    for value in values {
        *value = silu(*value);
    }
}

#[allow(clippy::too_many_arguments)]
fn c2f(
    net: &NudeNet,
    input: &[f32],
    in_c: usize,
    h: usize,
    w: usize,
    prefix: &str,
    out_c: usize,
    bottlenecks: usize,
    shortcut: bool,
) -> Result<Vec<f32>, ModelError> {
    let hidden = out_c / 2;
    let cv1 = conv(
        net,
        input,
        in_c,
        h,
        w,
        &format!("{prefix}.cv1.conv"),
        out_c,
        1,
        1,
        0,
    )?;
    let cv1 = silu_tensor(cv1);
    let (first, mut last) = split_half(&cv1, hidden, h, w);
    let mut parts = vec![first, last.clone()];
    for index in 0..bottlenecks {
        let cv1 = conv(
            net,
            &last,
            hidden,
            h,
            w,
            &format!("{prefix}.m.{index}.cv1.conv"),
            hidden,
            3,
            1,
            1,
        )?;
        let cv1 = silu_tensor(cv1);
        let cv2 = conv(
            net,
            &cv1,
            hidden,
            h,
            w,
            &format!("{prefix}.m.{index}.cv2.conv"),
            hidden,
            3,
            1,
            1,
        )?;
        let cv2 = silu_tensor(cv2);
        last = if shortcut { add(&last, &cv2) } else { cv2 };
        parts.push(last.clone());
    }
    let concat = concat_many(&parts, hidden, h, w);
    let cv2 = conv(
        net,
        &concat,
        hidden * parts.len(),
        h,
        w,
        &format!("{prefix}.cv2.conv"),
        out_c,
        1,
        1,
        0,
    )?;
    Ok(silu_tensor(cv2))
}

#[allow(clippy::too_many_arguments)]
fn sppf(
    net: &NudeNet,
    input: &[f32],
    in_c: usize,
    h: usize,
    w: usize,
    prefix: &str,
    out_c: usize,
) -> Result<Vec<f32>, ModelError> {
    let hidden = out_c / 2;
    let cv1 = conv(
        net,
        input,
        in_c,
        h,
        w,
        &format!("{prefix}.cv1.conv"),
        hidden,
        1,
        1,
        0,
    )?;
    let cv1 = silu_tensor(cv1);
    let p1 = maxpool5(&cv1, hidden, h, w);
    let p2 = maxpool5(&p1, hidden, h, w);
    let p3 = maxpool5(&p2, hidden, h, w);
    let concat = concat_many(&[cv1, p1, p2, p3], hidden, h, w);
    let cv2 = conv(
        net,
        &concat,
        hidden * 4,
        h,
        w,
        &format!("{prefix}.cv2.conv"),
        out_c,
        1,
        1,
        0,
    )?;
    Ok(silu_tensor(cv2))
}

#[allow(clippy::too_many_arguments)]
fn detect_scale(
    net: &NudeNet,
    input: &[f32],
    channels: usize,
    size: usize,
    scale: usize,
) -> Result<Vec<f32>, ModelError> {
    let mut box_branch = conv(
        net,
        input,
        channels,
        size,
        size,
        &format!("model.22.cv2.{scale}.0.conv"),
        64,
        3,
        1,
        1,
    )?;
    box_branch = silu_tensor(box_branch);
    box_branch = conv(
        net,
        &box_branch,
        64,
        size,
        size,
        &format!("model.22.cv2.{scale}.1.conv"),
        64,
        3,
        1,
        1,
    )?;
    box_branch = silu_tensor(box_branch);
    box_branch = conv(
        net,
        &box_branch,
        64,
        size,
        size,
        &format!("model.22.cv2.{scale}.2"),
        64,
        1,
        1,
        0,
    )?;

    let mut class_branch = conv(
        net,
        input,
        channels,
        size,
        size,
        &format!("model.22.cv3.{scale}.0.conv"),
        64,
        3,
        1,
        1,
    )?;
    class_branch = silu_tensor(class_branch);
    class_branch = conv(
        net,
        &class_branch,
        64,
        size,
        size,
        &format!("model.22.cv3.{scale}.1.conv"),
        64,
        3,
        1,
        1,
    )?;
    class_branch = silu_tensor(class_branch);
    class_branch = conv(
        net,
        &class_branch,
        64,
        size,
        size,
        &format!("model.22.cv3.{scale}.2"),
        18,
        1,
        1,
        0,
    )?;
    for value in &mut class_branch {
        *value = 1.0 / (1.0 + (-*value).exp());
    }

    let dfl = net.weight("model.22.dfl.conv.weight")?;
    let stride = STRIDES[scale] as f32;
    let mut out = Vec::with_capacity(22 * size * size);
    for anchor in 0..size * size {
        let y = anchor / size;
        let x = anchor % size;
        let anchor_x = (x as f32 + 0.5) * stride;
        let anchor_y = (y as f32 + 0.5) * stride;
        let mut distances = [0.0f32; 4];
        for side in 0..4 {
            let mut max = f32::NEG_INFINITY;
            for bin in 0..16 {
                max = max.max(box_branch[(side * 16 + bin) * size * size + anchor]);
            }
            let mut sum = 0.0f32;
            let mut weighted = 0.0f32;
            for bin in 0..16 {
                let value = (box_branch[(side * 16 + bin) * size * size + anchor] - max).exp();
                sum += value;
                weighted += value * dfl[bin];
            }
            distances[side] = weighted / sum;
        }
        let x1 = anchor_x - distances[0] * stride;
        let y1 = anchor_y - distances[1] * stride;
        let x2 = anchor_x + distances[2] * stride;
        let y2 = anchor_y + distances[3] * stride;
        out.push((x1 + x2) * 0.5);
        out.push((y1 + y2) * 0.5);
        out.push(x2 - x1);
        out.push(y2 - y1);
        for class in 0..18 {
            out.push(class_branch[class * size * size + anchor]);
        }
    }
    Ok(out)
}

fn nms(raw: &[f32], top_k: usize, iou_threshold: f32, score_threshold: f32) -> Vec<Detection> {
    let anchors = raw.len() / 22;
    let mut candidates = Vec::new();
    for anchor in 0..anchors {
        let base = anchor * 22;
        let mut label = 0usize;
        let mut score = 0.0f32;
        for class in 0..18 {
            let value = raw[base + 4 + class];
            if value > score {
                score = value;
                label = class;
            }
        }
        if score > score_threshold {
            candidates.push((score, label, base));
        }
    }
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut kept: Vec<(f32, usize, usize, [f32; 4])> = Vec::new();
    for (score, label, base) in candidates {
        let cx = raw[base];
        let cy = raw[base + 1];
        let bw = raw[base + 2];
        let bh = raw[base + 3];
        let box_xyxy = [cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5];
        let mut overlaps = false;
        for (_, _, _, other) in &kept {
            if iou(&box_xyxy, other) > iou_threshold {
                overlaps = true;
                break;
            }
        }
        if !overlaps {
            kept.push((score, label, base, box_xyxy));
            if kept.len() == top_k {
                break;
            }
        }
    }
    kept.into_iter()
        .map(|(score, label, _, box_xyxy)| Detection {
            label: LABELS[label],
            score,
            box_xyxy,
        })
        .collect()
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

fn upsample_nearest(input: &[f32], channels: usize, h: usize, w: usize) -> Vec<f32> {
    let out_h = h * 2;
    let out_w = w * 2;
    let mut out = vec![0.0f32; channels * out_h * out_w];
    for c in 0..channels {
        for y in 0..out_h {
            for x in 0..out_w {
                out[c * out_h * out_w + y * out_w + x] = input[c * h * w + (y / 2) * w + x / 2];
            }
        }
    }
    out
}

fn maxpool5(input: &[f32], channels: usize, h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; channels * h * w];
    for c in 0..channels {
        for y in 0..h {
            for x in 0..w {
                let mut value = f32::NEG_INFINITY;
                for ky in 0..5 {
                    let iy = y as isize + ky as isize - 2;
                    if iy < 0 || iy >= h as isize {
                        continue;
                    }
                    for kx in 0..5 {
                        let ix = x as isize + kx as isize - 2;
                        if ix < 0 || ix >= w as isize {
                            continue;
                        }
                        value = value.max(input[c * h * w + iy as usize * w + ix as usize]);
                    }
                }
                out[c * h * w + y * w + x] = value;
            }
        }
    }
    out
}

fn split_half(input: &[f32], half: usize, h: usize, w: usize) -> (Vec<f32>, Vec<f32>) {
    let plane = h * w;
    let mut first = vec![0.0f32; half * plane];
    let mut second = vec![0.0f32; half * plane];
    first.copy_from_slice(&input[..half * plane]);
    second.copy_from_slice(&input[half * plane..2 * half * plane]);
    (first, second)
}

fn concat(
    parts: &[Vec<f32>],
    first_channels: usize,
    second_channels: usize,
    h: usize,
    w: usize,
) -> Vec<f32> {
    let plane = h * w;
    let mut out = Vec::with_capacity((first_channels + second_channels) * plane);
    out.extend_from_slice(&parts[0][..first_channels * plane]);
    out.extend_from_slice(&parts[1][..second_channels * plane]);
    out
}

fn concat_many(parts: &[Vec<f32>], channels: usize, h: usize, w: usize) -> Vec<f32> {
    let plane = h * w;
    let mut out = Vec::with_capacity(parts.len() * channels * plane);
    for part in parts {
        out.extend_from_slice(&part[..channels * plane]);
    }
    out
}

fn add(left: &[f32], right: &[f32]) -> Vec<f32> {
    left.iter().zip(right).map(|(a, b)| a + b).collect()
}
