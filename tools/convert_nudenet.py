#!/usr/bin/env python3
"""Convert the NudeNet YOLOv8 ONNX weights to a FriedFlaxia GGUF.

The runtime implements the YOLOv8n graph directly, so the GGUF only carries
the convolution weights and metadata. The ONNX NMS graph is replaced by a
Rust implementation.

Usage:
  python3 tools/convert_nudenet.py 320n.onnx nudenet-320n-f32.gguf
"""

import sys

import gguf
import numpy as np
import onnx
from onnx import numpy_helper


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    source, target = sys.argv[1], sys.argv[2]
    model = onnx.load(source)
    writer = gguf.GGUFWriter(target, "yolov8-nudenet")
    writer.add_uint32("nudenet.input_size", 320)
    writer.add_uint32("nudenet.classes", 18)
    writer.add_array("nudenet.strides", [8, 16, 32])
    writer.add_string(
        "nudenet.labels",
        "FEMALE_GENITALIA_COVERED,FACE_FEMALE,BUTTOCKS_EXPOSED,FEMALE_BREAST_EXPOSED,"
        "FEMALE_GENITALIA_EXPOSED,MALE_BREAST_EXPOSED,ANUS_EXPOSED,FEET_EXPOSED,"
        "BELLY_COVERED,FEET_COVERED,ARMPITS_COVERED,ARMPITS_EXPOSED,FACE_MALE,"
        "BELLY_EXPOSED,MALE_GENITALIA_EXPOSED,ANUS_COVERED,FEMALE_BREAST_COVERED,"
        "BUTTOCKS_COVERED",
    )

    count = 0
    for initializer in model.graph.initializer:
        array = numpy_helper.to_array(initializer)
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        if array.dtype != np.float32:
            array = array.astype(np.float32)
        writer.add_tensor(initializer.name, array)
        count += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {target}: {count} tensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
