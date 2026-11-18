use super::GgufError;

pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    pub(crate) fn position(&self) -> usize {
        self.position
    }

    pub(crate) fn take(&mut self, len: usize) -> Result<&'a [u8], GgufError> {
        let end = self
            .position
            .checked_add(len)
            .ok_or(GgufError::IntegerOverflow)?;
        let slice = self
            .bytes
            .get(self.position..end)
            .ok_or(GgufError::UnexpectedEof)?;
        self.position = end;
        Ok(slice)
    }

    pub(crate) fn u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.take(1)?[0])
    }

    pub(crate) fn u16(&mut self) -> Result<u16, GgufError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    pub(crate) fn u32(&mut self) -> Result<u32, GgufError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    pub(crate) fn u64(&mut self) -> Result<u64, GgufError> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    pub(crate) fn string(&mut self, max_bytes: u64) -> Result<String, GgufError> {
        let len = self.u64()?;
        if len > max_bytes {
            return Err(GgufError::LimitExceeded("string length"));
        }
        let len = usize::try_from(len).map_err(|_| GgufError::IntegerOverflow)?;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| GgufError::InvalidUtf8)
    }
}
