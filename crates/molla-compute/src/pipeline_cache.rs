/// Opaque compiler data exported by a wgpu pipeline cache.
#[derive(Clone)]
pub struct PipelineCacheSnapshot {
    pub(crate) bytes: Vec<u8>,
}

impl PipelineCacheSnapshot {
    /// Serialized data for explicit application-managed persistence.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Import compiler data recovered from trusted application storage.
    ///
    /// # Safety
    /// Bytes must be an unmodified result of wgpu `PipelineCache::get_data`,
    /// including snapshots exported by this type. Arbitrary or backend-native
    /// cache bytes are unsupported. Integrity and provenance are caller obligations.
    pub unsafe fn from_persisted_bytes(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }
}

#[cfg(test)]
#[path = "pipeline_cache_tests.rs"]
mod tests;
