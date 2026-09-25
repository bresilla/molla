//! `Buffer<T>` — a host/device-mirrored array.
//!
//! Both halves are functional under the `gpu` feature: `ensure_device`
//! allocates the wgpu buffer on demand, `upload` / `download`
//! synchronise via `Queue::write_buffer` + `MAP_READ` round-trips.
//! Without the `gpu` feature only the CPU half exists. All tests
//! against the math + sim layer can drive the host-only path directly.

use molla_core::{Error, Result};
use molla_math::{
    GpuMat3, GpuSpatialVector, GpuTransform, Mat3, Real, SpatialVector, Transform, Vec3,
};

#[cfg(feature = "gpu")]
mod device_sealed {
    pub trait Sealed {}
    impl Sealed for wgpu::Device {}
    impl Sealed for std::sync::Arc<wgpu::Device> {}
}

/// Device reference with optional canonical ownership identity.
#[cfg(feature = "gpu")]
pub trait BufferDevice: device_sealed::Sealed {
    fn device(&self) -> &wgpu::Device;
    fn owner(&self) -> Option<std::sync::Arc<wgpu::Device>>;
}

#[cfg(feature = "gpu")]
impl BufferDevice for wgpu::Device {
    fn device(&self) -> &wgpu::Device {
        self
    }
    fn owner(&self) -> Option<std::sync::Arc<wgpu::Device>> {
        None
    }
}

#[cfg(feature = "gpu")]
impl BufferDevice for std::sync::Arc<wgpu::Device> {
    fn device(&self) -> &wgpu::Device {
        self
    }
    fn owner(&self) -> Option<std::sync::Arc<wgpu::Device>> {
        Some(self.clone())
    }
}

/// What the buffer will be used for. Used at GPU allocation time to set the
/// right `wgpu::BufferUsages`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BufferUsage {
    /// Read-only on the device (e.g. `Model` data).
    Storage,
    /// Read-write on the device (e.g. `State` arrays).
    StorageReadWrite,
    /// Small uniform data (constants).
    Uniform,
}

/// Trait implemented by every type that can live in a [`Buffer`].
///
/// Each impl ties the host type to its `f32` GPU mirror.
pub trait BufferElement: Copy + 'static {
    /// `f32` mirror suitable for upload to a wgpu storage buffer.
    type Gpu: bytemuck::Pod + bytemuck::Zeroable + Copy;

    /// Convert one element to its GPU representation.
    fn to_gpu(&self) -> Self::Gpu;

    /// Convert one element back from its GPU representation.
    fn from_gpu(gpu: &Self::Gpu) -> Self;
}

impl BufferElement for Real {
    type Gpu = f32;
    fn to_gpu(&self) -> Self::Gpu {
        *self as f32
    }
    fn from_gpu(gpu: &Self::Gpu) -> Self {
        *gpu as Real
    }
}

impl BufferElement for Vec3 {
    type Gpu = [f32; 4]; // padded to 16 bytes for WGSL
    fn to_gpu(&self) -> Self::Gpu {
        let v = self.as_vec3();
        [v.x, v.y, v.z, 0.0]
    }
    fn from_gpu(gpu: &Self::Gpu) -> Self {
        glam::Vec3::new(gpu[0], gpu[1], gpu[2]).as_dvec3()
    }
}

impl BufferElement for Mat3 {
    type Gpu = GpuMat3;
    fn to_gpu(&self) -> Self::Gpu {
        (*self).into()
    }
    fn from_gpu(gpu: &Self::Gpu) -> Self {
        let c0 = glam::Vec3::new(gpu.col0[0], gpu.col0[1], gpu.col0[2]);
        let c1 = glam::Vec3::new(gpu.col1[0], gpu.col1[1], gpu.col1[2]);
        let c2 = glam::Vec3::new(gpu.col2[0], gpu.col2[1], gpu.col2[2]);
        glam::Mat3::from_cols(c0, c1, c2).as_dmat3()
    }
}

impl BufferElement for Transform {
    type Gpu = GpuTransform;
    fn to_gpu(&self) -> Self::Gpu {
        (*self).into()
    }
    fn from_gpu(gpu: &Self::Gpu) -> Self {
        (*gpu).into()
    }
}

impl BufferElement for molla_math::Quat {
    type Gpu = [f32; 4];
    fn to_gpu(&self) -> Self::Gpu {
        [self.x as f32, self.y as f32, self.z as f32, self.w as f32]
    }
    fn from_gpu(gpu: &Self::Gpu) -> Self {
        molla_math::Quat::from_xyzw(gpu[0] as f64, gpu[1] as f64, gpu[2] as f64, gpu[3] as f64)
    }
}

impl BufferElement for SpatialVector {
    type Gpu = GpuSpatialVector;
    fn to_gpu(&self) -> Self::Gpu {
        (*self).into()
    }
    fn from_gpu(gpu: &Self::Gpu) -> Self {
        (*gpu).into()
    }
}

impl BufferElement for u32 {
    type Gpu = u32;
    fn to_gpu(&self) -> Self::Gpu {
        *self
    }
    fn from_gpu(gpu: &Self::Gpu) -> Self {
        *gpu
    }
}

impl BufferElement for i32 {
    type Gpu = i32;
    fn to_gpu(&self) -> Self::Gpu {
        *self
    }
    fn from_gpu(gpu: &Self::Gpu) -> Self {
        *gpu
    }
}

/// Tracks where the authoritative copy of the data lives.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[allow(
    dead_code,
    reason = "InSync and DeviceDirty are populated by the stage-2 GPU path"
)]
enum SyncState {
    /// Only the host buffer exists (no GPU yet allocated).
    HostOnly,
    /// Host and device hold equal data.
    InSync,
    /// Host has been mutated since the last upload; device is stale.
    HostDirty,
    /// Device has been mutated since the last download; host is stale.
    DeviceDirty,
}

/// A homogeneous array, mirrored across host and device.
///
/// `ensure_device`, `upload`, `download` are all functional under the
/// `gpu` feature; without it only the host half exists.
pub struct Buffer<T: BufferElement> {
    host: Vec<T>,
    usage: BufferUsage,
    state: SyncState,
    #[cfg(feature = "gpu")]
    device: Option<wgpu::Buffer>,
    #[cfg(feature = "gpu")]
    device_owner: Option<std::sync::Arc<wgpu::Device>>,
}

impl<T: BufferElement> Buffer<T> {
    /// Construct a buffer holding `host` on the CPU only. No GPU allocation
    /// happens yet — call [`Self::ensure_device`] to upload.
    #[must_use]
    pub fn from_vec(host: Vec<T>, usage: BufferUsage) -> Self {
        Self {
            host,
            usage,
            state: SyncState::HostOnly,
            #[cfg(feature = "gpu")]
            device: None,
            #[cfg(feature = "gpu")]
            device_owner: None,
        }
    }

    /// Construct a buffer of `len` zero-initialised elements.
    #[must_use]
    pub fn zeros(len: usize, usage: BufferUsage) -> Self
    where
        T: Default,
    {
        Self::from_vec(vec![T::default(); len], usage)
    }

    /// Construct from a slice; copies into a new owned `Vec`.
    #[must_use]
    pub fn from_slice(slice: &[T], usage: BufferUsage) -> Self {
        Self::from_vec(slice.to_vec(), usage)
    }

    /// Number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.host.len()
    }

    /// `true` if the buffer contains zero elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.host.is_empty()
    }

    /// Configured usage class.
    #[must_use]
    pub fn usage(&self) -> BufferUsage {
        self.usage
    }

    /// Read-only access to the host buffer.
    ///
    /// If the device is the authoritative copy, returns an error — call
    /// [`Self::download`] first.
    pub fn host(&self) -> Result<&[T]> {
        match self.state {
            SyncState::DeviceDirty => Err(Error::Storage(
                "host copy is stale; call download() first".into(),
            )),
            _ => Ok(&self.host),
        }
    }

    /// Read-write access to the host buffer. Marks the device copy stale.
    pub fn host_mut(&mut self) -> Result<&mut [T]> {
        if matches!(self.state, SyncState::DeviceDirty) {
            return Err(Error::Storage(
                "host copy is stale; call download() before mutating".into(),
            ));
        }
        self.state = match self.state {
            SyncState::HostOnly => SyncState::HostOnly,
            _ => SyncState::HostDirty,
        };
        Ok(&mut self.host)
    }

    /// Allocate the GPU mirror if not already present, sized for `host.len()`
    /// elements of `T::Gpu`. Does not upload — call [`Self::upload`] after.
    /// An `Arc<Device>` records canonical ownership at allocation; a raw device
    /// leaves ownership unverified. Existing mirrors never acquire a new owner.
    #[cfg(feature = "gpu")]
    pub fn ensure_device<D: BufferDevice + ?Sized>(&mut self, device: &D) -> Result<()> {
        let owner = device.owner();
        if self.device.is_some() {
            if let Some(requested) = &owner {
                if self
                    .device_owner
                    .as_ref()
                    .is_some_and(|existing| !std::sync::Arc::ptr_eq(existing, requested))
                {
                    return Err(Error::Storage(
                        "buffer belongs to a different owned device".into(),
                    ));
                }
            }
            return Ok(());
        }
        if self.host.is_empty() {
            // wgpu requires non-zero buffer size; nothing useful to allocate.
            return Ok(());
        }

        let usage = match self.usage {
            BufferUsage::Storage => {
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST
            }
            BufferUsage::StorageReadWrite => {
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST
            }
            BufferUsage::Uniform => {
                wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST
            }
        };

        let size = (self.host.len() * core::mem::size_of::<T::Gpu>()) as u64;
        let buf = device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::Buffer"),
            size,
            usage,
            mapped_at_creation: false,
        });
        self.device = Some(buf);
        self.device_owner = owner;
        // Host is the authoritative copy until we upload.
        self.state = SyncState::HostDirty;
        Ok(())
    }

    /// Upload host data to the device buffer. No-op if no device buffer
    /// is allocated.
    #[cfg(feature = "gpu")]
    pub fn upload(&mut self, queue: &wgpu::Queue) -> Result<()> {
        self.upload_with_scratch(queue, &mut Vec::new())
    }

    /// Upload host values through caller-retained packed storage; capacity grows only when needed.
    /// Missing/empty device mirrors leave scratch unchanged. Queue staging is separate.
    #[cfg(feature = "gpu")]
    pub fn upload_with_scratch(
        &mut self,
        queue: &wgpu::Queue,
        scratch: &mut Vec<T::Gpu>,
    ) -> Result<()> {
        let Some(buf) = self.device.as_ref() else {
            return Ok(());
        };
        if self.host.is_empty() {
            return Ok(());
        }
        debug_assert!(
            !matches!(self.state, SyncState::DeviceDirty),
            "Buffer::upload would overwrite device-dirty data with stale \
             host data — call download() before re-uploading"
        );
        scratch.clear();
        scratch.extend(self.host.iter().map(BufferElement::to_gpu));
        queue.write_buffer(buf, 0, bytemuck::cast_slice(scratch));
        self.state = SyncState::InSync;
        Ok(())
    }

    /// Download device data into the host buffer. Blocks until the
    /// transfer completes (uses `pollster`).
    #[cfg(feature = "gpu")]
    pub fn download(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<()> {
        let Some(buf) = self.device.as_ref() else {
            return Ok(());
        };
        if self.host.is_empty() {
            return Ok(());
        }
        // Downloading while the host copy is dirty would silently
        // clobber un-uploaded host mutations with stale device data.
        debug_assert!(
            !matches!(self.state, SyncState::HostDirty),
            "Buffer::download would overwrite host-dirty data with stale \
             device data — call upload() before downloading"
        );

        let elem_size = core::mem::size_of::<T::Gpu>();
        let byte_size = (self.host.len() * elem_size) as u64;

        // Staging buffer that supports MAP_READ.
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::Buffer staging"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, byte_size);
        queue.submit(std::iter::once(encoder.finish()));

        // Map and wait.
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("device poll: {e:?}")))?;
        rx.recv()
            .map_err(|e| Error::Gpu(format!("download recv: {e}")))?
            .map_err(|e| Error::Gpu(format!("buffer map: {e:?}")))?;

        let mapped = slice.get_mapped_range();
        let gpu_slice: &[T::Gpu] = bytemuck::cast_slice(&mapped);
        for (host, gpu) in self.host.iter_mut().zip(gpu_slice.iter()) {
            *host = T::from_gpu(gpu);
        }
        drop(mapped);
        staging.unmap();

        self.state = SyncState::InSync;
        Ok(())
    }

    /// Borrow the underlying `wgpu::Buffer` if it's been allocated.
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn device_buffer(&self) -> Option<&wgpu::Buffer> {
        self.device.as_ref()
    }

    /// Borrow an uploaded device mirror without accepting stale host edits.
    #[cfg(feature = "gpu")]
    pub fn device_buffer_current(&self) -> Result<&wgpu::Buffer> {
        if matches!(self.state, SyncState::HostOnly | SyncState::HostDirty) {
            return Err(Error::Storage(
                "device copy is stale; allocate and upload before GPU consumption".into(),
            ));
        }
        self.device
            .as_ref()
            .ok_or_else(|| Error::Storage("buffer has no device mirror".into()))
    }

    /// Borrow a current mirror allocated on the specified logical device.
    /// Requires the same canonical `Arc<Device>` used at allocation.
    #[cfg(feature = "gpu")]
    pub fn device_buffer_for(
        &self,
        device: &std::sync::Arc<wgpu::Device>,
    ) -> Result<&wgpu::Buffer> {
        if self
            .device_owner
            .as_ref()
            .is_none_or(|owner| !std::sync::Arc::ptr_eq(owner, device))
        {
            return Err(Error::Storage(
                "buffer device ownership is foreign or unverified".into(),
            ));
        }
        self.device_buffer_current()
    }

    /// Mark the device copy as authoritative (used after a kernel writes
    /// to the buffer in place). Subsequent [`Self::host`] calls will
    /// trigger an error until [`Self::download`] is invoked.
    #[cfg(feature = "gpu")]
    pub fn mark_device_dirty(&mut self) {
        self.state = SyncState::DeviceDirty;
    }
}

impl<T: BufferElement + core::fmt::Debug> core::fmt::Debug for Buffer<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Buffer")
            .field("len", &self.host.len())
            .field("usage", &self.usage)
            .field("state", &self.state)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_vec_round_trips() {
        let v = vec![1u32, 2, 3, 4];
        let b: Buffer<u32> = Buffer::from_vec(v.clone(), BufferUsage::Storage);
        assert_eq!(b.len(), 4);
        assert_eq!(b.host().unwrap(), v.as_slice());
    }

    #[test]
    fn host_mut_marks_dirty() {
        let mut b: Buffer<u32> = Buffer::from_vec(vec![0; 3], BufferUsage::StorageReadWrite);
        b.host_mut().unwrap()[0] = 42;
        assert_eq!(b.host().unwrap()[0], 42);
    }

    #[test]
    fn zeros_uses_default() {
        let b: Buffer<Real> = Buffer::zeros(8, BufferUsage::Storage);
        assert_eq!(b.len(), 8);
        assert!(b.host().unwrap().iter().all(|&v| v == 0.0));
    }
}
