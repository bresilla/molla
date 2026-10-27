//! Fixed-array storage budgets for IMU, contact and frame-transform sensors.

use molla_core::{Error, Result};
use molla_math::{Quat, Real, SpatialVector, Vec3};
use molla_storage::BufferElement;

/// Combined retained host/device array payload, excluding inputs, bindings and output pools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SensorArrayLimits {
    pub max_storage_bytes: u64,
}

impl Default for SensorArrayLimits {
    fn default() -> Self {
        Self {
            max_storage_bytes: 1 << 30,
        }
    }
}

/// Retained array payload; excludes transient packing, driver staging and collection.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SensorArrayStorage {
    pub host_bytes: u64,
    pub device_bytes: u64,
    pub total_bytes: u64,
    pub largest_device_buffer_bytes: u64,
}

impl SensorArrayStorage {
    /// Mount body/pose arrays, two output channels and per-body velocity history.
    pub fn for_imu(mounts: usize, bodies: usize) -> Result<Self> {
        if mounts == 0 {
            return Err(Error::Solver("imu: empty mount list".into()));
        }
        let mounts = checked_count(mounts)?;
        let bodies = checked_count(bodies)?;
        let mut usage = Self::default();
        usage.add::<u32>(mounts, 1);
        usage.add::<Vec3>(mounts, 3);
        usage.add::<Quat>(mounts, 1);
        usage.add::<SpatialVector>(bodies, 1);
        Ok(usage)
    }

    /// Four per-body output channels; an empty model has no array payload.
    pub fn for_contact(bodies: usize) -> Result<Self> {
        let bodies = checked_count(bodies)?;
        let mut usage = Self::default();
        usage.add::<u32>(bodies, 1);
        usage.add::<Real>(bodies, 1);
        usage.add::<Vec3>(bodies, 2);
        Ok(usage)
    }

    /// Target body/pose arrays and four outputs, including the optional reference slot.
    pub fn for_frames(targets: usize, has_reference: bool) -> Result<Self> {
        if targets == 0 {
            return Err(Error::Solver("frame_transform: empty target list".into()));
        }
        let count = targets
            .checked_add(usize::from(has_reference))
            .ok_or_else(|| Error::Solver("frame_transform: target count overflow".into()))?;
        let count = checked_count(count)?;
        let mut usage = Self::default();
        usage.add::<u32>(count, 1);
        usage.add::<Vec3>(count, 3);
        usage.add::<Quat>(count, 3);
        Ok(usage)
    }

    fn add<T: BufferElement>(&mut self, count: u32, arrays: u64) {
        let host = u64::from(count) * size_of::<T>() as u64;
        let device = u64::from(count) * size_of::<T::Gpu>() as u64;
        self.host_bytes += host * arrays;
        self.device_bytes += device * arrays;
        self.total_bytes = self.host_bytes + self.device_bytes;
        self.largest_device_buffer_bytes = self.largest_device_buffer_bytes.max(device);
    }
}

fn checked_count(count: usize) -> Result<u32> {
    u32::try_from(count).map_err(|_| Error::Solver("sensor array count exceeds u32".into()))
}

impl SensorArrayLimits {
    pub(crate) fn check(
        self,
        storage: SensorArrayStorage,
        invocations: u32,
        largest_input_bytes: u64,
        device: &wgpu::Limits,
    ) -> Result<()> {
        let largest = storage.largest_device_buffer_bytes.max(largest_input_bytes);
        if largest > device.max_buffer_size
            || largest > device.max_storage_buffer_binding_size
            || invocations.div_ceil(64) > device.max_compute_workgroups_per_dimension
        {
            return Err(Error::Solver("sensor arrays exceed GPU limits".into()));
        }
        if storage.total_bytes > self.max_storage_bytes {
            return Err(Error::Solver(format!(
                "sensor arrays require {} bytes, budget is {}",
                storage.total_bytes, self.max_storage_bytes
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn buffer_bytes<T: BufferElement>(buffer: &molla_storage::Buffer<T>) -> (u64, u64) {
        (
            std::mem::size_of_val(buffer.host().unwrap()) as u64,
            buffer.device_buffer().map_or(0, wgpu::Buffer::size),
        )
    }

    pub(crate) fn assert_payload(usage: SensorArrayStorage, buffers: &[(u64, u64)]) {
        assert_eq!(usage.host_bytes, buffers.iter().map(|b| b.0).sum::<u64>());
        assert_eq!(usage.device_bytes, buffers.iter().map(|b| b.1).sum::<u64>());
        assert_eq!(
            usage.largest_device_buffer_bytes,
            buffers.iter().map(|b| b.1).max().unwrap_or(0)
        );
        assert_eq!(usage.total_bytes, usage.host_bytes + usage.device_bytes);
    }

    #[test]
    fn fixed_sensor_arrays_count_history_outputs_and_reference() {
        let imu = SensorArrayStorage::for_imu(3, 5).unwrap();
        assert_eq!(
            imu.host_bytes,
            3 * (4 + 3 * size_of::<Vec3>() + size_of::<Quat>()) as u64
                + 5 * size_of::<SpatialVector>() as u64
        );
        assert_eq!(imu.device_bytes, 3 * (4 + 3 * 16 + 16) + 5 * 32);
        assert_eq!(imu.largest_device_buffer_bytes, 5 * 32);
        let contacts = SensorArrayStorage::for_contact(5).unwrap();
        assert_eq!(
            contacts.host_bytes,
            5 * (4 + size_of::<Real>() + 2 * size_of::<Vec3>()) as u64
        );
        assert_eq!(contacts.device_bytes, 5 * 40);
        let frames = SensorArrayStorage::for_frames(3, true).unwrap();
        assert_eq!(frames, SensorArrayStorage::for_frames(4, false).unwrap());
        assert_eq!(
            frames.host_bytes,
            4 * (4 + 3 * size_of::<Vec3>() + 3 * size_of::<Quat>()) as u64
        );
        assert_eq!(frames.device_bytes, 4 * (4 + 6 * 16));
        for usage in [imu, contacts, frames] {
            assert_eq!(usage.total_bytes, usage.host_bytes + usage.device_bytes);
        }
        assert_eq!(
            SensorArrayStorage::for_contact(0).unwrap(),
            SensorArrayStorage::default()
        );
    }

    #[test]
    fn fixed_sensor_array_limits_reject_extremes_without_allocation() {
        assert!(SensorArrayStorage::for_imu(0, 1).is_err());
        assert!(SensorArrayStorage::for_imu(1, usize::MAX).is_err());
        assert!(SensorArrayStorage::for_contact(usize::MAX).is_err());
        assert!(SensorArrayStorage::for_frames(0, true).is_err());
        assert!(SensorArrayStorage::for_frames(usize::MAX, true).is_err());
        assert!(SensorArrayStorage::for_frames(u32::MAX as usize, true).is_err());
        let device = wgpu::Limits::default();
        for usage in [
            SensorArrayStorage::for_imu(65, 1).unwrap(),
            SensorArrayStorage::for_contact(65).unwrap(),
            SensorArrayStorage::for_frames(64, true).unwrap(),
        ] {
            let exact = SensorArrayLimits {
                max_storage_bytes: usage.total_bytes,
            };
            assert!(exact.check(usage, 65, 0, &device).is_ok());
            assert!(
                SensorArrayLimits {
                    max_storage_bytes: usage.total_bytes - 1
                }
                .check(usage, 65, 0, &device)
                .is_err()
            );
            assert!(
                exact
                    .check(
                        usage,
                        65,
                        0,
                        &wgpu::Limits {
                            max_buffer_size: usage.largest_device_buffer_bytes - 1,
                            ..device.clone()
                        }
                    )
                    .is_err()
            );
            assert!(
                exact
                    .check(
                        usage,
                        65,
                        0,
                        &wgpu::Limits {
                            max_storage_buffer_binding_size: usage.largest_device_buffer_bytes - 1,
                            ..device.clone()
                        }
                    )
                    .is_err()
            );
            assert!(
                exact
                    .check(
                        usage,
                        65,
                        0,
                        &wgpu::Limits {
                            max_compute_workgroups_per_dimension: 1,
                            ..device.clone()
                        }
                    )
                    .is_err()
            );
            assert!(
                exact
                    .check(usage, 65, device.max_buffer_size + 1, &device)
                    .is_err()
            );
        }
        for usage in [
            SensorArrayStorage::for_imu(u32::MAX as usize, u32::MAX as usize).unwrap(),
            SensorArrayStorage::for_contact(u32::MAX as usize).unwrap(),
            SensorArrayStorage::for_frames(u32::MAX as usize, false).unwrap(),
        ] {
            assert!(
                SensorArrayLimits::default()
                    .check(usage, u32::MAX, 0, &device)
                    .is_err()
            );
        }
    }
}
