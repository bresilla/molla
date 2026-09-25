//! Retained batched-LiDAR arrays and per-call legacy CPU layout budgets.

use crate::{LidarScanRange, RayStorageUsage, RaycasterLimits};
use molla_core::{Error, Result};
use molla_kernels::LidarScanDescriptor;

/// Per-call CPU ray-layout payload for legacy terrestrial/aerial scans, excluding hit collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LidarLayoutLimits {
    pub max_layout_bytes: u64,
}

impl Default for LidarLayoutLimits {
    fn default() -> Self {
        Self {
            max_layout_bytes: 1 << 30,
        }
    }
}

impl LidarLayoutLimits {
    pub(crate) fn check(self, bytes: u64) -> Result<()> {
        if bytes > self.max_layout_bytes {
            return Err(Error::Solver(format!(
                "lidar layout: requires {bytes} bytes, budget is {}",
                self.max_layout_bytes
            )));
        }
        Ok(())
    }
}

/// Combined retained host/device arrays, with an independent shared-ray sub-budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LidarBatchLimits {
    pub ray_limits: RaycasterLimits,
    pub max_storage_bytes: u64,
}

impl Default for LidarBatchLimits {
    fn default() -> Self {
        Self {
            ray_limits: RaycasterLimits::default(),
            max_storage_bytes: 1 << 30,
        }
    }
}

/// Ray arrays, three return-index/count arrays and reserved scan descriptors/ranges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LidarBatchStorage {
    pub host_bytes: u64,
    pub device_bytes: u64,
    pub total_bytes: u64,
    pub descriptor_device_bytes: u64,
}

impl LidarBatchStorage {
    /// Estimate fixed capacity, independent of active scans or sub-rays per pulse.
    pub fn for_capacity(max_scans: usize, max_rays: u32) -> Result<Self> {
        if max_scans == 0 || max_scans > 1024 || max_rays == 0 || max_rays > i32::MAX as u32 {
            return Err(Error::Solver(
                "lidar batch: invalid scan/ray capacity".into(),
            ));
        }
        let rays = RayStorageUsage::for_capacity(max_rays)?;
        let descriptor_device_bytes = max_scans as u64 * size_of::<LidarScanDescriptor>() as u64;
        let host_bytes = rays.host_bytes
            + u64::from(max_rays) * 12
            + max_scans as u64
                * (size_of::<LidarScanDescriptor>() + size_of::<LidarScanRange>()) as u64;
        let device_bytes = rays.device_bytes + u64::from(max_rays) * 12 + descriptor_device_bytes;
        Ok(Self {
            host_bytes,
            device_bytes,
            total_bytes: host_bytes + device_bytes,
            descriptor_device_bytes,
        })
    }
}

impl LidarBatchLimits {
    pub(crate) fn check(
        self,
        max_scans: usize,
        max_rays: u32,
        device: &wgpu::Limits,
    ) -> Result<LidarBatchStorage> {
        let storage = LidarBatchStorage::for_capacity(max_scans, max_rays)?;
        crate::raycast::checked_storage(max_rays, self.ray_limits, device)?;
        if storage.descriptor_device_bytes > device.max_buffer_size
            || storage.descriptor_device_bytes > device.max_storage_buffer_binding_size
        {
            return Err(Error::Solver(
                "lidar batch: descriptors exceed GPU capacity".into(),
            ));
        }
        if storage.total_bytes > self.max_storage_bytes {
            return Err(Error::Solver(format!(
                "lidar batch: arrays require {} bytes, budget is {}",
                storage.total_bytes, self.max_storage_bytes
            )));
        }
        Ok(storage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lidar_storage_counts_return_arrays_and_reserved_scan_capacity() {
        for (scans, rays) in [(1, 1), (3, 65), (1024, 100_000)] {
            let storage = LidarBatchStorage::for_capacity(scans, rays).unwrap();
            let shared = RayStorageUsage::for_capacity(rays).unwrap();
            assert_eq!(size_of::<LidarScanDescriptor>(), 128);
            assert_eq!(
                storage.host_bytes,
                shared.host_bytes
                    + u64::from(rays) * 12
                    + scans as u64 * (128 + size_of::<LidarScanRange>() as u64)
            );
            assert_eq!(
                storage.device_bytes,
                shared.device_bytes + u64::from(rays) * 12 + scans as u64 * 128
            );
            assert_eq!(
                storage.total_bytes,
                storage.host_bytes + storage.device_bytes
            );
        }
        for (scans, rays) in [(0, 1), (1, 0), (1025, 1), (usize::MAX, 1), (1, u32::MAX)] {
            assert!(LidarBatchStorage::for_capacity(scans, rays).is_err());
        }
        assert!(LidarBatchStorage::for_capacity(1024, i32::MAX as u32).is_ok());
    }

    #[test]
    fn lidar_combined_budget_rejects_when_ray_subbudget_alone_fits() {
        let device = wgpu::Limits::default();
        let storage = LidarBatchStorage::for_capacity(1024, 1).unwrap();
        let exact = LidarBatchLimits {
            max_storage_bytes: storage.total_bytes,
            ..Default::default()
        };
        assert_eq!(exact.check(1024, 1, &device).unwrap(), storage);
        assert!(
            LidarBatchLimits {
                max_storage_bytes: storage.total_bytes - 1,
                ..exact
            }
            .check(1024, 1, &device)
            .is_err()
        );
        assert!(
            LidarBatchLimits {
                ray_limits: RaycasterLimits {
                    max_ray_storage_bytes: 0
                },
                ..exact
            }
            .check(1024, 1, &device)
            .is_err()
        );
        for device in [
            wgpu::Limits {
                max_buffer_size: storage.descriptor_device_bytes - 1,
                ..device.clone()
            },
            wgpu::Limits {
                max_storage_buffer_binding_size: storage.descriptor_device_bytes - 1,
                ..device.clone()
            },
            wgpu::Limits {
                max_compute_workgroups_per_dimension: 0,
                ..device.clone()
            },
        ] {
            assert!(exact.check(1024, 1, &device).is_err());
        }
        let ray_bytes = RayStorageUsage::for_capacity(1).unwrap().total_bytes;
        assert!(
            LidarBatchLimits {
                max_storage_bytes: ray_bytes,
                ..exact
            }
            .check(1, 1, &device)
            .is_err()
        );
        assert!(
            LidarBatchLimits::default()
                .check(1024, i32::MAX as u32, &device)
                .is_err()
        );
    }
}
