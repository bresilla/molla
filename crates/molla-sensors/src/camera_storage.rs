//! Fixed camera array budgets, including shared ray storage and reserved aperture scratch.

use crate::{RayStorageUsage, RaycasterLimits};
use molla_core::{Error, Result};
use molla_math::{Real, Transform, Vec3};
use molla_storage::BufferElement;

/// Combined host/device array payload; excludes scene, bindings, driver staging and output pools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TiledCameraLimits {
    pub ray_limits: RaycasterLimits,
    pub max_storage_bytes: u64,
}

impl Default for TiledCameraLimits {
    fn default() -> Self {
        Self {
            ray_limits: RaycasterLimits::default(),
            max_storage_bytes: 1 << 30,
        }
    }
}

/// Fixed host/device arrays, with lazy aperture sums and their uniform reserved in the estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CameraStorageUsage {
    pub pixels: u32,
    pub cameras: u32,
    pub host_bytes: u64,
    pub device_bytes: u64,
    pub aperture_device_bytes: u64,
    pub total_bytes: u64,
}

impl CameraStorageUsage {
    /// Estimate all five channels, independent of the active channel mask or aperture settings.
    pub fn for_dimensions(width: u32, height: u32, cameras: usize) -> Result<Self> {
        let cameras = u32::try_from(cameras)
            .map_err(|_| Error::Solver("camera: camera count exceeds u32".into()))?;
        let pixels = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(cameras))
            .filter(|&n| n != 0)
            .ok_or_else(|| Error::Solver("camera: pixel count is zero or exceeds u32".into()))?;
        let rays = RayStorageUsage::for_capacity(pixels)?;
        let host_stride = size_of::<Real>() + size_of::<Vec3>() + 3 * size_of::<u32>();
        let device_stride = size_of::<<Real as BufferElement>::Gpu>()
            + size_of::<<Vec3 as BufferElement>::Gpu>()
            + 3 * size_of::<u32>();
        let pose_gpu = size_of::<<Transform as BufferElement>::Gpu>();
        let aperture_device_bytes = u64::from(pixels) * 16 + 16;
        let host_bytes = rays.host_bytes
            + u64::from(pixels) * host_stride as u64
            + u64::from(cameras) * (size_of::<Transform>() + pose_gpu) as u64
            + size_of::<Vec3>() as u64;
        let device_bytes = rays.device_bytes
            + u64::from(pixels) * device_stride as u64
            + u64::from(cameras) * pose_gpu as u64
            + size_of::<<Vec3 as BufferElement>::Gpu>() as u64
            + aperture_device_bytes;
        Ok(Self {
            pixels,
            cameras,
            host_bytes,
            device_bytes,
            aperture_device_bytes,
            total_bytes: host_bytes + device_bytes,
        })
    }
}

impl TiledCameraLimits {
    pub(crate) fn check(
        self,
        width: u32,
        height: u32,
        cameras: usize,
        device: &wgpu::Limits,
    ) -> Result<CameraStorageUsage> {
        crate::tiled_camera::checked_pixel_count(width, height, cameras, device)?;
        let usage = CameraStorageUsage::for_dimensions(width, height, cameras)?;
        crate::raycast::checked_storage(usage.pixels, self.ray_limits, device)?;
        if usage.total_bytes > self.max_storage_bytes {
            return Err(Error::Solver(format!(
                "camera: array storage requires {} bytes, budget is {}",
                usage.total_bytes, self.max_storage_bytes
            )));
        }
        Ok(usage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_storage_counts_pixels_poses_rays_and_reserved_optics() {
        for (w, h, c) in [(1, 1, 1), (640, 480, 4), (4096, 4096, 1)] {
            let usage = CameraStorageUsage::for_dimensions(w, h, c).unwrap();
            let p = u64::from(w) * u64::from(h) * c as u64;
            let ray = RayStorageUsage::for_capacity(p as u32).unwrap();
            assert_eq!(
                usage.host_bytes,
                ray.host_bytes
                    + p * (size_of::<Real>() + size_of::<Vec3>() + 12) as u64
                    + c as u64 * (size_of::<Transform>() + 32) as u64
                    + size_of::<Vec3>() as u64
            );
            assert_eq!(
                usage.device_bytes,
                ray.device_bytes + 48 * p + 32 * c as u64 + 32
            );
            assert_eq!(usage.total_bytes, usage.host_bytes + usage.device_bytes);
            assert_eq!(usage.aperture_device_bytes, 16 * p + 16);
        }
        for (w, h, c) in [
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
            (u32::MAX, 2, 1),
            (1, 1, usize::MAX),
        ] {
            assert!(CameraStorageUsage::for_dimensions(w, h, c).is_err());
        }
    }

    #[test]
    fn camera_array_limit_and_device_limits_precede_allocation() {
        let device = wgpu::Limits::default();
        let usage = CameraStorageUsage::for_dimensions(64, 32, 2).unwrap();
        let exact = TiledCameraLimits {
            max_storage_bytes: usage.total_bytes,
            ..Default::default()
        };
        assert_eq!(exact.check(64, 32, 2, &device).unwrap(), usage);
        assert!(
            TiledCameraLimits {
                max_storage_bytes: usage.total_bytes - 1,
                ..exact
            }
            .check(64, 32, 2, &device)
            .is_err()
        );
        assert!(
            TiledCameraLimits {
                ray_limits: RaycasterLimits {
                    max_ray_storage_bytes: 0
                },
                ..exact
            }
            .check(64, 32, 2, &device)
            .is_err()
        );
        assert!(
            exact
                .check(
                    64,
                    32,
                    2,
                    &wgpu::Limits {
                        max_buffer_size: 32,
                        ..device.clone()
                    }
                )
                .is_err()
        );
        assert!(
            exact
                .check(
                    64,
                    32,
                    2,
                    &wgpu::Limits {
                        max_compute_workgroups_per_dimension: 1,
                        ..device
                    }
                )
                .is_err()
        );
        assert!(
            TiledCameraLimits::default()
                .check(1280, 720, 4, &wgpu::Limits::default())
                .is_err()
        );
        assert!(CameraStorageUsage::for_dimensions(u32::MAX, 1, 1).is_ok());
    }
}
