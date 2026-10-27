//! Construction payload budgets for GPU image, annotation and cloud processors.

use crate::{
    AnnotationConfig, CloudGridCell, CloudGridConfig, CloudPoint, CloudProcessorConfig,
    DeviceBoundingBox, ImageProcessorConfig,
};
use molla_core::{Error, Result};
use molla_math::Vec3;
use molla_storage::BufferElement;

/// Retained host/device arrays plus one upload-packing array, excluding driver staging,
/// allocator overhead, pipeline/binding metadata, frame metadata and output/readback pools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessorStorageLimits {
    pub max_construction_bytes: u64,
}

impl Default for ProcessorStorageLimits {
    fn default() -> Self {
        Self {
            max_construction_bytes: 1 << 30,
        }
    }
}

/// Requested payload bytes; construction adds the largest sequential upload conversion.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ProcessorStorageUsage {
    pub host_bytes: u64,
    pub device_bytes: u64,
    pub upload_scratch_bytes: u64,
    pub construction_bytes: u64,
}

impl ProcessorStorageUsage {
    /// Validate capacities and estimate image output, gains, reduction and aligned uniforms.
    pub fn for_image(config: ImageProcessorConfig, device: &wgpu::Limits) -> Result<Self> {
        let mut usage = Self::device(crate::image_processing::checked_budget(config, device)?);
        usage.add_host::<u32>(u64::from(config.max_pixels))?;
        usage.add_host::<Vec3>(
            u64::from(config.max_images) * u64::from(config.max_operations_per_image),
        )?;
        Ok(usage)
    }

    /// Validate capacities and estimate shapes, classes, boxes, labels and working arrays.
    pub fn for_annotation(config: AnnotationConfig, device: &wgpu::Limits) -> Result<Self> {
        let mut usage = Self::device(crate::annotation::checked_budget(config, device)?);
        usage.add_host::<u32>(u64::from(config.max_pixels))?;
        usage.add_host::<u32>(u64::from(config.max_pixels))?;
        usage.add_host::<DeviceBoundingBox>(
            u64::from(config.max_images) * u64::from(config.max_labels),
        )?;
        usage.add_host::<u32>(u64::from(config.max_images) * 4)?;
        Ok(usage)
    }

    /// Validate capacities and estimate input, working/output records, counts and compaction.
    pub fn for_cloud(config: CloudProcessorConfig, device: &wgpu::Limits) -> Result<Self> {
        let mut usage = Self::device(crate::cloud_processing::checked_budget(config, device)?);
        usage.add_host::<CloudPoint>(u64::from(config.max_points))?;
        usage.add_host::<CloudPoint>(u64::from(config.max_points))?;
        usage.add_host::<u32>(1 + u64::from(config.max_jobs) * 5)?;
        Ok(usage)
    }

    /// Validate capacities and estimate points, indices, summaries and grid work arrays.
    pub fn for_grid(config: CloudGridConfig, device: &wgpu::Limits) -> Result<Self> {
        let mut usage = Self::device(crate::cloud_grid::checked_budget(config, device)?);
        usage.add_host::<CloudPoint>(u64::from(config.max_points))?;
        usage.add_host::<u32>(u64::from(config.max_points))?;
        usage.add_host::<CloudGridCell>(u64::from(config.max_cells))?;
        usage.add_host::<u32>(6)?;
        Ok(usage)
    }

    fn device(device_bytes: u64) -> Self {
        Self {
            device_bytes,
            construction_bytes: device_bytes,
            ..Default::default()
        }
    }

    fn add_host<T: BufferElement>(&mut self, count: u64) -> Result<()> {
        let overflow = || Error::Storage("processor construction payload overflow".into());
        let host = count
            .checked_mul(size_of::<T>() as u64)
            .ok_or_else(overflow)?;
        let packed = count
            .checked_mul(size_of::<T::Gpu>() as u64)
            .ok_or_else(overflow)?;
        self.host_bytes = self.host_bytes.checked_add(host).ok_or_else(overflow)?;
        self.upload_scratch_bytes = self.upload_scratch_bytes.max(packed);
        self.construction_bytes = self
            .host_bytes
            .checked_add(self.device_bytes)
            .and_then(|bytes| bytes.checked_add(self.upload_scratch_bytes))
            .ok_or_else(overflow)?;
        Ok(())
    }
}

impl ProcessorStorageLimits {
    pub(crate) fn check(self, usage: ProcessorStorageUsage) -> Result<()> {
        if usage.construction_bytes > self.max_construction_bytes {
            return Err(Error::Storage(format!(
                "processor construction requires {} payload bytes, budget is {}",
                usage.construction_bytes, self.max_construction_bytes
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn processor_estimates_include_host_arrays_and_peak_upload_conversion() {
        let device = wgpu::Limits::default();
        let image = ProcessorStorageUsage::for_image(
            ImageProcessorConfig {
                max_pixels: 257,
                max_images: 3,
                max_operations_per_image: 2,
                max_buffer_bytes: u64::MAX,
            },
            &device,
        )
        .unwrap();
        assert_eq!(image.host_bytes, 4 * 257 + 6 * size_of::<Vec3>() as u64);
        assert_eq!(image.device_bytes, 4 * 257 + 2 * 16 + 6 * 16 + 6 * 256);
        assert_eq!(image.upload_scratch_bytes, 4 * 257);
        let annotation = ProcessorStorageUsage::for_annotation(
            AnnotationConfig {
                max_pixels: 257,
                max_images: 3,
                max_labels: 5,
                max_buffer_bytes: u64::MAX,
            },
            &device,
        )
        .unwrap();
        assert_eq!(annotation.host_bytes, 8 * 257 + 32 * 15 + 16 * 3);
        assert_eq!(
            annotation.device_bytes,
            8 * 257 + 64 * 15 + 8 * 5 + 32 * 3 + 16
        );
        assert_eq!(annotation.upload_scratch_bytes, 4 * 257);
        let cloud = ProcessorStorageUsage::for_cloud(
            CloudProcessorConfig {
                max_points: 257,
                max_jobs: 3,
                max_buffer_bytes: u64::MAX,
            },
            &device,
        )
        .unwrap();
        assert_eq!(cloud.host_bytes, 224 * 257 + 16 * 4);
        assert_eq!(
            cloud.device_bytes,
            288 * 257 + 2 * 8 + 3 * 160 + 16 * 4 + 16
        );
        assert_eq!(cloud.upload_scratch_bytes, 112 * 257);
        let grid = ProcessorStorageUsage::for_grid(
            CloudGridConfig {
                max_points: 257,
                max_cells: 17,
                max_buffer_bytes: u64::MAX,
            },
            &device,
        )
        .unwrap();
        assert_eq!(grid.host_bytes, 116 * 257 + 64 * 17 + 24);
        assert_eq!(grid.device_bytes, 120 * 257 + 76 * 17 + 8 + 104);
        assert_eq!(grid.upload_scratch_bytes, 112 * 257);
        for usage in [image, annotation, cloud, grid] {
            assert_eq!(
                usage.construction_bytes,
                usage.host_bytes + usage.device_bytes + usage.upload_scratch_bytes
            );
            assert!(
                ProcessorStorageLimits {
                    max_construction_bytes: usage.construction_bytes
                }
                .check(usage)
                .is_ok()
            );
            assert!(
                ProcessorStorageLimits {
                    max_construction_bytes: usage.construction_bytes - 1
                }
                .check(usage)
                .is_err()
            );
        }
    }

    #[test]
    fn host_and_scratch_can_exceed_budget_when_device_arrays_fit() {
        let device = wgpu::Limits {
            max_buffer_size: 1 << 40,
            max_storage_buffer_binding_size: 1 << 40,
            ..Default::default()
        };
        let usage = ProcessorStorageUsage::for_cloud(
            CloudProcessorConfig {
                max_points: 2_000_000,
                max_jobs: 1,
                max_buffer_bytes: 1 << 30,
            },
            &device,
        )
        .unwrap();
        assert!(usage.device_bytes < 1 << 30);
        assert!(usage.construction_bytes > 1 << 30);
        assert!(ProcessorStorageLimits::default().check(usage).is_err());
        assert!(
            ProcessorStorageLimits {
                max_construction_bytes: usage.construction_bytes
            }
            .check(usage)
            .is_ok()
        );
    }

    #[test]
    fn invalid_capacities_and_alignment_fail_without_allocating_payloads() {
        let device = wgpu::Limits::default();
        for count in [0, u32::MAX] {
            assert!(
                ProcessorStorageUsage::for_image(
                    ImageProcessorConfig {
                        max_pixels: count,
                        ..Default::default()
                    },
                    &device
                )
                .is_err()
            );
            assert!(
                ProcessorStorageUsage::for_annotation(
                    AnnotationConfig {
                        max_pixels: count,
                        ..Default::default()
                    },
                    &device
                )
                .is_err()
            );
            assert!(
                ProcessorStorageUsage::for_cloud(
                    CloudProcessorConfig {
                        max_points: count,
                        ..Default::default()
                    },
                    &device
                )
                .is_err()
            );
            assert!(
                ProcessorStorageUsage::for_grid(
                    CloudGridConfig {
                        max_points: count,
                        ..Default::default()
                    },
                    &device
                )
                .is_err()
            );
        }
        for alignment in [0, 3] {
            assert!(
                ProcessorStorageUsage::for_image(
                    ImageProcessorConfig {
                        max_pixels: 1,
                        ..Default::default()
                    },
                    &wgpu::Limits {
                        min_uniform_buffer_offset_alignment: alignment,
                        ..device.clone()
                    }
                )
                .is_err()
            );
        }
    }

    pub(crate) fn check_live(usage: ProcessorStorageUsage, arrays: &[(u64, u64)]) {
        assert_eq!(usage.host_bytes, arrays.iter().map(|v| v.0).sum::<u64>());
        assert_eq!(
            usage.upload_scratch_bytes,
            arrays.iter().map(|v| v.1).max().unwrap()
        );
        assert!(usage.device_bytes >= arrays.iter().map(|v| v.1).sum::<u64>());
    }
}
