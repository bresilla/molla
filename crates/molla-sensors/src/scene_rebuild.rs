//! Bounded periodic or GPU-quality-triggered spatial reordering for owned-scene queries.

use molla_core::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SceneRebuildPolicy {
    /// Reorder or check quality every N encoded queries, starting with the first; zero disables it.
    pub interval: u32,
    /// Rebuild above this percentage of the best normalized area since rebuilding; None is periodic.
    pub quality_ratio_percent: Option<u32>,
    /// Scenes below this finite-leaf count only refit their original topology.
    pub min_leaves: u32,
    /// Additional fixed GPU payload; excludes the base hierarchy, host data and pipelines.
    pub max_device_bytes: u64,
}

impl Default for SceneRebuildPolicy {
    fn default() -> Self {
        Self {
            interval: 64,
            quality_ratio_percent: None,
            min_leaves: 64,
            max_device_bytes: 128 << 20,
        }
    }
}

impl SceneRebuildPolicy {
    pub(crate) fn validate(self) -> Result<()> {
        if self.quality_ratio_percent.is_some_and(|ratio| ratio <= 100) {
            return Err(Error::Build(
                "scene quality ratio must exceed 100 percent".into(),
            ));
        }
        if self.min_leaves < 2 {
            return Err(Error::Build(
                "scene rebuild minimum must be at least two leaves".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn enabled(self, leaves: u32) -> bool {
        self.interval != 0 && leaves >= self.min_leaves
    }

    pub(crate) fn check(self, leaves: u32, limits: &wgpu::Limits) -> Result<SceneRebuildStorage> {
        self.validate()?;
        let count = if self.enabled(leaves) { leaves } else { 0 };
        let storage = if self.quality_ratio_percent.is_some() {
            SceneRebuildStorage::for_quality_leaves(count)?
        } else {
            SceneRebuildStorage::for_leaves(count)?
        };
        let capacity = u64::from(leaves.max(256));
        let biggest = (u64::from(leaves) * 80)
            .max(capacity * 4)
            .max(capacity.div_ceil(256) * 1024);
        if storage.device_bytes > self.max_device_bytes
            || storage.device_bytes != 0
                && (biggest > limits.max_buffer_size
                    || biggest > limits.max_storage_buffer_binding_size
                    || leaves.div_ceil(64) > limits.max_compute_workgroups_per_dimension
                    || limits.max_compute_invocations_per_workgroup < 256
                    || limits.max_compute_workgroup_size_x < 256)
        {
            return Err(Error::Build(
                "scene rebuild exceeds allocation/device budget".into(),
            ));
        }
        Ok(storage)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SceneRebuildStorage {
    pub leaves: u32,
    pub device_bytes: u64,
}

impl SceneRebuildStorage {
    /// Periodic storage plus quality reductions, diagnostics, indirect arguments and parameters.
    pub fn for_quality_leaves(leaves: u32) -> Result<Self> {
        let mut storage = Self::for_leaves(leaves)?;
        if leaves >= 2 {
            storage.device_bytes += 84 + 4 * u64::from((leaves - 1).div_ceil(256));
        }
        Ok(storage)
    }

    /// Additional scatter, sorting, permutation and immutable uniform buffers; zero below two leaves.
    pub fn for_leaves(leaves: u32) -> Result<Self> {
        if leaves < 2 {
            return Ok(Self {
                leaves,
                device_bytes: 0,
            });
        }
        if leaves > (u32::MAX / 80) {
            return Err(Error::Build(
                "scene rebuild address capacity exceeded".into(),
            ));
        }
        let capacity = u64::from(leaves.max(256));
        let device_bytes =
            92 * u64::from(leaves) + 8 * capacity + 1024 * capacity.div_ceil(256) + 80;
        Ok(Self {
            leaves,
            device_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_storage_and_policy_validate_before_allocation() {
        let limits = wgpu::Limits::default();
        for n in [0, 1, 2, 64, 257, 258, 10_000] {
            let periodic = SceneRebuildStorage::for_leaves(n).unwrap();
            let quality = SceneRebuildStorage::for_quality_leaves(n).unwrap();
            let extra = if n < 2 {
                0
            } else {
                84 + 4 * u64::from((n - 1).div_ceil(256))
            };
            assert_eq!(quality.device_bytes, periodic.device_bytes + extra);
            let policy = SceneRebuildPolicy {
                interval: 1,
                min_leaves: 2,
                quality_ratio_percent: Some(125),
                max_device_bytes: quality.device_bytes,
            };
            assert_eq!(
                policy.check(n, &limits).unwrap().device_bytes,
                quality.device_bytes
            );
            if n >= 2 {
                assert!(
                    SceneRebuildPolicy {
                        max_device_bytes: quality.device_bytes - 1,
                        ..policy
                    }
                    .check(n, &limits)
                    .is_err()
                );
            }
            assert_eq!(
                SceneRebuildPolicy {
                    interval: 0,
                    max_device_bytes: 0,
                    ..policy
                }
                .check(n, &limits)
                .unwrap()
                .device_bytes,
                0
            );
        }
        for ratio in [0, 1, 100] {
            assert!(
                SceneRebuildPolicy {
                    quality_ratio_percent: Some(ratio),
                    ..Default::default()
                }
                .validate()
                .is_err()
            );
        }
        assert!(SceneRebuildStorage::for_quality_leaves(u32::MAX).is_err());
    }

    #[test]
    fn rebuild_storage_budgets_disabled_cases_and_extreme_counts() {
        let limits = wgpu::Limits::default();
        let policy = SceneRebuildPolicy {
            min_leaves: 2,
            ..Default::default()
        };
        assert_eq!(SceneRebuildStorage::for_leaves(0).unwrap().device_bytes, 0);
        assert_eq!(SceneRebuildStorage::for_leaves(1).unwrap().device_bytes, 0);
        for count in [2, 64, 256, 257, 10_000] {
            let n = u64::from(count);
            let cap = n.max(256);
            let expected = 92 * n + 8 * cap + 1024 * cap.div_ceil(256) + 80;
            assert_eq!(
                SceneRebuildStorage::for_leaves(count).unwrap().device_bytes,
                expected
            );
            assert!(
                SceneRebuildPolicy {
                    max_device_bytes: expected,
                    ..policy
                }
                .check(count, &limits)
                .is_ok()
            );
            assert!(
                SceneRebuildPolicy {
                    max_device_bytes: expected - 1,
                    ..policy
                }
                .check(count, &limits)
                .is_err()
            );
        }
        assert!(SceneRebuildStorage::for_leaves(u32::MAX).is_err());
        assert!(
            SceneRebuildPolicy {
                min_leaves: 1,
                ..policy
            }
            .validate()
            .is_err()
        );
        assert_eq!(
            SceneRebuildPolicy {
                interval: 0,
                max_device_bytes: 0,
                ..policy
            }
            .check(1000, &limits)
            .unwrap()
            .device_bytes,
            0
        );
        assert_eq!(
            SceneRebuildPolicy::default()
                .check(63, &limits)
                .unwrap()
                .device_bytes,
            0
        );
        assert!(
            policy
                .check(
                    2,
                    &wgpu::Limits {
                        max_storage_buffer_binding_size: 1023,
                        ..limits.clone()
                    }
                )
                .is_err()
        );
        assert!(
            policy
                .check(
                    2,
                    &wgpu::Limits {
                        max_buffer_size: 1023,
                        ..limits.clone()
                    }
                )
                .is_err()
        );
        assert!(
            policy
                .check(
                    2,
                    &wgpu::Limits {
                        max_compute_workgroup_size_x: 128,
                        ..limits
                    }
                )
                .is_err()
        );
    }
}
