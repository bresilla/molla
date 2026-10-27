use molla_core::{Error, Result};
use molla_math::Real;
use molla_storage::BufferElement;

/// Kelvin-Voigt viscosity on increments of total Green strain.
/// Zero coefficients disable viscosity; both coefficients have units Pa s.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TetViscosity {
    pub shear: Real,
    pub bulk: Real,
}

impl TetViscosity {
    pub fn validate(self) -> Result<()> {
        for (name, value) in [("shear", self.shear), ("bulk", self.bulk)] {
            let gpu = value as f32;
            if !value.is_finite()
                || value < 0.0
                || !gpu.is_finite()
                || (value > 0.0 && gpu < f32::MIN_POSITIVE)
            {
                return Err(Error::Build(format!(
                    "tet viscosity {name} must be zero or finite positive normal f32 (Pa s)"
                )));
            }
        }
        Ok(())
    }
}

impl BufferElement for TetViscosity {
    type Gpu = [f32; 2];
    fn to_gpu(&self) -> Self::Gpu {
        [self.shear as f32, self.bulk as f32]
    }
    fn from_gpu(gpu: &Self::Gpu) -> Self {
        Self {
            shear: gpu[0] as Real,
            bulk: gpu[1] as Real,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MaterialNeoHookean, ModelBuilder};
    use molla_core::WorldId;
    use molla_math::Vec3;
    #[test]
    fn validates_both_coefficients_and_gpu_representation() {
        for value in [-1.0, Real::NAN, Real::INFINITY, Real::MAX, 1e-50, 1e-40] {
            assert!(
                TetViscosity {
                    shear: value,
                    bulk: 0.0
                }
                .validate()
                .is_err()
            );
            assert!(
                TetViscosity {
                    shear: 0.0,
                    bulk: value
                }
                .validate()
                .is_err()
            );
        }
        for value in [0.0, f32::MIN_POSITIVE as Real, 3.0, f32::MAX as Real] {
            let material = TetViscosity {
                shear: value,
                bulk: value,
            };
            material.validate().unwrap();
            assert_eq!(TetViscosity::from_gpu(&material.to_gpu()), material);
        }
        assert_eq!(
            std::mem::size_of::<<TetViscosity as BufferElement>::Gpu>(),
            8
        );
    }

    #[test]
    fn builder_preserves_per_material_values_and_rejects_before_mutation() {
        let mut builder = ModelBuilder::new();
        for offset in [Vec3::ZERO, Vec3::splat(2.0)] {
            for p in [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z] {
                builder.add_particle(offset + p, 1.0, 0.01, WorldId(0));
            }
        }
        let zero = MaterialNeoHookean::elastic(2000.0, 0.3);
        assert_eq!(zero.viscosity, TetViscosity::default());
        assert_eq!(
            MaterialNeoHookean::plastic(2000.0, 0.3, 10.0, 0.0).viscosity,
            zero.viscosity
        );
        assert_eq!(
            MaterialNeoHookean::plastic_kinematic(2000.0, 0.3, 10.0, 0.0, 1.0).viscosity,
            zero.viscosity
        );
        let invalid = MaterialNeoHookean {
            viscosity: TetViscosity {
                shear: -1.0,
                bulk: 0.0,
            },
            ..zero
        };
        let tets = [[0, 1, 2, 3], [4, 5, 6, 7]];
        assert!(
            builder
                .add_tet_mesh_per_material(&tets, &[zero, invalid], &[0, 1], None)
                .is_err()
        );
        let viscous = MaterialNeoHookean {
            viscosity: TetViscosity {
                shear: 3.0,
                bulk: 7.0,
            },
            ..zero
        };
        builder
            .add_tet_mesh_per_material(&tets, &[zero, viscous], &[1, 0], None)
            .unwrap();
        let mut model = builder.build().unwrap();
        assert_eq!(model.tet_count, 2);
        assert_eq!(model.tet_mesh_count, 1);
        assert_eq!(
            model.tet_viscosity.host().unwrap(),
            &[viscous.viscosity, zero.viscosity]
        );
        model.validate_tet_viscosity().unwrap();
        model.tet_viscosity.host_mut().unwrap()[0].bulk = Real::NAN;
        assert!(model.validate_tet_viscosity().is_err());
        model.tet_viscosity =
            molla_storage::Buffer::from_vec(vec![], molla_storage::BufferUsage::Storage);
        assert!(model.validate_tet_viscosity().is_err());
    }

    #[test]
    fn mesh_viscosity_overrides_only_its_elements_and_rejects_atomically() {
        let mut builder = ModelBuilder::new();
        for p in [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z] {
            builder.add_particle(p, 1.0, 0.01, WorldId(0));
        }
        let first = builder
            .add_tet_mesh(&[[0, 1, 2, 3]], 2000.0, 0.3, None)
            .unwrap();
        let second = builder
            .add_tet_mesh(&[[0, 1, 2, 3]; 2], 2000.0, 0.3, None)
            .unwrap();
        let third = builder
            .add_tet_mesh(&[[0, 1, 2, 3]], 2000.0, 0.3, None)
            .unwrap();
        let viscosity = TetViscosity {
            shear: 3.0,
            bulk: 7.0,
        };
        builder.set_tet_mesh_viscosity(second, viscosity).unwrap();
        for value in [-1.0, Real::NAN, Real::INFINITY, Real::MAX, 1e-40] {
            assert!(
                builder
                    .set_tet_mesh_viscosity(
                        second,
                        TetViscosity {
                            shear: value,
                            bulk: 0.0
                        }
                    )
                    .is_err()
            );
        }
        assert!(
            builder
                .set_tet_mesh_viscosity(crate::TetMeshId(u32::MAX), viscosity)
                .is_err()
        );
        assert!(
            builder
                .set_tet_mesh_viscosity(crate::TetMeshId(3), viscosity)
                .is_err()
        );
        builder.set_tet_mesh_viscosity(first, viscosity).unwrap();
        builder
            .set_tet_mesh_viscosity(first, TetViscosity::default())
            .unwrap();
        builder
            .set_tet_mesh_viscosity(third, TetViscosity::default())
            .unwrap();
        let model = builder.build().unwrap();
        assert_eq!(
            model.tet_viscosity.host().unwrap(),
            &[
                TetViscosity::default(),
                viscosity,
                viscosity,
                TetViscosity::default()
            ]
        );
    }
}
