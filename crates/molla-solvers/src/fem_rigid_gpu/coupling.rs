use super::*;
use molla_kernels::soft_rigid_solve::{SoftMaterialIterationBindings, SoftMaterialIterationKernel};

/// Fixed-geometry GPU material/contact iteration limits and velocity-correction tolerances.
#[derive(Clone, Copy, Debug)]
pub struct MaterialContactConfig {
    pub iterations: u32,
    pub linear_tolerance: f32,
    pub angular_tolerance: f32,
}

impl MaterialContactConfig {
    fn validate(self) -> Result<()> {
        if !(2..=4096).contains(&self.iterations)
            || !self.linear_tolerance.is_normal()
            || self.linear_tolerance <= 0.0
            || !self.angular_tolerance.is_normal()
            || self.angular_tolerance <= 0.0
        {
            return Err(Error::Build(
                "invalid material contact iteration configuration".into(),
            ));
        }
        Ok(())
    }
}

pub(super) struct MaterialContactState {
    pub(super) config: MaterialContactConfig,
    pub(super) reuse_iterations: Option<u32>,
    pub(super) contact_sweeps: Option<u32>,
    pub(super) polish_iterations: Option<u32>,
    pub(super) polish_cycles: u32,
    pub(super) pre_polish: wgpu::Buffer,
    pub(super) input: State,
    predicted: wgpu::Buffer,
    previous_body: wgpu::Buffer,
    pub(super) kernel: SoftMaterialIterationKernel,
    particles: u32,
    bodies: u32,
}

impl MaterialContactState {
    pub(super) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        bodies: usize,
        config: MaterialContactConfig,
    ) -> Result<Self> {
        config.validate()?;
        if model.linear_damping != 0.0
            || model
                .tet_yield_stress
                .host()?
                .iter()
                .any(|v| *v <= 1e29 || v.is_nan())
        {
            return Err(Error::Build(
                "material contact iteration requires elastic tets and zero world damping".into(),
            ));
        }
        let mut input = model.state()?;
        input.upload_to_device(device, queue)?;
        let allocate = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                mapped_at_creation: false,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };
        Ok(Self {
            input,
            config,
            reuse_iterations: None,
            contact_sweeps: None,
            polish_iterations: None,
            polish_cycles: 1,
            pre_polish: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("material pre-polish diagnostics"),
                size: 16,
                mapped_at_creation: false,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            }),
            kernel: SoftMaterialIterationKernel::new(device),
            predicted: allocate(
                "material predictor velocities",
                model.particle_count as u64 * 16,
            ),
            previous_body: allocate("material previous body velocities", bodies as u64 * 32),
            particles: model.particle_count as u32,
            bodies: bodies as u32,
        })
    }

    pub(super) fn begin(&mut self, encoder: &mut wgpu::CommandEncoder, state: &State) {
        encoder.clear_buffer(&self.pre_polish, 0, None);
        macro_rules! copy {
            ($field:ident) => {{
                let source = state.$field.device_buffer().unwrap();
                let destination = self.input.$field.device_buffer().unwrap();
                encoder.copy_buffer_to_buffer(source, 0, destination, 0, source.size());
                self.input.$field.mark_device_dirty();
            }};
        }
        copy!(particle_q);
        copy!(particle_qd);
        copy!(tet_f_plastic);
        copy!(tet_plastic_strain);
        copy!(tet_back_stress);
    }

    pub(super) fn capture(&self, encoder: &mut wgpu::CommandEncoder, soft: &State, rigid: &State) {
        encoder.copy_buffer_to_buffer(
            soft.particle_qd.device_buffer().unwrap(),
            0,
            &self.predicted,
            0,
            self.particles as u64 * 16,
        );
        encoder.copy_buffer_to_buffer(
            rigid.body_qd.device_buffer().unwrap(),
            0,
            &self.previous_body,
            0,
            self.bodies as u64 * 32,
        );
    }

    pub(super) fn accumulate(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        soft: &State,
        rigid: &State,
        status: &wgpu::Buffer,
        final_iteration: bool,
    ) -> Result<()> {
        self.kernel.encode(
            device,
            encoder,
            SoftMaterialIterationBindings {
                input_velocity: self.input.particle_qd.device_buffer().unwrap(),
                predicted_velocity: &self.predicted,
                corrected_velocity: soft.particle_qd.device_buffer().unwrap(),
                previous_body_velocity: &self.previous_body,
                corrected_body_velocity: rigid.body_qd.device_buffer().unwrap(),
                status,
                particles: self.particles,
                bodies: self.bodies,
            },
            self.config.linear_tolerance,
            self.config.angular_tolerance,
            final_iteration,
        )?;
        self.input.particle_qd.mark_device_dirty();
        Ok(())
    }

    pub(super) fn check_polish(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        soft: &State,
        rigid: &State,
        status: &wgpu::Buffer,
    ) -> Result<()> {
        self.kernel.encode_check(
            device,
            encoder,
            SoftMaterialIterationBindings {
                input_velocity: self.input.particle_qd.device_buffer().unwrap(),
                predicted_velocity: &self.predicted,
                corrected_velocity: soft.particle_qd.device_buffer().unwrap(),
                previous_body_velocity: &self.previous_body,
                corrected_body_velocity: rigid.body_qd.device_buffer().unwrap(),
                status,
                particles: self.particles,
                bodies: self.bodies,
            },
            self.config.linear_tolerance,
            self.config.angular_tolerance,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_iteration_config_rejects_invalid_bounds() {
        let config = MaterialContactConfig {
            iterations: 256,
            linear_tolerance: 1e-4,
            angular_tolerance: 1e-4,
        };
        assert!(config.validate().is_ok());
        for iterations in [0, 1, 4097, u32::MAX] {
            assert!(
                MaterialContactConfig {
                    iterations,
                    ..config
                }
                .validate()
                .is_err()
            );
        }
        for tolerance in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::from_bits(1)] {
            assert!(
                MaterialContactConfig {
                    linear_tolerance: tolerance,
                    ..config
                }
                .validate()
                .is_err()
            );
            assert!(
                MaterialContactConfig {
                    angular_tolerance: tolerance,
                    ..config
                }
                .validate()
                .is_err()
            );
        }
    }
}
