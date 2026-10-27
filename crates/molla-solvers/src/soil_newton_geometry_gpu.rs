//! Analytic geometry, device-prepared Newton contact and world-space recovery.

use crate::soil_contact_geometry_gpu::*;
pub use crate::soil_friction_geometry_gpu::SoilFrictionGeometryInputs as SoilNewtonGeometryInputs;
use crate::soil_newton_failure::NewtonFailureCapture;
use crate::soil_newton_gpu::*;
use crate::soil_newton_prepare_gpu::*;
use molla_compute::Engine;
use molla_core::{Error, Result};

#[derive(Clone, Copy, Debug)]
pub struct SoilNewtonGeometryCapacity {
    pub dofs: u32,
    pub bodies: u32,
    pub contacts: u32,
    pub newton_iterations: u32,
    pub krylov_iterations: u32,
    pub max_owned_bytes: u64,
}

/// Uses paired generalized rates and cold-starts contact impulses per encoding.
/// Geometry/candidate/solver faults remain latched in the common status until cleared.
pub struct ExperimentalSoilNewtonGeometryGpu {
    geometry: ExperimentalSoilContactGeometryGpu,
    preparation: ExperimentalSoilNewtonPrepareGpu,
    solver: ExperimentalSoilNewtonGpu,
    #[cfg(all(
        test,
        feature = "active-publication-newton",
        not(target_arch = "wasm32")
    ))]
    shadow: Option<crate::soil_newton_gpu::shadow::Shadow>,
    failure: NewtonFailureCapture,
    program: std::sync::Arc<GeometryProgram>,
    groups: [wgpu::BindGroup; 2],
    outputs: [wgpu::Buffer; 3],
    status: wgpu::Buffer,
    prepared_contacts: wgpu::Buffer,
    contacts: u32,
    bytes: u64,
}

pub(crate) struct GeometryProgram {
    pub(crate) pipelines: [wgpu::ComputePipeline; 2],
}

pub(crate) fn geometry_program(engine: &Engine) -> Result<std::sync::Arc<GeometryProgram>> {
    let source = shader_source();
    let key = format!(
        "newton-geometry-v1;prepare,recover;wgsl={}",
        blake3::hash(source.as_bytes())
    );
    engine.cached_gpu_program(&key, |device| {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Newton geometry bridge"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipelines = ["prepare", "recover"].map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: engine.pipeline_cache(),
            })
        });
        Ok(GeometryProgram { pipelines })
    })
}

fn invalid() -> Error {
    Error::Solver("soil Newton geometry: invalid capacity, usage or device limit".into())
}

fn solver_capacity(c: SoilNewtonGeometryCapacity) -> SoilNewtonCapacity {
    SoilNewtonCapacity {
        dofs: c.dofs,
        contacts: c.contacts,
        newton_iterations: c.newton_iterations,
        krylov_iterations: c.krylov_iterations,
        max_owned_bytes: u64::MAX,
    }
}

pub(crate) fn shader_source() -> String {
    let source = include_str!("../shaders/soil_shared_wall_geometry.wgsl");
    let (prepare, _) = source
        .split_once("@compute @workgroup_size(64)\nfn recover(")
        .unwrap();
    format!("{}\n{}",prepare.replace("struct Status { fault:atomic<u32> }","struct Status { fault:atomic<u32>, preparation:u32, newton:u32, publication:u32, steps:u32, linear:u32, trials:u32, pad:u32 }"),include_str!("../shaders/soil_newton_geometry_recover.wgsl"))
}

impl ExperimentalSoilNewtonGeometryGpu {
    pub fn required_bytes(cap: SoilNewtonGeometryCapacity) -> Result<u64> {
        if !(1..=64).contains(&cap.bodies) || cap.contacts > 16384 {
            return Err(invalid());
        }
        let c = u64::from(cap.contacts.max(1));
        let n = u64::from(cap.dofs);
        Ok(
            ExperimentalSoilNewtonGpu::required_condensed_bytes(solver_capacity(cap))?
                + ExperimentalSoilNewtonPrepareGpu::required_bytes(cap.dofs, cap.contacts)?
                + NewtonFailureCapture::required_bytes(cap.dofs, cap.contacts)?
                + (176 + 20 * n) * c
                + 64,
        )
    }

    pub fn new(
        engine: &Engine,
        input: SoilNewtonGeometryInputs<'_>,
        cap: SoilNewtonGeometryCapacity,
    ) -> Result<Self> {
        #[cfg(test)]
        let _setup = crate::soil_setup_trace::stage("newton.geometry");
        engine.check_device()?;
        let bytes = Self::required_bytes(cap)?;
        let device = engine.device().ok_or_else(invalid)?;
        let limits = device.limits();
        let (newton_lanes, newton_storage) = crate::soil_newton_gpu::execution_limits(engine);
        let n = u64::from(cap.dofs);
        let b = u64::from(cap.bodies);
        let c = u64::from(cap.contacts.max(1));
        let borrowed = [
            input.mass,
            input.free_rates,
            input.centers,
            input.shapes,
            input.jacobians,
            input.queries,
            input.coefficients,
        ];
        if bytes > cap.max_owned_bytes
            || limits.max_storage_buffers_per_shader_stage < 21
            || limits.max_bindings_per_bind_group < 22
            || limits.max_uniform_buffers_per_shader_stage < 1
            || limits.max_compute_invocations_per_workgroup < newton_lanes
            || limits.max_compute_workgroup_size_x < newton_lanes
            || limits.max_compute_workgroup_storage_size < newton_storage
            || cap.contacts.max(1).div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || borrowed
                .iter()
                .zip([4 * n * n, 8 * n, 16 * b, 64 * b, 24 * b * n, 48 * c, 4 * b])
                .any(|(v, s)| {
                    v.size() < s
                        || v.size() > limits.max_storage_buffer_binding_size
                        || !v.usage().contains(wgpu::BufferUsages::STORAGE)
                })
            || [
                456 * c,
                33280 + 432 * c,
                64 * c * u64::from(cap.krylov_iterations + 1),
                16 * c * n,
                4 * n * n,
                8 * n,
                32,
                NewtonFailureCapture::required_bytes(cap.dofs, cap.contacts)? - 16,
            ]
            .iter()
            .any(|&s| s > limits.max_buffer_size || s > limits.max_storage_buffer_binding_size)
        {
            return Err(invalid());
        }
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let create = |label, size, usage| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage,
                    mapped_at_creation: false
                }
            )
        };
        let nodes = create("Newton geometry nodes", 48 * c, usage);
        let normals = create("Newton normal rows", 4 * c * n, usage);
        let contacts = create("Newton geometry contacts", 64 * c, usage);
        let rows = create("Newton point rows", 16 * c * n, usage);
        let outputs = std::array::from_fn(|_| create("Newton world output", 16 * c, usage));
        let status = create("Newton geometry common status", 32, usage);
        let preparation = ExperimentalSoilNewtonPrepareGpu::new(
            engine,
            SoilNewtonPrepareInputs {
                mass: input.mass,
                point_rows: &rows,
                contacts: &contacts,
                upstream_status: &status,
            },
            cap.dofs,
            cap.contacts,
            u64::MAX,
        )?;
        let solver = ExperimentalSoilNewtonGpu::new_fischer_conditioned(
            engine,
            SoilNewtonInputs {
                mass: input.mass,
                free_rates: input.free_rates,
                point_rows: &rows,
                point_response: preparation.response_buffer(),
                contacts: &contacts,
                upstream_status: preparation.status_buffer(),
            },
            solver_capacity(cap),
        )?;
        #[cfg(all(
            test,
            feature = "active-publication-newton",
            not(target_arch = "wasm32")
        ))]
        let shadow = if std::env::var("MOLLA_NEWTON_PUBLICATION_SHADOW").as_deref() == Ok("1") {
            Some(crate::soil_newton_gpu::shadow::Shadow::new(
                engine,
                SoilNewtonInputs {
                    mass: input.mass,
                    free_rates: input.free_rates,
                    point_rows: &rows,
                    point_response: preparation.response_buffer(),
                    contacts: &contacts,
                    upstream_status: preparation.status_buffer(),
                },
                solver_capacity(cap),
                &solver,
            )?)
        } else {
            None
        };
        let failure = NewtonFailureCapture::new_with_method(
            engine,
            [
                input.mass,
                input.free_rates,
                &rows,
                preparation.response_buffer(),
                &contacts,
                solver.state_buffer(),
                solver.status_buffer(),
                solver.publication_status_buffer(),
                preparation.status_buffer(),
                &status,
            ],
            [
                cap.dofs,
                cap.contacts,
                cap.newton_iterations,
                cap.krylov_iterations,
            ],
            solver.failure_method(),
        )?;
        let geometry = ExperimentalSoilContactGeometryGpu::new(
            engine,
            SoilContactGeometryInputs {
                centers: input.centers,
                shapes: input.shapes,
                jacobians: input.jacobians,
                queries: input.queries,
            },
            SoilContactGeometryOutputs {
                nodes: &nodes,
                rows: &normals,
                status: &status,
            },
            SoilContactGeometryCapacity {
                dofs: cap.dofs,
                bodies: cap.bodies,
                queries: cap.contacts,
                max_owned_bytes: 16 * c + 16,
            },
        )?;
        let params = create(
            "Newton geometry dimensions",
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        molla_core::gpu_write_buffer!(
            engine.queue().unwrap(),
            &params,
            0,
            bytemuck::cast_slice(&[cap.dofs, cap.bodies, cap.contacts, 0])
        );
        let program = geometry_program(engine)?;
        let bindings = [
            &nodes,
            input.queries,
            input.centers,
            input.jacobians,
            input.coefficients,
            &contacts,
            &rows,
            solver.outputs_buffer(),
            &outputs[0],
            &outputs[1],
            &outputs[2],
            &status,
            &params,
            solver.status_buffer(),
            preparation.status_buffer(),
            solver.publication_status_buffer(),
        ];
        let groups = std::array::from_fn(|stage| {
            let indices: &[usize] = if stage == 0 {
                &[0, 1, 2, 3, 4, 5, 6, 11, 12]
            } else {
                &[0, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            };
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Newton geometry stage"),
                layout: &program.pipelines[stage].get_bind_group_layout(0),
                entries: &indices
                    .iter()
                    .map(|&i| wgpu::BindGroupEntry {
                        binding: i as u32,
                        resource: bindings[i].as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            })
        });
        Ok(Self {
            geometry,
            preparation,
            solver,
            #[cfg(all(
                test,
                feature = "active-publication-newton",
                not(target_arch = "wasm32")
            ))]
            shadow,
            failure,
            program,
            groups,
            outputs,
            status,
            prepared_contacts: contacts,
            contacts: cap.contacts,
            bytes,
        })
    }

    fn stage(&self, encoder: &mut wgpu::CommandEncoder, stage: usize, groups: u32) {
        if groups == 0 {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.program.pipelines[stage]);
        pass.set_bind_group(0, &self.groups[stage], &[]);
        pass.dispatch_workgroups(groups, 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_preparation(encoder);
        self.encode_solve(encoder);
        self.encode_publication(encoder);
    }
    pub(crate) fn encode_preparation(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_preparation_traced(encoder, |_, _, _| {});
    }
    pub(crate) fn encode_preparation_traced(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        mut trace: impl FnMut(&mut wgpu::CommandEncoder, PreparationPhase, bool),
    ) {
        trace(encoder, PreparationPhase::Reset, true);
        self.preparation.encode_reset(encoder);
        self.solver.encode_reset(encoder);
        trace(encoder, PreparationPhase::Reset, false);
        trace(encoder, PreparationPhase::Geometry, true);
        self.geometry.encode(encoder);
        trace(encoder, PreparationPhase::Geometry, false);
        trace(encoder, PreparationPhase::Contacts, true);
        self.stage(encoder, 0, self.contacts.div_ceil(64));
        trace(encoder, PreparationPhase::Contacts, false);
        self.preparation.encode_traced(encoder, &mut trace);
    }
    pub(crate) fn encode_solve(&self, encoder: &mut wgpu::CommandEncoder) {
        #[cfg(test)]
        self.failure.encode_initial(encoder);
        self.solver.encode(encoder);
        #[cfg(all(
            test,
            feature = "active-publication-newton",
            not(target_arch = "wasm32")
        ))]
        if let Some(shadow) = &self.shadow {
            shadow.encode(encoder);
        }
    }
    #[cfg(all(
        test,
        feature = "active-publication-newton",
        not(target_arch = "wasm32")
    ))]
    pub(crate) fn shadow_summary(&self) -> Option<&wgpu::Buffer> {
        self.shadow.as_ref().map(|s| s.summary())
    }
    pub(crate) fn encode_publication(&self, encoder: &mut wgpu::CommandEncoder) {
        self.failure.encode(encoder);
        self.stage(encoder, 1, self.contacts.max(1).div_ceil(64));
    }
    /// Common fault, preparation/solve/publication faults, steps, inner iterations,
    /// trials and padding. Common stage bits are 4096, 8192 and 16384 respectively.
    pub fn status_buffer(&self) -> &wgpu::Buffer {
        &self.status
    }
    pub(crate) fn history_inputs<'a>(
        &'a self,
        ids: &'a wgpu::Buffer,
        queries: &'a wgpu::Buffer,
    ) -> [&'a wgpu::Buffer; 7] {
        [
            ids,
            queries,
            &self.prepared_contacts,
            self.solver.state_buffer(),
            &self.status,
            self.solver.status_buffer(),
            self.solver.publication_status_buffer(),
        ]
    }
    pub fn rates_buffer(&self) -> &wgpu::Buffer {
        self.solver.rates_buffer()
    }
    pub fn velocities_buffer(&self) -> &wgpu::Buffer {
        &self.outputs[0]
    }
    pub fn contact_impulses_buffer(&self) -> &wgpu::Buffer {
        &self.outputs[1]
    }
    pub fn wall_impulses_buffer(&self) -> &wgpu::Buffer {
        &self.outputs[2]
    }
    pub fn geometry_buffer(&self) -> &wgpu::Buffer {
        self.geometry.geometry_buffer()
    }
    pub fn preparation_failure_buffer(&self) -> &wgpu::Buffer {
        self.preparation.failure_buffer()
    }
    /// First failed solve: header/reports, mass, paired rates, then per-contact
    /// rows, response, contact and candidate state. Schema zero means empty.
    pub fn failure_buffer(&self) -> &wgpu::Buffer {
        self.failure.buffer()
    }

    #[cfg(test)]
    pub(crate) fn success_seed_buffer(&self) -> Option<&wgpu::Buffer> {
        self.failure.success_seed_buffer()
    }
    pub(crate) fn method_name(&self) -> &'static str {
        self.solver.failure_method().name()
    }
    pub fn owned_buffer_bytes(&self) -> u64 {
        self.bytes
    }
}

#[cfg(test)]
#[path = "soil_newton_geometry_cache_tests.rs"]
mod cache_tests;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn newton_geometry_shader_and_budget() {
        let capacity = SoilNewtonGeometryCapacity {
            dofs: 3,
            bodies: 1,
            contacts: 128,
            newton_iterations: 64,
            krylov_iterations: 64,
            max_owned_bytes: u64::MAX,
        };
        assert_eq!(
            ExperimentalSoilNewtonGeometryGpu::required_bytes(capacity).unwrap(),
            817356
        );
        for invalid in [
            SoilNewtonGeometryCapacity {
                dofs: 0,
                ..capacity
            },
            SoilNewtonGeometryCapacity {
                bodies: 65,
                ..capacity
            },
            SoilNewtonGeometryCapacity {
                contacts: 16385,
                ..capacity
            },
            SoilNewtonGeometryCapacity {
                newton_iterations: 65,
                ..capacity
            },
        ] {
            assert!(ExperimentalSoilNewtonGeometryGpu::required_bytes(invalid).is_err());
        }
        let source = shader_source();
        let module = wgpu::naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
    }
}
