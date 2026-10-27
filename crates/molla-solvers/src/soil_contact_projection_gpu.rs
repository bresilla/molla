//! Retained finite-mass sticking projection for one articulated contact island.

use molla_compute::Engine;
use molla_core::{Error, Result};

#[path = "soil_rate_publication_gpu.rs"]
mod publication;
pub use publication::{
    SoilPairedRatePublicationInputs, SoilRatePublicationGpu, SoilRatePublicationInputs,
};

pub const SOIL_PROJECTION_INPUT: u32 = 1;
pub const SOIL_PROJECTION_MASS: u32 = 2;
pub const SOIL_PROJECTION_NONFINITE: u32 = 4;
pub const SOIL_PROJECTION_RESIDUAL: u32 = 8;
pub const SOIL_PROJECTION_BOUNDARY_SOLVE: u32 = 64;
pub const SOIL_PROJECTION_BOUNDARY_BASIS: u32 = 128;
pub const SOIL_PROJECTION_BOUNDARY_ITERATIONS: u32 = 256;
pub const SOIL_PROJECTION_BOUNDARY_DUAL: u32 = 512;

/// Fixed-boundary constraint enumeration; both paths retain every constraint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SoilBoundaryScan {
    #[default]
    Full,
    /// Ordered 128-node GPU cache with complete-scan overflow fallback.
    ExperimentalCached,
}

impl SoilBoundaryScan {
    pub fn label(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::ExperimentalCached => "experimental-cached",
        }
    }
}

/// Node arms are measured in world coordinates from the bound body's COM.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SoilProjectionNode {
    pub arm_mass: [f32; 4],
    /// XYZ momentum; W is an integer mask: +X,-X,+Y,-Y,+Z,-Z inward normals.
    pub momentum: [f32; 4],
}

pub struct SoilProjectionInputs<'a> {
    /// Packed nodes, partitioned by ordered, nonoverlapping body spans.
    pub nodes: &'a wgpu::Buffer,
    /// One [first_node, node_count] pair per bound body.
    pub spans: &'a wgpu::Buffer,
    /// Symmetric positive-definite generalized mass, N x N row-major.
    pub mass: &'a wgpu::Buffer,
    /// World COM Jacobians, one 6 x N row-major block per bound body.
    pub jacobians: &'a wgpu::Buffer,
    pub free_rates: &'a wgpu::Buffer,
}

#[derive(Clone, Copy, Debug)]
pub struct SoilProjectionCapacity {
    pub dofs: u32,
    pub bodies: u32,
    pub nodes: u32,
    pub max_owned_bytes: u64,
}

/// Solves sticking contact with optional unilateral fixed-boundary normals.
/// Inputs must belong to the Engine and use a common configuration. All outputs
/// require zero status after completion; publication into machine/soil state is
/// the caller's transaction. This component does not advance MPM or machine time.
pub struct SoilContactProjectionGpu {
    #[cfg(test)]
    pass_profile: Option<(wgpu::QuerySet, u32)>,
    #[cfg(test)]
    replay_sources: [wgpu::Buffer; 5],
    #[cfg(test)]
    replay_working: [wgpu::Buffer; 3],
    #[cfg(test)]
    replay_capture: Option<(
        SoilProjectionReplay,
        wgpu::ComputePipeline,
        [wgpu::BindGroup; 5],
    )>,
    pipelines: [wgpu::ComputePipeline; 4],
    boundary_pipeline: Option<wgpu::ComputePipeline>,
    group: wgpu::BindGroup,
    rates: wgpu::Buffer,
    motion: wgpu::Buffer,
    impulses: wgpu::Buffer,
    boundary_impulses: wgpu::Buffer,
    status: wgpu::Buffer,
    capacity: SoilProjectionCapacity,
    bytes: u64,
}

#[cfg(test)]
#[derive(Clone)]
pub(crate) struct SoilProjectionReplay {
    pub inputs: [wgpu::Buffer; 5],
    pub outputs: [wgpu::Buffer; 5],
    pub intermediates: [wgpu::Buffer; 3],
    pub capacity: SoilProjectionCapacity,
}

fn invalid(message: &str) -> Error {
    Error::Solver(format!("soil contact projection: {message}"))
}

fn projection_source(width: u32) -> std::borrow::Cow<'static, str> {
    assert!((1..=32).contains(&width));
    let source = concat!(
        include_str!("../shaders/soil_contact_projection.wgsl"),
        "\n",
        include_str!("../shaders/soil_boundary_projection.wgsl")
    );
    if width == 32 {
        return source.into();
    }
    source
        .replace("array<f32,1024>", &format!("array<f32,{}>", width * width))
        .replace("array<f32,32>", &format!("array<f32,{width}>"))
        .replace("array<vec3<u32>,32>", &format!("array<vec3<u32>,{width}>"))
        .into()
}

fn shared_boundary_source(width: u32) -> String {
    let source = projection_source(width);
    let (ordinary, boundary) = source.split_once("struct BoundaryScan").unwrap();
    let cells = width * width;
    let mut boundary = format!("struct BoundaryScan{boundary}");
    let independent = format!(
        "fn independent_boundary(key: vec3<u32>, lower: ptr<function,array<f32,{cells}>>,\n    basis: ptr<function,array<f32,{cells}>>, count: u32)"
    );
    let scan = format!(
        ",\n    lower: ptr<function,array<f32,{cells}>>, basis: ptr<function,array<f32,{cells}>>)"
    );
    assert!(boundary.contains(&independent) && boundary.contains(&scan));
    boundary = boundary
        .replace(
            &independent,
            "fn independent_boundary(key: vec3<u32>, count: u32)",
        )
        .replace(&scan, ")")
        .replace(
            "independent_boundary(key,lower,basis,count)",
            "independent_boundary(key,count)",
        )
        .replace(",&lower,&q)", ")")
        .replace("(*lower)", "lower")
        .replace("(*basis)", "q");
    let mut globals = String::new();
    for name in ["lower", "q", "r"] {
        let local = format!("var {name}: array<f32,{cells}>;");
        assert_eq!(boundary.matches(&local).count(), 1);
        boundary = boundary.replace(&local, "");
        globals.push_str(&format!("var<workgroup> {name}: array<f32,{cells}>;\n"));
    }
    format!("{ordinary}\n{globals}\n{boundary}")
}

fn cached_boundary_source(width: u32) -> String {
    let source = projection_source(width);
    let (prefix, rest) = source.split_once("fn scan_boundaries(").unwrap();
    let (scan, entry) = rest.split_once("\n@compute").unwrap();
    let nodes = "for (var b=0u;b<params.bodies;b++) {\n        let span=spans[b];\n        for (var k=0u;k<span.y;k++) {\n            let index=span.x+k;\n            if nodes[index].arm_mass.w==0.0 { continue; }\n            let mask=u32(nodes[index].momentum.w);";
    assert!(scan.contains(nodes));
    let cached = scan
        .replace(") -> BoundaryScan {", ") -> BoundaryScan {\n    if !boundary_cache_complete { return scan_boundaries_full(x,direction,working,count,lower,basis); }")
        .replace(nodes, "for (var cached=0u;cached<boundary_cache_count;cached++) {\n        let key=boundary_cache[cached];\n        let b=key.x; let index=key.y; let mask=key.z;")
        .replace("        }\n    }\n    return result;", "    }\n    return result;");
    let entry = entry.replace(
        "if !present { return; }",
        "if !present { return; }\n    cache_boundary_nodes();",
    );
    format!(
        "{prefix}\n{}\nfn scan_boundaries_full({scan}\nfn scan_boundaries({cached}\n@compute{entry}",
        include_str!("../shaders/soil_boundary_cache.wgsl")
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProjectionShaderMode {
    Sized,
    ConstantWidth,
    SplitBoundary,
    SharedBoundary,
    CachedBoundary,
}

impl SoilContactProjectionGpu {
    pub fn prepare_publication(
        &self,
        engine: &Engine,
        input: SoilRatePublicationInputs<'_>,
    ) -> Result<SoilRatePublicationGpu> {
        SoilRatePublicationGpu::new(engine, self, input)
    }

    pub fn new(
        engine: &Engine,
        input: SoilProjectionInputs<'_>,
        capacity: SoilProjectionCapacity,
    ) -> Result<Self> {
        Self::prepare(
            engine,
            input,
            capacity,
            false,
            32,
            ProjectionShaderMode::Sized,
        )
    }

    /// Experimental fixed-boundary QP with a bounded feasible active set.
    pub fn new_with_fixed_boundaries(
        engine: &Engine,
        input: SoilProjectionInputs<'_>,
        capacity: SoilProjectionCapacity,
    ) -> Result<Self> {
        Self::prepare(
            engine,
            input,
            capacity,
            true,
            32,
            ProjectionShaderMode::Sized,
        )
    }

    /// Experimental fixed-boundary projection with an ordered 128-node cache.
    /// Uses 2048 bytes of workgroup memory and the original 32-wide solver scratch.
    /// Cache overflow uses the complete original GPU scan, without truncation or
    /// CPU fallback. Owned buffers and dispatch count match the uncached path.
    pub fn new_with_fixed_boundaries_cached(
        engine: &Engine,
        input: SoilProjectionInputs<'_>,
        capacity: SoilProjectionCapacity,
    ) -> Result<Self> {
        Self::prepare(
            engine,
            input,
            capacity,
            true,
            32,
            ProjectionShaderMode::CachedBoundary,
        )
    }

    fn prepare(
        engine: &Engine,
        input: SoilProjectionInputs<'_>,
        capacity: SoilProjectionCapacity,
        boundaries: bool,
        scratch_width: u32,
        mode: ProjectionShaderMode,
    ) -> Result<Self> {
        engine.check_device()?;
        let device = engine.device().ok_or_else(|| invalid("GPU required"))?;
        let n = u64::from(capacity.dofs);
        let b = u64::from(capacity.bodies);
        let bytes = b * 160 + n * n * 4 + n * 8 + 20;
        if !(1..=32).contains(&capacity.dofs)
            || scratch_width < capacity.dofs
            || scratch_width > 32
            || capacity.bodies == 0
            || capacity.nodes == 0
            || capacity.bodies > device.limits().max_compute_workgroups_per_dimension
            || bytes > capacity.max_owned_bytes
            || device.limits().max_storage_buffers_per_shader_stage < 13
            || (mode == ProjectionShaderMode::SharedBoundary
                && 12 * scratch_width * scratch_width
                    > device.limits().max_compute_workgroup_storage_size)
            || (mode == ProjectionShaderMode::CachedBoundary
                && 2048 > device.limits().max_compute_workgroup_storage_size)
        {
            return Err(invalid("capacity or owned-buffer budget exceeded"));
        }
        let inputs = [
            input.nodes,
            input.spans,
            input.mass,
            input.jacobians,
            input.free_rates,
        ];
        for (buffer, required) in inputs.into_iter().zip([
            u64::from(capacity.nodes) * 32,
            b * 8,
            n * n * 4,
            b * 6 * n * 4,
            n * 4,
        ]) {
            if buffer.size() < required
                || buffer.size() > device.limits().max_storage_buffer_binding_size
                || !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
            {
                return Err(invalid("input buffer capacity or usage mismatch"));
            }
        }
        if b * 64 > device.limits().max_storage_buffer_binding_size {
            return Err(invalid("moment storage limit exceeded"));
        }
        let buffer = |label, size| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }
            )
        };
        let moments = buffer("soil contact moments", b * 64);
        let system = buffer("soil contact mass", n * n * 4);
        let rhs = buffer("soil contact rhs", n * 4);
        let rates = buffer("soil projected rates", n * 4);
        let motion = buffer("soil projected body motion", b * 32);
        let impulses = buffer("soil projected body impulse", b * 32);
        let boundary_impulses = buffer("soil fixed-boundary impulse", b * 32);
        let status = buffer("soil projection status", 4);
        let params = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("soil projection parameters"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        molla_core::gpu_write_buffer!(
            engine.queue().unwrap(),
            &params,
            0,
            bytemuck::cast_slice(&[
                capacity.dofs,
                capacity.bodies,
                capacity.nodes,
                u32::from(boundaries)
            ])
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("soil contact projection"),
            source: wgpu::ShaderSource::Wgsl(if mode == ProjectionShaderMode::ConstantWidth {
                projection_source(scratch_width)
                    .replace("params.width", &format!("{}u", capacity.dofs))
                    .into()
            } else {
                projection_source(scratch_width)
            }),
        });
        let entries: Vec<_> = (0..14)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 12 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding < 5,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("soil projection"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("soil projection"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let boundary_module = (boundaries
            && matches!(
                mode,
                ProjectionShaderMode::SplitBoundary
                    | ProjectionShaderMode::SharedBoundary
                    | ProjectionShaderMode::CachedBoundary
            ))
        .then(|| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("soil projection separate boundary scratch"),
                source: wgpu::ShaderSource::Wgsl(if mode == ProjectionShaderMode::SplitBoundary {
                    projection_source(32)
                } else if mode == ProjectionShaderMode::SharedBoundary {
                    shared_boundary_source(scratch_width).into()
                } else {
                    cached_boundary_source(scratch_width).into()
                }),
            })
        });
        let pipeline = |entry, module: &wgpu::ShaderModule| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipelines =
            ["reduce", "assemble", "solve", "publish"].map(|entry| pipeline(entry, &module));
        let boundary_pipeline =
            boundaries.then(|| pipeline("boundary", boundary_module.as_ref().unwrap_or(&module)));
        let buffers = [
            input.nodes,
            input.spans,
            input.mass,
            input.jacobians,
            input.free_rates,
            &moments,
            &system,
            &rhs,
            &rates,
            &motion,
            &impulses,
            &status,
            &params,
            &boundary_impulses,
        ];
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("soil projection"),
            layout: &layout,
            entries: &buffers
                .iter()
                .enumerate()
                .map(|(i, buffer)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        Ok(Self {
            #[cfg(test)]
            pass_profile: None,
            #[cfg(test)]
            replay_sources: [
                input.nodes.clone(),
                input.spans.clone(),
                input.mass.clone(),
                input.jacobians.clone(),
                input.free_rates.clone(),
            ],
            #[cfg(test)]
            replay_working: [moments.clone(), system.clone(), rhs.clone()],
            #[cfg(test)]
            replay_capture: None,
            pipelines,
            boundary_pipeline,
            group,
            rates,
            motion,
            impulses,
            boundary_impulses,
            status,
            capacity,
            bytes,
        })
    }

    /// Encode four passes plus an optional boundary solve; status is sticky.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        #[cfg(test)]
        if let Some((capture, pipeline, groups)) = &self.replay_capture {
            for (target, group) in capture.inputs.iter().zip(groups) {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups((target.size() / 4).div_ceil(64) as u32, 1, 1);
            }
        }
        let work = [
            self.capacity.bodies,
            (self.capacity.dofs * self.capacity.dofs).div_ceil(64),
            1,
            self.capacity.bodies.div_ceil(64),
        ];
        for (i, count) in work.into_iter().enumerate() {
            if i == 3 {
                if let Some(pipeline) = &self.boundary_pipeline {
                    let descriptor = wgpu::ComputePassDescriptor::default();
                    #[cfg(test)]
                    let descriptor = self.profile_descriptor(descriptor, 3);
                    let mut pass = encoder.begin_compute_pass(&descriptor);
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &self.group, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                    molla_core::gpu_metrics::record_dispatch();
                }
            }
            let descriptor = wgpu::ComputePassDescriptor::default();
            #[cfg(test)]
            let descriptor = self.profile_descriptor(descriptor, if i == 3 { 4 } else { i as u32 });
            let mut pass = encoder.begin_compute_pass(&descriptor);
            pass.set_pipeline(&self.pipelines[i]);
            pass.set_bind_group(0, &self.group, &[]);
            pass.dispatch_workgroups(count, 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        }
        #[cfg(test)]
        if let Some((capture, _, _)) = &self.replay_capture {
            for (source, target) in self.replay_working.iter().zip(&capture.intermediates) {
                encoder.copy_buffer_to_buffer(source, 0, target, 0, target.size());
            }
            for (source, target) in [
                &self.rates,
                &self.motion,
                &self.impulses,
                &self.boundary_impulses,
                &self.status,
            ]
            .into_iter()
            .zip(&capture.outputs)
            {
                encoder.copy_buffer_to_buffer(source, 0, target, 0, target.size());
            }
        }
    }

    #[cfg(test)]
    fn profile_descriptor<'a>(
        &'a self,
        mut descriptor: wgpu::ComputePassDescriptor<'a>,
        stage: u32,
    ) -> wgpu::ComputePassDescriptor<'a> {
        if let Some((queries, offset)) = &self.pass_profile {
            descriptor.timestamp_writes = Some(wgpu::ComputePassTimestampWrites {
                query_set: queries,
                beginning_of_pass_write_index: Some(offset + stage * 2),
                end_of_pass_write_index: Some(offset + stage * 2 + 1),
            });
        }
        descriptor
    }

    #[cfg(test)]
    pub(crate) fn prepare_replay_capture(&mut self, engine: &Engine) -> SoilProjectionReplay {
        let buffer = |source: &wgpu::Buffer| {
            molla_core::gpu_create_buffer!(
                engine.device().unwrap(),
                &wgpu::BufferDescriptor {
                    label: Some("soil projection fixed-input capture"),
                    size: source.size(),
                    usage: wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                }
            )
        };
        let capture = SoilProjectionReplay {
            inputs: self.replay_sources.each_ref().map(buffer),
            outputs: [
                &self.rates,
                &self.motion,
                &self.impulses,
                &self.boundary_impulses,
                &self.status,
            ]
            .map(buffer),
            capacity: self.capacity,
            intermediates: self.replay_working.each_ref().map(buffer),
        };
        let device = engine.device().unwrap();
        let shader=device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("soil projection input capture"),
            source: wgpu::ShaderSource::Wgsl("@group(0) @binding(0) var<storage,read> source: array<u32>; @group(0) @binding(1) var<storage,read_write> output_words: array<u32>; @compute @workgroup_size(64) fn copy(@builtin(global_invocation_id) id:vec3<u32>) { if id.x<arrayLength(&source) { output_words[id.x]=source[id.x]; } }".into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("soil projection input capture"),
            layout: None,
            module: &shader,
            entry_point: Some("copy"),
            compilation_options: Default::default(),
            cache: None,
        });
        let groups = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("soil projection input capture"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.replay_sources[i].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: capture.inputs[i].as_entire_binding(),
                    },
                ],
            })
        });
        self.replay_capture = Some((capture.clone(), pipeline, groups));
        capture
    }

    pub fn owned_buffer_bytes(&self) -> u64 {
        self.bytes
    }
    pub fn rates_buffer(&self) -> &wgpu::Buffer {
        &self.rates
    }
    pub fn motion_buffer(&self) -> &wgpu::Buffer {
        &self.motion
    }
    pub fn impulses_buffer(&self) -> &wgpu::Buffer {
        &self.impulses
    }
    /// Additional fixed-boundary impulse, excluding the free-grid boundary step.
    /// Two world-space vec4s per body: linear and angular about its COM.
    pub fn boundary_impulses_buffer(&self) -> &wgpu::Buffer {
        &self.boundary_impulses
    }
    pub fn status_buffer(&self) -> &wgpu::Buffer {
        &self.status
    }
}

#[cfg(test)]
#[path = "soil_contact_projection_gpu_tests.rs"]
pub(crate) mod tests;
