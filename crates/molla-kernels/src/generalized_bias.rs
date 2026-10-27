//! Retained translated-origin Newton–Euler bias for an unrestricted joint tree.

use molla_core::{Error, Result};
use molla_math::Vec3;
use molla_sim::Model;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Body {
    metadata: [u32; 4],
    scalars: [[f32; 4]; 11],
}

/// Device state inputs in global body/DOF order, all belonging to the supplied device.
pub struct ExperimentalBiasBindings<'a> {
    pub poses: &'a wgpu::Buffer,
    pub position_low: &'a wgpu::Buffer,
    pub paired_rates: &'a wgpu::Buffer,
    pub external_com_wrenches: &'a wgpu::Buffer,
}

/// Computes C(q, qd) - gravity - external COM loads and a dense COM-Gram mass matrix.
///
/// Model data is an immutable snapshot; rebuild after model changes. Supports
/// one topologically ordered dynamic tree of fixed, hinge, slider and free joints,
/// up to 64 bodies and 32 DOFs, without reversed/disabled/limited joints.
/// Poses and paired rates must describe the same configuration. Joint controls,
/// damping, constraints and state advancement are not evaluated by this kernel.
pub struct ExperimentalBiasKernel {
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    output: wgpu::Buffer,
    status: wgpu::Buffer,
    mass: wgpu::Buffer,
    paired_mass: Option<wgpu::Buffer>,
    com_jacobians: Option<wgpu::Buffer>,
    snapshot: Vec<Body>,
    signature: Vec<u64>,
}

fn signature(model: &Model) -> Result<Vec<u64>> {
    let mut values = vec![
        model.body_count as u64,
        model.joint_count as u64,
        model.joint_dof_total as u64,
        model.joint_coord_total as u64,
        model.articulation_count as u64,
    ];
    values.extend(
        model
            .joint_coord_offset
            .host()?
            .iter()
            .map(|&v| u64::from(v)),
    );
    values.extend(
        model
            .articulation_start
            .host()?
            .iter()
            .map(|&v| u64::from(v)),
    );
    values.extend(model.body_world.host()?.iter().map(|&v| v as u64));
    values.extend(model.body_mass.host()?.iter().map(|v| v.to_bits()));
    for tensor in model.body_inertia.host()? {
        values.extend(tensor.to_cols_array().map(f64::to_bits));
    }
    for v in model
        .body_com
        .host()?
        .iter()
        .chain(model.joint_axis.host()?)
    {
        values.extend(v.to_array().map(f64::to_bits));
    }
    for t in model
        .joint_x_p
        .host()?
        .iter()
        .chain(model.joint_x_c.host()?)
    {
        values.extend(t.position.to_array().map(f64::to_bits));
        values.extend(t.rotation.to_array().map(f64::to_bits));
    }
    let gravity = model.gravity.host()?;
    for &world in model.body_world.host()? {
        let g = gravity
            .get(world.max(0) as usize)
            .ok_or_else(|| Error::Gpu("invalid bias world".into()))?;
        values.extend(g.to_array().map(f64::to_bits));
    }
    Ok(values)
}

fn source(bodies: usize, dofs: usize) -> String {
    include_str!("../shaders/generalized_bias.wgsl")
        .replace("__BODIES__", &bodies.to_string())
        .replace("__DOFS__", &dofs.to_string())
}

fn paired_source(text: String) -> String {
    let publication = "mass_matrix[j*D+i]=value;";
    assert_eq!(text.matches(publication).count(), 1);
    format!("@group(0) @binding(8) var<storage,read_write> paired_mass: array<vec2<f32>>;\n{}",
        text.replace(publication,&format!("{publication}\nlet pair=vec2<f32>(value,f32(gram[i*D+j]-f64(value)));\npaired_mass[i*D+j]=pair; paired_mass[j*D+i]=pair;")))
}

fn coupling_source(mut text: String) -> String {
    for (marker, insertion) in [
        (
            "jacobian[joint.metadata.w+d]=columns[joint.metadata.w+d];",
            "ancestors[k] |= 1u << (joint.metadata.w+d);",
        ),
        (
            "let vi=jacobian[i].linear+cross(jacobian[i].angular,center);",
            "if !finite3(vec3<f64>(vec3<f32>(vi))) || !finite3(vec3<f64>(vec3<f32>(jacobian[i].angular))) { status[0]=2u; return; }",
        ),
    ] {
        assert_eq!(text.matches(marker).count(), 1);
        text = text.replace(marker, &format!("{marker}\n{insertion}"));
    }
    let marker = "    for (var i=0u; i<D; i++) {\n        bias[i]=result[i];";
    assert_eq!(text.matches(marker).count(), 1);
    text = text.replace(
        marker,
        &format!("    publish_com_jacobians(root);\n{marker}"),
    );
    format!(
        "{}\n{text}",
        include_str!("../shaders/generalized_com_jacobians.wgsl")
    )
}

fn pack(model: &Model) -> Result<Vec<Body>> {
    let invalid = || Error::Gpu("unsupported or invalid experimental bias model".into());
    let n = model.joint_count;
    let dofs = model.joint_dof_total;
    if n == 0
        || n > 64
        || model.body_count != n
        || model.articulation_count > 1
        || !(1..=32).contains(&dofs)
    {
        return Err(invalid());
    }
    let kinds = model.joint_type.host()?;
    let parents = model.joint_parent.host()?;
    let children = model.joint_child.host()?;
    let reversed = model.joint_reversed.host()?;
    let enabled = model.joint_enabled.host()?;
    let flags = model.body_flags.host()?;
    let offsets = model.joint_dof_offset.host()?;
    let xp = model.joint_x_p.host()?;
    let xc = model.joint_x_c.host()?;
    let axes = model.joint_axis.host()?;
    let masses = model.body_mass.host()?;
    let inertias = model.body_inertia.host()?;
    let coms = model.body_com.host()?;
    let worlds = model.body_world.host()?;
    let gravity = model.gravity.host()?;
    let lower = model.joint_limit_lower.host()?;
    let upper = model.joint_limit_upper.host()?;
    if [
        kinds.len(),
        parents.len(),
        children.len(),
        reversed.len(),
        enabled.len(),
        flags.len(),
        xp.len(),
        xc.len(),
        axes.len(),
        masses.len(),
        inertias.len(),
        coms.len(),
        worlds.len(),
    ]
    .iter()
    .any(|&length| length != n)
        || offsets.len() != n + 1
        || offsets[0] != 0
        || offsets[n] as usize != dofs
        || lower.len() != dofs
        || upper.len() != dofs
        || lower.iter().any(|&v| v != f64::NEG_INFINITY)
        || upper.iter().any(|&v| v != f64::INFINITY)
    {
        return Err(invalid());
    }
    let vector = |v: Vec3| [v.x as f32, v.y as f32, v.z as f32, 0.0];
    let mut owner = vec![None; n];
    let mut packed = Vec::with_capacity(n);
    let mut world = None;
    for j in 0..n {
        let child = children[j] as usize;
        let parent = parents[j];
        let kind = kinds[j];
        let width = match kind {
            0 | 1 => 1,
            3 => 0,
            4 => 6,
            _ => return Err(invalid()),
        };
        if child >= n
            || owner[child].is_some()
            || reversed[j] != 0
            || enabled[j] != 1
            || flags[child] != 0
            || offsets[j].checked_add(width) != Some(offsets[j + 1])
            || (j == 0 && parent != -1)
            || (j > 0 && parent < 0)
            || !masses[child].is_finite()
            || masses[child] < 0.0
            || (kind <= 1 && (axes[j].length_squared() - 1.0).abs() > 1e-6)
        {
            return Err(invalid());
        }
        let parent_local = if parent < 0 {
            u32::MAX
        } else {
            owner
                .get(parent as usize)
                .copied()
                .flatten()
                .ok_or_else(invalid)?
        };
        let w = worlds[child].max(0) as usize;
        let g = *gravity.get(w).ok_or_else(invalid)?;
        if world.is_some_and(|prior| prior != w) {
            return Err(invalid());
        }
        world = Some(w);
        let tensor = inertias[child];
        let a = tensor.to_cols_array();
        if !tensor.is_finite()
            || a[1] != a[3]
            || a[2] != a[6]
            || a[5] != a[7]
            || a[0] < 0.0
            || a[4] < 0.0
            || a[8] < 0.0
            || tensor.determinant() < 0.0
            || a[0] * a[4] < a[1] * a[1]
            || a[0] * a[8] < a[2] * a[2]
            || a[4] * a[8] < a[5] * a[5]
        {
            return Err(invalid());
        }
        for transform in [xp[j], xc[j]] {
            if (transform.rotation.length_squared() - 1.0).abs() > 1e-6 {
                return Err(invalid());
            }
        }
        let record = Body {
            metadata: [parent_local, child as u32, kind, offsets[j]],
            scalars: [
                [masses[child] as f32, 0.0, 0.0, 0.0],
                vector(coms[child]),
                vector(tensor.x_axis),
                vector(tensor.y_axis),
                vector(tensor.z_axis),
                vector(xp[j].position),
                xp[j].rotation.to_array().map(|v| v as f32),
                vector(xc[j].position),
                xc[j].rotation.to_array().map(|v| v as f32),
                vector(axes[j]),
                vector(g),
            ],
        };
        if record.scalars.iter().flatten().any(|v| !v.is_finite()) {
            return Err(invalid());
        }
        packed.push(record);
        owner[child] = Some(j as u32);
    }
    Ok(packed)
}

impl ExperimentalBiasKernel {
    /// Retains model/output/status buffers within `retained_buffer_budget` bytes.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        inputs: ExperimentalBiasBindings<'_>,
        retained_buffer_budget: u64,
    ) -> Result<Self> {
        Self::build(
            device,
            queue,
            model,
            inputs,
            retained_buffer_budget,
            false,
            false,
        )
    }

    /// Publishes paired Gram entries alongside the f32 coupling mirror.
    pub fn new_with_paired_mass(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        inputs: ExperimentalBiasBindings<'_>,
        retained_buffer_budget: u64,
    ) -> Result<Self> {
        Self::build(
            device,
            queue,
            model,
            inputs,
            retained_buffer_budget,
            true,
            false,
        )
    }

    /// Publishes paired mass and global-body, row-major 6-by-DOF COM Jacobians.
    pub fn new_with_coupling_operators(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        inputs: ExperimentalBiasBindings<'_>,
        retained_buffer_budget: u64,
    ) -> Result<Self> {
        Self::build(
            device,
            queue,
            model,
            inputs,
            retained_buffer_budget,
            true,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        inputs: ExperimentalBiasBindings<'_>,
        retained_buffer_budget: u64,
        paired: bool,
        coupling: bool,
    ) -> Result<Self> {
        let invalid = || Error::Gpu("invalid experimental bias device, budget or binding".into());
        let packed = pack(model)?;
        let signature = signature(model)?;
        let n = model.body_count as u64;
        let d = model.joint_dof_total as u64;
        let mut sizes = vec![192 * n, 32 * n, 16 * n, 8 * d, 32 * n, 8 * d, 4, 4 * d * d];
        if paired {
            sizes.push(8 * d * d);
        }
        if coupling {
            sizes.push(24 * n * d);
        }
        let limits = device.limits();
        let inputs = [
            inputs.poses,
            inputs.position_low,
            inputs.paired_rates,
            inputs.external_com_wrenches,
        ];
        if !device.features().contains(wgpu::Features::SHADER_F64)
            || retained_buffer_budget < sizes[0] + sizes[5..].iter().sum::<u64>()
            || limits.max_storage_buffers_per_shader_stage < sizes.len() as u32
            || limits.max_bindings_per_bind_group < sizes.len() as u32
            || sizes
                .iter()
                .any(|&s| s > limits.max_storage_buffer_binding_size || s > limits.max_buffer_size)
            || inputs
                .iter()
                .zip(&sizes[1..5])
                .any(|(b, &s)| b.size() < s || !b.usage().contains(wgpu::BufferUsages::STORAGE))
        {
            return Err(invalid());
        }
        let create = |size, usage| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("experimental bias"),
                    size,
                    usage: wgpu::BufferUsages::STORAGE | usage,
                    mapped_at_creation: false,
                }
            )
        };
        let model_buffer = create(sizes[0], wgpu::BufferUsages::COPY_DST);
        molla_core::gpu_write_buffer!(queue, &model_buffer, 0, bytemuck::cast_slice(&packed));
        let output = create(sizes[5], wgpu::BufferUsages::COPY_SRC);
        let mass = create(sizes[7], wgpu::BufferUsages::COPY_SRC);
        let paired_mass = paired.then(|| create(sizes[8], wgpu::BufferUsages::COPY_SRC));
        let com_jacobians = coupling.then(|| create(sizes[9], wgpu::BufferUsages::COPY_SRC));
        let status = create(
            4,
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );
        let entries: Vec<_> = sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: i < 5 },
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(size),
                },
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("experimental bias"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("experimental bias"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let text = source(n as usize, d as usize);
        let text = if paired { paired_source(text) } else { text };
        let text = if coupling {
            coupling_source(text)
        } else {
            text
        };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("experimental bias"),
            source: wgpu::ShaderSource::Wgsl(text.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("experimental bias"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("evaluate"),
            compilation_options: Default::default(),
            cache: None,
        });
        let mut buffers = vec![
            &model_buffer,
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            &output,
            &status,
            &mass,
        ];
        if let Some(mass) = &paired_mass {
            buffers.push(mass);
        }
        if let Some(jacobians) = &com_jacobians {
            buffers.push(jacobians);
        }
        let entries: Vec<_> = buffers
            .iter()
            .zip(sizes)
            .enumerate()
            .map(|(i, (buffer, size))| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(size),
                }),
            })
            .collect();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("experimental bias"),
            layout: &layout,
            entries: &entries,
        });
        Ok(Self {
            pipeline,
            group,
            output,
            status,
            mass,
            paired_mass,
            com_jacobians,
            snapshot: packed,
            signature,
        })
    }

    /// Encodes one dispatch without allocating buffers or transferring host data.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }

    /// Returns paired bias; initially zero and unchanged on a fault.
    pub fn bias(&self) -> &wgpu::Buffer {
        &self.output
    }

    /// Returns a symmetric row-major f32 matrix accumulated in f64, preserved on faults.
    pub fn mass_matrix(&self) -> &wgpu::Buffer {
        &self.mass
    }

    /// Paired row-major mass, when requested; its high halves equal the f32 mirror.
    pub fn paired_mass_matrix(&self) -> Option<&wgpu::Buffer> {
        self.paired_mass.as_ref()
    }

    /// World-axis COM Jacobians in global body order; unchanged on faults.
    pub fn com_jacobians(&self) -> Option<&wgpu::Buffer> {
        self.com_jacobians.as_ref()
    }

    /// Rejects changed physical parameters, topology or coordinate layout before encoding.
    pub fn validate_model(&self, model: &Model) -> Result<()> {
        let packed = pack(model)?;
        if bytemuck::cast_slice::<Body, u8>(&packed)
            != bytemuck::cast_slice::<Body, u8>(&self.snapshot)
            || signature(model)? != self.signature
        {
            return Err(Error::Gpu(
                "experimental bias model changed; rebuild prepared prediction".into(),
            ));
        }
        Ok(())
    }

    /// Returns sticky status: 0 no fault, 1 invalid state, 2 nonfinite result.
    pub fn status(&self) -> &wgpu::Buffer {
        &self.status
    }

    /// Clears status without changing the previous output.
    pub fn reset(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.status, 0, None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generalized_bias_model_contract() {
        use molla_core::BodyId;
        use molla_math::Transform;
        use molla_sim::{BodyParams, ModelBuilder};
        let mut builder = ModelBuilder::new();
        let root = builder.add_body(BodyParams::new());
        builder
            .add_joint_free(BodyId::NONE, root, Transform::IDENTITY, Transform::IDENTITY)
            .unwrap();
        let hinge = builder.add_body(BodyParams::new());
        builder
            .add_joint_revolute(
                root,
                hinge,
                Transform::IDENTITY,
                Transform::IDENTITY,
                Vec3::Z,
            )
            .unwrap();
        let slider = builder.add_body(BodyParams::new());
        builder
            .add_joint_prismatic(
                hinge,
                slider,
                Transform::IDENTITY,
                Transform::IDENTITY,
                Vec3::X,
            )
            .unwrap();
        let fixed = builder.add_body(BodyParams::new());
        builder
            .add_joint_fixed(slider, fixed, Transform::IDENTITY, Transform::IDENTITY)
            .unwrap();
        let mut model = builder.build().unwrap();
        let packed = pack(&model).unwrap();
        let prior = signature(&model).unwrap();
        let mass = model.body_mass.host().unwrap()[0];
        model.body_mass.host_mut().unwrap()[0] = mass + 2f64.powi(-40);
        assert_eq!(
            bytemuck::cast_slice::<Body, u8>(&packed),
            bytemuck::cast_slice::<Body, u8>(&pack(&model).unwrap())
        );
        assert_ne!(prior, signature(&model).unwrap());
        model.body_mass.host_mut().unwrap()[0] = mass;
        assert_eq!(packed.len(), 4);
        assert_eq!(packed[3].metadata, [2, 3, 3, 8]);
        model.joint_reversed.host_mut().unwrap()[1] = 1;
        assert!(pack(&model).is_err());
        model.joint_reversed.host_mut().unwrap()[1] = 0;
        model.joint_enabled.host_mut().unwrap()[1] = 0;
        assert!(pack(&model).is_err());
        model.joint_enabled.host_mut().unwrap()[1] = 1;
        model.joint_limit_lower.host_mut().unwrap()[6] = 0.0;
        assert!(pack(&model).is_err());
        model.joint_limit_lower.host_mut().unwrap()[6] = f64::NEG_INFINITY;
        model.body_flags.host_mut().unwrap()[2] = 1;
        assert!(pack(&model).is_err());
        model.body_flags.host_mut().unwrap()[2] = 0;
        model.joint_parent.host_mut().unwrap()[0] = 0;
        assert!(pack(&model).is_err());
        model.joint_parent.host_mut().unwrap()[0] = -1;
        model.body_inertia.host_mut().unwrap()[2].x_axis.x = -1.0;
        assert!(pack(&model).is_err());
    }

    #[test]
    fn generalized_bias_source_contract() {
        assert_eq!(std::mem::size_of::<Body>(), 192);
        for (b, d) in [(1, 6), (6, 10), (64, 32)] {
            let text = source(b, d);
            for text in [
                text.clone(),
                paired_source(text.clone()),
                coupling_source(paired_source(text)),
            ] {
                let module = naga::front::wgsl::parse_str(&text)
                    .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&text)));
                naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::all(),
                )
                .validate(&module)
                .unwrap();
            }
        }
    }
}
