//! Device radial and cone constraints over connected articulated islands.

use crate::distance_constraints::{has_scalar_bounds, scalar_bound_axes};
use crate::rigid_islands::RigidIslands;
use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_kernels::{
    JacobianDispatch, JacobianKernel, JacobianRequest, MassMatrixDispatch, MassMatrixKernel,
    QuaternionFkBindings, QuaternionFkDispatch, QuaternionFkKernel,
};
use molla_math::{Quat, Real, Vec3};
use molla_sim::runtime::{DistanceLimits, JointHandle, RigidScene, SphericalConeLimits};
use molla_sim::{JointType, State};
use std::collections::BTreeMap;

#[path = "scalar_joint_publication_gpu.rs"]
mod publication;
pub(crate) use publication::ScalarJointPublication;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Endpoint {
    body: i32,
    local: u32,
    width: u32,
    jacobian: u32,
    root_coordinate: i32,
    boundary: u32,
    padding: [u32; 2],
}
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Row {
    parent: Endpoint,
    child: Endpoint,
    /// Radial anchors or cone frames; radial w stores orientation and bound residue.
    a: [f32; 4],
    b: [f32; 4],
    /// Radial 1–6, scalar 7/8, cone 9/10, hard 11/12, drive 13/14, tree speed 15/16, XYZ 17/18, loop speed 19–22.
    kind: u32,
    coordinate: u32,
    /// Scalar velocity index or one plus the cone's relative quaternion coordinate offset.
    dof: u32,
    bound: f32,
    primary: [f32; 4],
    transverse0: [f32; 4],
    transverse1: [f32; 4],
    angles: [f32; 2],
    enabled: [u32; 2],
    a_low: [f32; 4],
    b_low: [f32; 4],
    primary_low: [f32; 4],
    transverse0_low: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<Row>() == 240);

fn isolated_response_rank(model: &molla_sim::Model, begin: usize, end: usize) -> Result<u32> {
    let offsets = model.joint_dof_offset.host()?;
    if end != begin + 1 || offsets[end] - offsets[begin] != 6 {
        return Ok(u32::MAX);
    }
    let kind = model.joint_type.host()?[begin];
    if !matches!(kind, v if v == JointType::Free as u32 || v == JointType::Distance as u32 || v == JointType::D6 as u32)
    {
        return Ok(u32::MAX);
    }
    let first = offsets[begin] as usize;
    let lower = model.joint_limit_lower.host()?;
    let upper = model.joint_limit_upper.host()?;
    if kind == JointType::D6 as u32
        && (first..first + 3).any(|d| lower[d].is_finite() && lower[d] == upper[d])
    {
        return Ok(u32::MAX);
    }
    let body = model.joint_child.host()?[begin] as usize;
    if model.body_flags.host()?[body] & 3 != 0 || model.body_mass.host()?[body] <= 0.0 {
        return Ok(u32::MAX);
    }
    let support = model.body_inertia.host()?[body] * model.body_inv_inertia.host()?[body];
    let trace = support.x_axis.x + support.y_axis.y + support.z_axis.z;
    if !trace.is_finite() || (trace - trace.round()).abs() > 1e-6 {
        return Err(Error::Gpu("invalid isolated inertia support rank".into()));
    }
    Ok(3 + trace.round().clamp(0.0, 3.0) as u32)
}

#[cfg(test)]
mod isolated_response_rank_tests {
    use super::isolated_response_rank;
    use molla_math::{Mat3, Quat, Transform, Vec3};
    use molla_sim::joint_control::{JointAxes, JointAxis};
    use molla_sim::runtime::{BodyDesc, JointDesc, RigidScene};
    use molla_sim::{JointType, MassProperties};

    #[test]
    fn root_support_rank_preserves_scaled_rotated_and_weak_inertia_modes() {
        for mass in [1e-6, 1.0, 1e6] {
            for rotation in [
                Quat::IDENTITY,
                Quat::from_rotation_z(0.4) * Quat::from_rotation_x(-0.2),
            ] {
                let axes = Mat3::from_quat(rotation);
                for (diagonal, rank) in [
                    (Vec3::ZERO, 3),
                    (Vec3::new(0.0, 1.0, 1.0), 5),
                    (Vec3::ONE, 6),
                ] {
                    let mut scene = RigidScene::new().unwrap();
                    scene
                        .insert_body(BodyDesc {
                            additional_mass: Some(MassProperties {
                                mass,
                                local_com: Vec3::new(0.1, -0.2, 0.3),
                                inertia: axes
                                    * Mat3::from_diagonal(diagonal * mass)
                                    * axes.transpose(),
                            }),
                            ..Default::default()
                        })
                        .unwrap();
                    assert_eq!(scene.model().joint_count, 1);
                    assert_eq!(isolated_response_rank(scene.model(), 0, 1).unwrap(), rank);
                }
            }
            let mut scene = RigidScene::new().unwrap();
            scene
                .insert_body(BodyDesc {
                    additional_mass: Some(MassProperties {
                        mass,
                        local_com: Vec3::ZERO,
                        inertia: Mat3::from_diagonal(Vec3::new(1e-10, 1.0, 1.0) * mass),
                    }),
                    ..Default::default()
                })
                .unwrap();
            assert_eq!(isolated_response_rank(scene.model(), 0, 1).unwrap(), 6);
            assert!(
                MassProperties {
                    mass,
                    local_com: Vec3::ZERO,
                    inertia: Mat3::from_diagonal(Vec3::new(0.0, 0.0, mass)),
                }
                .validate()
                .is_err()
            );
        }
    }

    #[test]
    fn partial_angular_locks_preserve_rank_but_linear_locks_use_general_response() {
        let mut scene = RigidScene::new().unwrap();
        let body = scene
            .insert_body(BodyDesc {
                additional_mass: Some(MassProperties {
                    mass: 1.0,
                    local_com: Vec3::ZERO,
                    inertia: Mat3::from_diagonal(Vec3::new(0.0, 1.0, 1.0)),
                }),
                ..Default::default()
            })
            .unwrap();
        let joint = scene
            .insert_joint(JointDesc {
                parent: None,
                child: body,
                kind: JointType::D6,
                axis: Vec3::X,
                frame_parent: Transform::IDENTITY,
                frame_child: Transform::IDENTITY,
            })
            .unwrap();
        scene
            .set_joint_locked_axes(joint, JointAxes::NONE.with(JointAxis::AngZ))
            .unwrap();
        assert_eq!(isolated_response_rank(scene.model(), 0, 1).unwrap(), 5);
        scene
            .set_body_mass_properties(
                body,
                MassProperties {
                    mass: 1.0,
                    local_com: Vec3::ZERO,
                    inertia: Mat3::IDENTITY,
                },
            )
            .unwrap();
        assert_eq!(isolated_response_rank(scene.model(), 0, 1).unwrap(), 6);
        scene
            .set_joint_locked_axes(joint, JointAxes::NONE.with(JointAxis::LinX))
            .unwrap();
        assert_eq!(
            isolated_response_rank(scene.model(), 0, 1).unwrap(),
            u32::MAX
        );
    }
}

// A connected graph with C contact edges spans at most C+1 constraint components.
fn contact_scalar_capacity(counts: &[u32], contacts: u32) -> Result<u32> {
    let mut counts = counts.to_vec();
    counts.sort_unstable_by(|a, b| b.cmp(a));
    let components = (u64::from(contacts) + 1).min(counts.len() as u64) as usize;
    counts
        .into_iter()
        .take(components)
        .try_fold(0u32, |sum, n| {
            sum.checked_add(n)
                .ok_or_else(|| Error::Gpu("joint/contact block count overflow".into()))
        })
}

#[cfg(test)]
mod contact_capacity_tests {
    use super::contact_scalar_capacity;

    #[test]
    fn independent_constraint_components_do_not_pad_every_contact_island() {
        for count in [1, 100, 1000, 10000] {
            assert_eq!(
                contact_scalar_capacity(&vec![6; count], 8).unwrap(),
                6 * count.min(9) as u32
            );
        }
    }

    #[test]
    fn contact_edges_retain_the_largest_reachable_joint_row_sum() {
        let counts = [2, 11, 3, 7, 5];
        assert_eq!(contact_scalar_capacity(&counts, 0).unwrap(), 11);
        assert_eq!(contact_scalar_capacity(&counts, 1).unwrap(), 18);
        assert_eq!(contact_scalar_capacity(&counts, 2).unwrap(), 23);
        assert_eq!(contact_scalar_capacity(&counts, u32::MAX).unwrap(), 28);
        assert_eq!(contact_scalar_capacity(&[], 8).unwrap(), 0);
        assert!(contact_scalar_capacity(&[u32::MAX, 1], 1).is_err());
    }
}

enum LinkLimits {
    Distance(DistanceLimits),
    Cone(SphericalConeLimits),
    Rigid,
    Velocity,
}

#[derive(Clone, Copy)]
pub(crate) struct DistancePrecision<'a> {
    pub coordinates: [&'a wgpu::Buffer; 2],
    pub positions: [&'a wgpu::Buffer; 2],
}

pub(crate) struct DistanceGpu {
    mass: [Option<MassMatrixDispatch>; 2],
    jacobian: [Option<JacobianDispatch>; 2],
    orientations: Option<[QuaternionFkDispatch; 2]>,
    pipeline: Option<wgpu::ComputePipeline>,
    groups: [wgpu::BindGroup; 2],
    row_buffer: wgpu::Buffer,
    rows: Vec<Row>,
    params: wgpu::Buffer,
    pub samples: wgpu::Buffer,
    pub row_forces: wgpu::Buffer,
    pub owners: Vec<Option<JointHandle>>,
    pub angular_rows: Vec<bool>,
    island_count: u32,
    island_articulations: Vec<Vec<usize>>,
    scalar_island_rows: Vec<u32>,
    active_buffer: wgpu::Buffer,
    active_upload: Vec<u32>,
    scalar_export: Option<ScalarExport>,
}

struct DistanceProgram {
    layout: wgpu::BindGroupLayout,
    pipeline: Option<wgpu::ComputePipeline>,
    export_pipeline: Option<wgpu::ComputePipeline>,
    verification: Option<wgpu::ComputePipeline>,
}

impl DistanceProgram {
    fn new(device: &wgpu::Device, export_rows: bool, interval_prediction: bool) -> Self {
        let entries: Vec<_> = (0..28)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 11 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding < 7
                                || (12..16).contains(&binding)
                                || binding == 21
                                || binding == 22
                                || binding >= 24,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("distance layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("distance pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        #[cfg(test)]
        trace_prepare("shader module");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("runtime distance"),
            source: wgpu::ShaderSource::Wgsl(
                concat!(
                    include_str!("../../molla-kernels/shaders/cone_geometry.wgsl"),
                    include_str!("../../molla-kernels/shaders/quaternion_precision.wgsl"),
                    include_str!("../../molla-kernels/shaders/joint_euler_chart.wgsl"),
                    include_str!("../../molla-kernels/shaders/euler_bound_precision.wgsl"),
                    include_str!("../../molla-kernels/shaders/joint_limit_geometry.wgsl"),
                    include_str!("../../molla-kernels/shaders/featherstone_mass_inverse.wgsl"),
                    include_str!("../../molla-kernels/shaders/runtime_distance.wgsl")
                )
                .into(),
            ),
        });
        let pipeline = (!export_rows).then(|| {
            #[cfg(test)]
            trace_prepare("standalone solve pipeline");
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("runtime distance"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("solve"),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        #[cfg(test)]
        trace_prepare("scalar export pipeline");
        let export_pipeline = export_rows.then(|| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("runtime scalar rows"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("prepare_rows"),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let verification = interval_prediction.then(|| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("contact angular prediction verification"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("verify_prediction"),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        Self {
            layout,
            pipeline,
            export_pipeline,
            verification,
        }
    }
}

struct ScalarExport {
    pipeline: wgpu::ComputePipeline,
    verification: Option<wgpu::ComputePipeline>,
    prediction_displacement: Option<wgpu::Buffer>,
    rows: wgpu::Buffer,
    positions: wgpu::Buffer,
    jacobians: wgpu::Buffer,
    directions: wgpu::Buffer,
}

fn buffer(device: &wgpu::Device, queue: &wgpu::Queue, bytes: &[u8], size: u64) -> wgpu::Buffer {
    let buffer = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some("molla distance data"),
            size: size.max(bytes.len() as u64).max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }
    );
    if !bytes.is_empty() {
        molla_core::gpu_write_buffer!(queue, &buffer, 0, bytes);
    }
    buffer
}

#[cfg(test)]
fn trace_prepare(stage: &str) {
    if std::env::var_os("MOLLA_DISTANCE_PREPARE_TRACE").is_some() {
        eprintln!("distance_prepare stage={stage}");
    }
}

impl DistanceGpu {
    pub(crate) fn is_rigid_row(&self, index: usize) -> bool {
        matches!(self.rows[index].kind, 11..=14 | 19..=22)
    }

    pub(crate) fn activation_layout(&self) -> (&wgpu::Buffer, &[Vec<usize>]) {
        (&self.active_buffer, &self.island_articulations)
    }

    pub fn set_active(&mut self, queue: &wgpu::Queue, selected: &[bool]) -> Result<()> {
        for (i, arts) in self.island_articulations.iter().enumerate() {
            let count = arts.iter().filter(|&&art| selected[art]).count();
            if count != 0 && count != arts.len() {
                return Err(Error::Solver(
                    "GPU activation splits a distance dependency".into(),
                ));
            }
            self.active_upload[i] = u32::from(arts.is_empty() || count != 0);
        }
        molla_core::gpu_write_buffer!(
            queue,
            &self.active_buffer,
            0,
            bytemuck::cast_slice(&self.active_upload),
        );
        Ok(())
    }

    pub fn bind_groups_created(&self) -> u64 {
        2 + 3 * self.mass.iter().filter(|m| m.is_some()).count() as u64
            + 4 * self.jacobian.iter().filter(|j| j.is_some()).count() as u64
            + 2 * u64::from(self.orientations.is_some())
    }

    pub(crate) fn contact_scalar_capacity(&self, contacts: u32) -> Result<u32> {
        if self.scalar_export.is_none() {
            return Ok(0);
        }
        contact_scalar_capacity(&self.scalar_island_rows, contacts)
    }

    pub fn new(
        engine: &Engine,
        scene: &RigidScene,
        states: &[State; 2],
        status: &wgpu::Buffer,
        velocity_delta: &wgpu::Buffer,
        precision: Option<DistancePrecision<'_>>,
    ) -> Result<Option<Self>> {
        Self::build(
            engine,
            scene,
            states,
            status,
            velocity_delta,
            false,
            precision,
        )
    }

    pub(crate) fn new_for_coupling(
        engine: &Engine,
        scene: &RigidScene,
        states: &[State; 2],
        status: &wgpu::Buffer,
        velocity_delta: &wgpu::Buffer,
        precision: Option<DistancePrecision<'_>>,
    ) -> Result<Option<Self>> {
        Self::build(
            engine,
            scene,
            states,
            status,
            velocity_delta,
            true,
            precision,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        engine: &Engine,
        scene: &RigidScene,
        states: &[State; 2],
        status: &wgpu::Buffer,
        velocity_delta: &wgpu::Buffer,
        export_rows: bool,
        precision: Option<DistancePrecision<'_>>,
    ) -> Result<Option<Self>> {
        let model = scene.model();
        let external = crate::distance_constraints::point_constraints(scene)
            .next()
            .is_some()
            || scene.cone_joints().next().is_some()
            || crate::rigid_loop_constraints::joints(scene)
                .next()
                .is_some()
            || crate::rigid_loop_constraints::velocity_joints(scene)
                .next()
                .is_some();
        let scalar_bounds = export_rows || external || model.requires_angular_lock_rows()?;
        let coupled: Vec<_> = (0..model.joint_count)
            .map(|joint| has_scalar_bounds(model, joint))
            .collect::<Result<_>>()?;
        if !external {
            if !scalar_bounds {
                return Ok(None);
            }
            if !coupled.iter().any(|&bounded| bounded) {
                return Ok(None);
            }
        }
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        if device.limits().max_storage_buffers_per_shader_stage < 27 {
            return Err(Error::Gpu(
                "insufficient distance row storage bindings".into(),
            ));
        }
        if precision.is_some_and(|buffers| {
            [
                (
                    buffers.coordinates,
                    model.joint_coord_total.max(1) as u64 * 4,
                ),
                (buffers.positions, model.body_count.max(1) as u64 * 16),
            ]
            .into_iter()
            .any(|(buffers, minimum)| {
                buffers.iter().any(|buffer| {
                    buffer.size() < minimum
                        || !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
                        || buffer.size() > device.limits().max_storage_buffer_binding_size
                })
            })
        }) {
            return Err(Error::Gpu("invalid distance precision buffer".into()));
        }
        #[cfg(test)]
        trace_prepare("mass kernel");
        let mass = MassMatrixKernel::new(engine, model)?;
        #[cfg(test)]
        trace_prepare("jacobian kernel");
        let jacobian = JacobianKernel::new(engine, model)?;
        #[cfg(test)]
        trace_prepare("row layout");
        let islands = RigidIslands::build(scene, [])?;
        let mut links = BTreeMap::new();
        for (id, joint, limits) in crate::distance_constraints::point_constraints(scene)
            .map(|(id, j, l)| (id, j, LinkLimits::Distance(l)))
            .chain(
                scene
                    .cone_joints()
                    .map(|(id, j, l)| (id, j, LinkLimits::Cone(l))),
            )
            .chain(
                crate::rigid_loop_constraints::joints(scene)
                    .map(|(id, joint)| (id, joint, LinkLimits::Rigid)),
            )
            .chain(
                crate::rigid_loop_constraints::velocity_joints(scene)
                    .map(|(id, joint)| (id, joint, LinkLimits::Velocity)),
            )
        {
            if !scene.joint_settings(id).unwrap().enabled
                || !scene.body(joint.child).unwrap().enabled
                || joint
                    .parent
                    .is_some_and(|p| !scene.body(p).unwrap().enabled)
            {
                continue;
            }
            let group = islands
                .body_island(joint.child)
                .or_else(|| joint.parent.and_then(|p| islands.body_island(p)));
            let arts = group.map(|g| g.articulations.clone()).unwrap_or_default();
            let separate = arts.is_empty().then_some(id);
            links
                .entry((arts, separate))
                .or_insert_with(Vec::new)
                .push((id, joint, limits));
        }
        let starts = model.articulation_start.host()?;
        let offsets = model.joint_dof_offset.host()?;
        let children = model.joint_child.host()?;
        if scalar_bounds {
            for group in islands.groups() {
                let bounded = group.articulations.iter().any(|&art| {
                    (starts[art] as usize..starts[art + 1] as usize).any(|j| coupled[j])
                });
                if bounded {
                    links
                        .entry((group.articulations.clone(), None))
                        .or_insert_with(Vec::new);
                }
            }
        }
        let mut root_coordinates = BTreeMap::new();
        let boundaries = crate::kinematic_motion::driven_ancestors(model)?;
        for (j, &body) in children.iter().enumerate() {
            if model.joint_parent.host()?[j] < 0
                && model.joint_type.host()?[j] == JointType::Free as u32
                && model.joint_x_p.host()?[j] == molla_math::Transform::IDENTITY
                && model.joint_x_c.host()?[j]
                    == molla_math::Transform::from_translation(
                        model.body_com.host()?[body as usize],
                    )
            {
                root_coordinates.insert(body, model.joint_coord_offset.host()?[j] as i32);
            }
        }
        let mut gpu_islands = Vec::<[u32; 8]>::new();
        let mut island_articulations = Vec::new();
        let mut gpu_arts = Vec::<[u32; 4]>::new();
        let mut rows = Vec::<Row>::new();
        let mut owners = Vec::new();
        let mut requests = Vec::new();
        let mut jacobian_size = 0u32;
        let mut scratch_size = 0u64;
        for ((arts, _), joints) in links {
            let art_start = gpu_arts.len() as u32;
            let row_start = rows.len() as u32;
            let mut local = 0u32;
            let mut endpoints = BTreeMap::new();
            for &art in &arts {
                let begin = starts[art] as usize;
                let end = starts[art + 1] as usize;
                let width = offsets[end] - offsets[begin];
                gpu_arts.push([offsets[begin], width, mass.mass_offset(art), local]);
                for &body in &children[begin..end] {
                    endpoints.insert(body, (art, local, width));
                }
                local += width;
            }
            for (id, joint, limits) in joints {
                let mut endpoint = |body: Option<molla_sim::runtime::BodyHandle>, point: Vec3| {
                    let body = body
                        .map(|b| scene.body_index(b).unwrap().index() as i32)
                        .unwrap_or(-1);
                    let mut result = Endpoint {
                        body,
                        root_coordinate: root_coordinates.get(&body).copied().unwrap_or(-1),
                        boundary: if body < 0 {
                            u32::MAX
                        } else {
                            boundaries[body as usize]
                        },
                        ..Default::default()
                    };
                    if let Some(&(art, local, width)) = endpoints.get(&body) {
                        result.local = local;
                        result.width = width;
                        result.jacobian = jacobian_size;
                        requests.push(JacobianRequest {
                            articulation: art as u32,
                            target_body: body as u32,
                            output_offset: jacobian_size,
                            body_q_offset: 0,
                            point_local: [point.x as f32, point.y as f32, point.z as f32, 0.0],
                        });
                        jacobian_size += 6 * width;
                    }
                    result
                };
                let limits = match limits {
                    LinkLimits::Rigid | LinkLimits::Velocity => {
                        let velocity_only = matches!(limits, LinkLimits::Velocity);
                        let parent = endpoint(joint.parent, joint.frame_parent.position);
                        let child = endpoint(Some(joint.child), joint.frame_child.position);
                        let frames = crate::rigid_loop_constraints::frames(joint);
                        for axis in molla_sim::joint_control::JointAxis::ALL {
                            let bounds = if velocity_only {
                                Vec::new()
                            } else {
                                crate::rigid_loop_constraints::bounds(scene, id, axis)?
                            };
                            for (bound, mode) in bounds {
                                let kind = if axis as usize >= 3 { 11 } else { 12 };
                                let coordinate = axis as u32 % 3;
                                let low = |values: [f64; 4]| {
                                    values.map(|value| (value - f64::from(value as f32)) as f32)
                                };
                                rows.push(Row {
                                    parent,
                                    child,
                                    a: joint.frame_parent.position.as_vec3().extend(0.0).to_array(),
                                    b: joint.frame_child.position.as_vec3().extend((bound - f64::from(bound as f32)) as f32).to_array(),
                                    primary: frames[0].rotation.as_quat().to_array(),
                                    transverse0: frames[1].rotation.as_quat().to_array(),
                                    a_low: low(joint.frame_parent.position.extend(0.0).to_array()),
                                    b_low: low(joint.frame_child.position.extend(0.0).to_array()),
                                    primary_low: low(frames[0].rotation.to_array()),
                                    transverse0_low: low(frames[1].rotation.to_array()),
                                    kind,
                                    coordinate,
                                    bound: bound as f32,
                                    enabled: [match mode { crate::distance_constraints::DistanceBound::Equality => 0, crate::distance_constraints::DistanceBound::Minimum => 1, _ => 2 }, 0],
                                    ..Default::default()
                                });
                                owners.push(Some(id));
                            }
                            if axis.dof(joint.kind).is_ok() && !velocity_only {
                                for brake in [0, 1] {
                                    let low = |values: [f64; 4]| {
                                        values.map(|value| (value - f64::from(value as f32)) as f32)
                                    };
                                    rows.push(Row {
                                        parent,
                                        child,
                                        a: joint
                                            .frame_parent
                                            .position
                                            .extend(0.0)
                                            .as_vec4()
                                            .to_array(),
                                        b: joint
                                            .frame_child
                                            .position
                                            .extend(0.0)
                                            .as_vec4()
                                            .to_array(),
                                        primary: frames[0].rotation.as_quat().to_array(),
                                        transverse0: frames[1].rotation.as_quat().to_array(),
                                        a_low: low(joint
                                            .frame_parent
                                            .position
                                            .extend(0.0)
                                            .to_array()),
                                        b_low: low(joint
                                            .frame_child
                                            .position
                                            .extend(0.0)
                                            .to_array()),
                                        primary_low: low(frames[0].rotation.to_array()),
                                        transverse0_low: low(frames[1].rotation.to_array()),
                                        kind: if axis as usize >= 3 { 13 } else { 14 },
                                        coordinate: axis as u32 % 3,
                                        enabled: [0, brake],
                                        ..Default::default()
                                    });
                                    owners.push(Some(id));
                                }
                            }
                            if let Some(cap) =
                                crate::rigid_loop_constraints::velocity_limit(scene, id, axis)
                            {
                                if !(cap as f32).is_finite() || (cap > 0.0 && cap as f32 == 0.0) {
                                    return Err(Error::Gpu(
                                        "loop velocity cap exceeds GPU precision".into(),
                                    ));
                                }
                                let low = |values: [f64; 4]| {
                                    values.map(|value| (value - f64::from(value as f32)) as f32)
                                };
                                for side in 0..if cap == 0.0 { 1 } else { 2 } {
                                    rows.push(Row {
                                        parent,
                                        child,
                                        a: joint
                                            .frame_parent
                                            .position
                                            .as_vec3()
                                            .extend(0.0)
                                            .to_array(),
                                        b: joint
                                            .frame_child
                                            .position
                                            .as_vec3()
                                            .extend((-cap - f64::from(-cap as f32)) as f32)
                                            .to_array(),
                                        a_low: low(joint
                                            .frame_parent
                                            .position
                                            .extend(0.0)
                                            .to_array()),
                                        b_low: low(joint
                                            .frame_child
                                            .position
                                            .extend(0.0)
                                            .to_array()),
                                        primary: frames[0].rotation.as_quat().to_array(),
                                        transverse0: frames[1].rotation.as_quat().to_array(),
                                        primary_low: low(frames[0].rotation.to_array()),
                                        transverse0_low: low(frames[1].rotation.to_array()),
                                        kind: if axis as usize >= 3 {
                                            19 + side
                                        } else {
                                            21 + side
                                        },
                                        coordinate: axis as u32 % 3,
                                        bound: -cap as f32,
                                        enabled: [u32::from(cap != 0.0), 0],
                                        ..Default::default()
                                    });
                                    owners.push(Some(id));
                                }
                            }
                        }
                        continue;
                    }
                    LinkLimits::Cone(limits) => {
                        if limits.is_unrestricted() {
                            continue;
                        }
                        let packed =
                            molla_kernels::ConeGpuRequest::new(limits, joint.axis, Quat::IDENTITY)?;
                        let geometry = molla_sim::spherical_cone::evaluate(
                            limits,
                            joint.axis,
                            Quat::IDENTITY,
                        )?;
                        let coordinate = if let Some(j) = scene.joint_index(id)
                            && model.joint_type.host()?[j.index()] == JointType::Ball as u32
                        {
                            model.joint_coord_offset.host()?[j.index()] + 1
                        } else {
                            0
                        };
                        let parent = endpoint(joint.parent, Vec3::ZERO);
                        let child = endpoint(Some(joint.child), Vec3::ZERO);
                        for (index, cone) in geometry.rows[..geometry.row_count].iter().enumerate()
                        {
                            rows.push(Row {
                                parent,
                                child,
                                a: joint.frame_parent.rotation.as_quat().to_array(),
                                b: joint.frame_child.rotation.as_quat().to_array(),
                                a_low: joint
                                    .frame_parent
                                    .rotation
                                    .to_array()
                                    .map(|v| (v - f64::from(v as f32)) as f32),
                                b_low: joint
                                    .frame_child
                                    .rotation
                                    .to_array()
                                    .map(|v| (v - f64::from(v as f32)) as f32),
                                kind: if cone.bilateral { 10 } else { 9 },
                                coordinate: index as u32,
                                dof: coordinate,
                                primary: packed.primary,
                                transverse0: packed.transverse0,
                                transverse1: packed.transverse1,
                                angles: packed.angles,
                                enabled: packed.enabled,
                                ..Default::default()
                            });
                            owners.push(Some(id));
                        }
                        continue;
                    }
                    LinkLimits::Distance(limits) => limits,
                };
                if limits.max.is_none() && limits.min.is_none_or(|v| v == 0.0) {
                    continue;
                }
                let parent = endpoint(joint.parent, joint.frame_parent.position);
                let child = endpoint(Some(joint.child), joint.frame_child.position);
                let a = joint
                    .frame_parent
                    .position
                    .as_vec3()
                    .extend(crate::distance_constraints::coincident_axis(joint).x as f32)
                    .to_array();
                let b = joint.frame_child.position.as_vec3().extend(0.0).to_array();
                let mut add = |kind, bound: Real| {
                    let mut b = b;
                    b[3] = (bound - bound as f32 as Real) as f32;
                    rows.push(Row {
                        parent,
                        child,
                        a,
                        b,
                        a_low: joint
                            .frame_parent
                            .position
                            .extend(0.0)
                            .to_array()
                            .map(|v| (v - f64::from(v as f32)) as f32),
                        b_low: joint
                            .frame_child
                            .position
                            .extend(0.0)
                            .to_array()
                            .map(|v| (v - f64::from(v as f32)) as f32),
                        kind,
                        bound: bound as f32,
                        ..Default::default()
                    });
                    owners.push(Some(id));
                };
                if limits.max == Some(0.0) {
                    for kind in 4..=6 {
                        add(kind, 0.0);
                    }
                } else if let (Some(lo), Some(hi)) = (limits.min, limits.max)
                    && lo == hi
                {
                    add(3, lo);
                } else {
                    if let Some(lo) = limits.min.filter(|v| *v > 0.0) {
                        add(1, lo);
                    }
                    if let Some(hi) = limits.max {
                        add(2, hi);
                    }
                }
            }
            for (i, &art) in arts.iter().enumerate() {
                let meta = gpu_arts[art_start as usize + i];
                for j in starts[art] as usize..starts[art + 1] as usize {
                    if model.joint_enabled.host()?[j] == 0 {
                        continue;
                    }
                    let joint_kind = model.joint_type.host()?[j];
                    let angular_joint =
                        joint_kind == JointType::D6 as u32 || joint_kind == JointType::Ball as u32;
                    let angular_start = if joint_kind == JointType::Ball as u32 {
                        0
                    } else {
                        3
                    };
                    let angular_bounds = if angular_joint {
                        let lower = model.joint_limit_lower.host()?;
                        let upper = model.joint_limit_upper.host()?;
                        let start = offsets[j] as usize;
                        [
                            lower[start + angular_start],
                            upper[start + angular_start],
                            lower[start + angular_start + 2],
                            upper[start + angular_start + 2],
                            lower[start + angular_start + 1],
                            upper[start + angular_start + 1],
                        ]
                    } else {
                        [0.; 6]
                    };
                    if angular_bounds
                        .iter()
                        .any(|v| v.is_nan() || (v.is_finite() && !(*v as f32).is_finite()))
                    {
                        return Err(Error::Gpu("Ball/D6 angular bounds exceed f32 range".into()));
                    }
                    let packed = angular_bounds.map(|v| v as f32);
                    let packed_low = angular_bounds.map(|v| {
                        if v.is_finite() {
                            (v - f64::from(v as f32)) as f32
                        } else {
                            0.
                        }
                    });
                    let primary = [packed[0], packed[1], packed[2], packed[3]];
                    let partial = angular_joint
                        && molla_sim::joint_control::partial_angular_lock(
                            &[angular_bounds[0], angular_bounds[4], angular_bounds[2]],
                            &[angular_bounds[1], angular_bounds[5], angular_bounds[3]],
                        );
                    let angular_points = [
                        angular_bounds[0].is_finite()
                            && (angular_bounds[1] - angular_bounds[0]).abs() < 1e-12,
                        angular_bounds[4].is_finite()
                            && (angular_bounds[5] - angular_bounds[4]).abs() < 1e-12,
                        angular_bounds[2].is_finite()
                            && (angular_bounds[3] - angular_bounds[2]).abs() < 1e-12,
                    ];
                    let point_mask = angular_points
                        .iter()
                        .enumerate()
                        .fold(0_u32, |mask, (axis, &point)| {
                            mask | (u32::from(point) << axis)
                        });
                    let pair = angular_points.iter().filter(|&&point| point).count() == 2;
                    let (pair_left, pair_right) = if pair {
                        crate::joint_angles::pair_frames(
                            Vec3::new(angular_bounds[0], angular_bounds[4], angular_bounds[2]),
                            angular_points.iter().position(|&point| !point).unwrap(),
                        )
                    } else {
                        (Quat::IDENTITY, Quat::IDENTITY)
                    };
                    let quaternion_low = |q: Quat| {
                        q.to_array()
                            .map(|value| (value - f64::from(value as f32)) as f32)
                    };
                    let first_point = angular_points.iter().position(|&point| point);
                    let primary_low = [packed_low[0], packed_low[1], packed_low[2], packed_low[3]];
                    let transverse0 = [packed[4], packed[5], 0., 0.];
                    let transverse0_low = [packed_low[4], packed_low[5], 0., 0.];
                    if scalar_bounds
                        && angular_joint
                        && coupled[j]
                        && angular_bounds.iter().any(|v| v.is_finite())
                    {
                        let phase_ranges = [
                            [
                                angular_bounds[0] + angular_bounds[2],
                                angular_bounds[1] + angular_bounds[3],
                            ],
                            [
                                angular_bounds[0] - angular_bounds[3],
                                angular_bounds[1] - angular_bounds[2],
                            ],
                        ];
                        let phase_width = (angular_bounds[1] - angular_bounds[0])
                            + (angular_bounds[3] - angular_bounds[2]);
                        let mut phase_flags = 1 | (point_mask << 1);
                        let mut phase_active = [false; 2];
                        for (pole, range) in phase_ranges.iter().enumerate() {
                            phase_active[pole] = range.iter().all(|v| v.is_finite())
                                && phase_width < std::f64::consts::TAU;
                            if phase_active[pole] {
                                if range.iter().any(|v| !(*v as f32).is_finite()) {
                                    return Err(Error::Gpu(
                                        "singular phase bounds exceed f32 range".into(),
                                    ));
                                }
                                phase_flags |= 1 << (6 + pole);
                                if phase_width.abs() < 1e-12 {
                                    phase_flags |= 1 << (4 + pole);
                                }
                            }
                        }
                        for kind in [17, 18] {
                            let targets: [f64; 2] = std::array::from_fn(|pole| {
                                if phase_active[pole] {
                                    phase_ranges[pole][usize::from(kind == 18)]
                                } else {
                                    0.
                                }
                            });
                            let positive = targets[0].sin_cos();
                            let negative = targets[1].sin_cos();
                            let a = [targets[0], targets[1], positive.0, positive.1];
                            let b = [negative.0, negative.1, 0., 0.];
                            let low =
                                |values: [f64; 4]| values.map(|v| (v - f64::from(v as f32)) as f32);
                            rows.push(Row {
                                parent: Endpoint {
                                    body: model.joint_parent.host()?[j],
                                    root_coordinate: -1,
                                    ..Default::default()
                                },
                                child: Endpoint {
                                    body: children[j],
                                    root_coordinate: -1,
                                    ..Default::default()
                                },
                                kind,
                                coordinate: model.joint_coord_offset.host()?[j]
                                    + angular_start as u32,
                                dof: meta[3] + offsets[j] + angular_start as u32 - meta[0],
                                a: a.map(|v| v as f32),
                                a_low: low(a),
                                b: b.map(|v| v as f32),
                                b_low: low(b),
                                primary,
                                primary_low,
                                transverse0,
                                transverse0_low,
                                enabled: [0, phase_flags],
                                ..Default::default()
                            });
                            owners.push(None);
                        }
                    }
                    for axis in 0..scalar_bound_axes(model.joint_type.host()?[j]) {
                        if axis >= angular_start && !scalar_bounds {
                            continue;
                        }
                        let dof = offsets[j] + axis as u32;
                        let lo = model.joint_limit_lower.host()?[dof as usize];
                        let hi = model.joint_limit_upper.host()?[dof as usize];
                        let angular = axis >= angular_start;
                        let equality = lo.is_finite() && hi.is_finite() && (hi - lo).abs() < 1e-12;
                        if equality && !(angular && partial) {
                            continue;
                        }
                        if equality && angular && pair && Some(axis - angular_start) != first_point
                        {
                            continue;
                        }
                        for (kind, bound) in [
                            (
                                if angular {
                                    if equality { 23 } else { 17 }
                                } else {
                                    7
                                },
                                lo,
                            ),
                            (
                                if angular {
                                    if equality { 24 } else { 18 }
                                } else {
                                    8
                                },
                                hi,
                            ),
                        ] {
                            if equality
                                && kind == 24
                                && !pair
                                && !(axis == angular_start + 1
                                    && lo.cos().abs() <= 1.9073486328125e-6)
                            {
                                continue;
                            }
                            if bound.is_finite() {
                                let (sine, cosine) =
                                    (bound * if pair { 0.5 } else { 1.0 }).sin_cos();
                                let mut transverse0 = transverse0;
                                transverse0[2] = (sine - f64::from(sine as f32)) as f32;
                                transverse0[3] = (cosine - f64::from(cosine as f32)) as f32;
                                let (left, right) = if !pair && angular && equality {
                                    match axis - angular_start {
                                        0 => (Quat::from_rotation_x(bound), Quat::IDENTITY),
                                        1 => (Quat::from_rotation_y(bound), Quat::IDENTITY),
                                        _ => (Quat::IDENTITY, Quat::from_rotation_z(bound)),
                                    }
                                } else {
                                    (pair_left, pair_right)
                                };
                                rows.push(Row {
                                    parent: Endpoint {
                                        body: model.joint_parent.host()?[j],
                                        root_coordinate: -1,
                                        ..Default::default()
                                    },
                                    child: Endpoint {
                                        body: children[j],
                                        root_coordinate: -1,
                                        ..Default::default()
                                    },
                                    kind,
                                    coordinate: model.joint_coord_offset.host()?[j]
                                        + if angular {
                                            angular_start as u32
                                        } else {
                                            axis as u32
                                        },
                                    dof: meta[3]
                                        + if angular {
                                            offsets[j] + angular_start as u32
                                        } else {
                                            dof
                                        }
                                        - meta[0],
                                    bound: bound as f32,
                                    b: [0., 0., 0., (bound - f64::from(bound as f32)) as f32],
                                    a: left.as_quat().to_array(),
                                    a_low: quaternion_low(left),
                                    transverse1: right.as_quat().to_array(),
                                    b_low: quaternion_low(right),
                                    angles: [sine as f32, cosine as f32],
                                    primary,
                                    primary_low,
                                    transverse0,
                                    transverse0_low,
                                    enabled: [
                                        if angular {
                                            (axis - angular_start) as u32
                                        } else {
                                            0
                                        },
                                        if angular { point_mask << 1 } else { 0 },
                                    ],
                                });
                                owners.push(None);
                            }
                        }
                    }
                    if scalar_bounds {
                        for dof in offsets[j]..offsets[j + 1] {
                            let lower = model.joint_limit_lower.host()?[dof as usize];
                            let upper = model.joint_limit_upper.host()?[dof as usize];
                            let locked = lower.is_finite()
                                && upper.is_finite()
                                && (upper - lower).abs() < 1e-12;
                            let cap = model.joint_velocity_limit.host()?[dof as usize];
                            let structural = locked
                                && (joint_kind <= 1
                                    || (joint_kind == JointType::Ball as u32 && !partial)
                                    || (joint_kind == JointType::D6 as u32
                                        && (dof - offsets[j] < 3 || !partial)));
                            if structural || !cap.is_finite() {
                                continue;
                            }
                            for kind in if cap == 0. { 15..16 } else { 15..17 } {
                                rows.push(Row {
                                    parent: Endpoint {
                                        body: model.joint_parent.host()?[j],
                                        root_coordinate: -1,
                                        ..Default::default()
                                    },
                                    child: Endpoint {
                                        body: children[j],
                                        root_coordinate: -1,
                                        ..Default::default()
                                    },
                                    kind,
                                    dof: meta[3] + dof - meta[0],
                                    bound: -cap as f32,
                                    b: [0., 0., 0., (-cap + f64::from(cap as f32)) as f32],
                                    enabled: [u32::from(cap != 0.), 0],
                                    ..Default::default()
                                });
                                owners.push(None);
                            }
                        }
                    }
                }
            }
            if rows.len() == row_start as usize {
                gpu_arts.truncate(art_start as usize);
                continue;
            }
            let count = rows.len() as u32 - row_start;
            let response_rank = if arts.len() == 1 {
                isolated_response_rank(
                    model,
                    starts[arts[0]] as usize,
                    starts[arts[0] + 1] as usize,
                )?
            } else {
                u32::MAX
            };
            gpu_islands.push([
                art_start,
                arts.len() as u32,
                row_start,
                count,
                local,
                u32::try_from(scratch_size)
                    .map_err(|_| Error::Gpu("distance scratch overflow".into()))?,
                response_rank,
                0,
            ]);
            island_articulations.push(arts);
            let n = local as u64;
            let r = count as u64;
            scratch_size = [
                n.checked_mul(n),
                r.checked_mul(n).and_then(|v| v.checked_mul(2)),
                r.checked_mul(r),
                r.checked_mul(9),
                n.checked_mul(2),
            ]
            .into_iter()
            .try_fold(scratch_size, |size, term| size.checked_add(term?))
            .ok_or_else(|| Error::Gpu("distance scratch overflow".into()))?;
        }
        if rows.is_empty() {
            return Ok(None);
        }
        if scratch_size * 4 > device.limits().max_storage_buffer_binding_size
            || gpu_islands.len().div_ceil(64)
                > device.limits().max_compute_workgroups_per_dimension as usize
            || rows
                .iter()
                .any(|r| !r.bound.is_finite() || r.a.iter().chain(&r.b).any(|v| !v.is_finite()))
        {
            return Err(Error::Gpu(
                "distance GPU capacity or f32 range exceeded".into(),
            ));
        }
        let mass_buffer = mass.create_output_buffer(device);
        let jacobian_buffer = jacobian.create_output_buffer(device, jacobian_size as usize);
        let islands_buffer = buffer(device, queue, bytemuck::cast_slice(&gpu_islands), 32);
        let active_upload = vec![1u32; gpu_islands.len()];
        let active_buffer = buffer(device, queue, bytemuck::cast_slice(&active_upload), 4);
        let arts_buffer = buffer(device, queue, bytemuck::cast_slice(&gpu_arts), 0);
        let rows_buffer = buffer(
            device,
            queue,
            bytemuck::cast_slice(&rows),
            std::mem::size_of::<Row>() as u64,
        );
        let scratch = buffer(device, queue, &[], scratch_size * 4);
        let samples = buffer(device, queue, &[], (rows.len() as u64 * 32).max(32));
        let row_forces = buffer(device, queue, &[], rows.len() as u64 * 16);
        let empty = buffer(device, queue, &[], 32);
        let has_paired_geometry = rows.iter().any(|row| {
            matches!(row.kind, 11..=14 | 19..=22)
                || (precision.is_some() && matches!(row.kind, 1..=6 | 9..=10))
        });
        let rotations = std::array::from_fn::<_, 2, _>(|_| {
            buffer(
                device,
                queue,
                &[],
                if has_paired_geometry {
                    model.body_count.max(1) as u64 * 32
                } else {
                    32
                },
            )
        });
        let centers_low: Vec<[f32; 4]> = model
            .body_com
            .host()?
            .iter()
            .map(|center| {
                center
                    .extend(0.0)
                    .to_array()
                    .map(|value| (value - f64::from(value as f32)) as f32)
            })
            .collect();
        let centers_low = buffer(device, queue, bytemuck::cast_slice(&centers_low), 16);
        let orientations = if has_paired_geometry {
            let precision = precision.ok_or_else(|| {
                Error::Gpu("paired constraint rows require retained coordinate precision".into())
            })?;
            let kernel = QuaternionFkKernel::new_with_coordinate_residues(engine, model)?;
            let rounding = buffer(device, queue, &[], model.body_count.max(1) as u64 * 4);
            let prepare = |i: usize| {
                kernel.prepare(
                    device,
                    QuaternionFkBindings {
                        coordinates: if model.joint_coord_total == 0 {
                            None
                        } else {
                            Some(states[i].joint_q.device_buffer_current()?)
                        },
                        coordinate_residues: Some(precision.coordinates[i]),
                        poses: states[i].body_q.device_buffer_current()?,
                        rotations: &rotations[i],
                        rounding: &rounding,
                        status,
                    },
                )
            };
            Some([prepare(0)?, prepare(1)?])
        } else {
            None
        };
        let empty_velocity = buffer(device, queue, &[], 16);
        let scalar_size = if export_rows { rows.len() as u64 } else { 0 };
        let scalar_jacobian_bytes = scalar_size
            .checked_mul(model.joint_dof_total as u64)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| Error::Gpu("scalar Jacobian capacity overflow".into()))?;
        if scalar_jacobian_bytes.max(scalar_size * 32)
            > device.limits().max_storage_buffer_binding_size
        {
            return Err(Error::Gpu("scalar Jacobian capacity exceeded".into()));
        }
        let scalar_rows = buffer(device, queue, &[], (scalar_size * 32).max(32));
        let scalar_positions = buffer(device, queue, &[], (scalar_size * 32).max(32));
        let scalar_jacobians = buffer(device, queue, &[], scalar_jacobian_bytes.max(4));
        let scalar_directions = buffer(device, queue, &[], (scalar_size * 16).max(16));
        let interval_prediction = export_rows
            && rows.iter().any(|row| {
                matches!(row.kind, 17 | 18 | 23 | 24) && ((row.enabled[1] >> 1) & 7) == 0
            });
        let prediction_displacement = interval_prediction
            .then(|| buffer(device, queue, &[], model.joint_dof_total.max(1) as u64 * 4));
        let params = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("distance params"),
                size: 32,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        molla_core::gpu_write_buffer!(
            queue,
            &params,
            16,
            bytemuck::cast_slice(&[
                model.joint_dof_total as u32,
                u32::from(precision.is_some()),
                0,
                0,
            ]),
        );
        #[cfg(test)]
        trace_prepare("cached row program");
        #[cfg(not(target_arch = "wasm32"))]
        let variant = format!("export={export_rows};verification={interval_prediction}");
        #[cfg(not(target_arch = "wasm32"))]
        let program = engine.cached_gpu_program(&variant, |device| {
            Ok(DistanceProgram::new(
                device,
                export_rows,
                interval_prediction,
            ))
        })?;
        #[cfg(target_arch = "wasm32")]
        let program = DistanceProgram::new(device, export_rows, interval_prediction);
        let layout = &program.layout;
        let pipeline = program.pipeline.clone();
        let export_pipeline = program.export_pipeline.clone();
        let verification = program.verification.clone();
        let groups = std::array::from_fn(|i| {
            let input = &states[i];
            let output = &states[1 - i];
            let buffers = [
                &islands_buffer,
                &arts_buffer,
                &rows_buffer,
                &mass_buffer,
                &jacobian_buffer,
                input.body_q.device_buffer().unwrap_or(&empty),
                input.joint_q.device_buffer().unwrap_or(&empty),
                output.joint_qd.device_buffer().unwrap_or(&empty_velocity),
                &scratch,
                &samples,
                status,
                &params,
                &active_buffer,
                input.body_qd.device_buffer().unwrap_or(&empty),
                model.body_com.device_buffer().unwrap_or(&empty),
                input.joint_qd.device_buffer().unwrap_or(&empty),
                velocity_delta,
                &row_forces,
                &scalar_rows,
                &scalar_jacobians,
                &scalar_directions,
                precision.map_or(&empty, |buffers| buffers.coordinates[i]),
                model
                    .body_kinematic_acceleration
                    .device_buffer()
                    .unwrap_or(&empty),
                &scalar_positions,
                precision.map_or(&empty, |buffers| buffers.positions[i]),
                &rotations[i],
                &centers_low,
                prediction_displacement.as_ref().unwrap_or(&empty),
            ];
            let entries: Vec<_> = buffers
                .iter()
                .enumerate()
                .map(|(binding, b)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: b.as_entire_binding(),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("distance group"),
                layout,
                entries: &entries,
            })
        });
        let prepare_mass = |i: usize| -> Result<_> {
            if model.joint_dof_total == 0 {
                return Ok(None);
            }
            mass.prepare_on_device(device, queue, model, &states[i], &mass_buffer)
                .map(Some)
        };
        let prepare_jacobian = |i: usize| {
            jacobian.prepare_with_body_q_on_device(
                device,
                queue,
                model,
                states[i].body_q.device_buffer().unwrap_or(&empty),
                &requests,
                &jacobian_buffer,
            )
        };
        #[cfg(test)]
        trace_prepare("mass and jacobian bindings");
        let mass = [prepare_mass(0)?, prepare_mass(1)?];
        let jacobian = [prepare_jacobian(0)?, prepare_jacobian(1)?];
        #[cfg(test)]
        trace_prepare("ready");
        Ok(Some(Self {
            mass,
            jacobian,
            orientations,
            pipeline,
            groups,
            params,
            samples,
            row_forces,
            owners,
            angular_rows: rows
                .iter()
                .map(|row| matches!(row.kind, 9..=11 | 13 | 19..=20))
                .collect(),
            island_count: gpu_islands.len() as u32,
            island_articulations,
            scalar_island_rows: gpu_islands.iter().map(|island| island[3]).collect(),
            active_buffer,
            active_upload,
            scalar_export: export_pipeline.map(|pipeline| ScalarExport {
                pipeline,
                verification,
                prediction_displacement,
                rows: scalar_rows,
                positions: scalar_positions,
                jacobians: scalar_jacobians,
                directions: scalar_directions,
            }),
            row_buffer: rows_buffer,
            rows,
        }))
    }

    pub(crate) fn refresh_feedback(
        &mut self,
        queue: &wgpu::Queue,
        scene: &RigidScene,
    ) -> Result<()> {
        use molla_sim::joint_control::{JointAxis, JointMotor};
        for (i, row) in self
            .rows
            .iter_mut()
            .enumerate()
            .filter(|(_, r)| matches!(r.kind, 13..=14))
        {
            let axis = JointAxis::ALL[row.coordinate as usize + if row.kind == 13 { 3 } else { 0 }];
            let setting = scene.joint_axis_settings(self.owners[i].unwrap(), axis)?;
            let motor = if row.enabled[1] == 1 {
                JointMotor {
                    mode: if setting.brake > 0.0 {
                        molla_sim::JointTargetMode::Velocity
                    } else {
                        molla_sim::JointTargetMode::None
                    },
                    damping: setting.brake,
                    ..Default::default()
                }
            } else {
                setting.motor.unwrap_or_default()
            };
            let before = (
                row.bound,
                row.b[3],
                row.transverse1,
                row.angles,
                row.enabled[0],
            );
            row.bound = motor.target_position as f32;
            row.b[3] = (motor.target_position - f64::from(row.bound)) as f32;
            row.transverse1 = [
                0.0,
                motor.target_velocity as f32,
                motor.max_force as f32,
                motor.model as u32 as f32,
            ];
            row.angles = [
                if motor.mode == molla_sim::JointTargetMode::Velocity {
                    0.0
                } else {
                    motor.stiffness as f32
                },
                motor.damping as f32,
            ];
            row.enabled[0] = motor.mode as u32;
            if !row.bound.is_finite()
                || !row.b[3].is_finite()
                || !row.transverse1[1].is_finite()
                || row.angles.iter().any(|v| !v.is_finite())
                || row.transverse1[2].is_nan()
                || (motor.max_force.is_finite() && !row.transverse1[2].is_finite())
            {
                return Err(Error::Gpu("rigid-loop control exceeds f32 range".into()));
            }
            if before
                != (
                    row.bound,
                    row.b[3],
                    row.transverse1,
                    row.angles,
                    row.enabled[0],
                )
            {
                molla_core::gpu_write_buffer!(
                    queue,
                    &self.row_buffer,
                    (i * std::mem::size_of::<Row>()) as u64,
                    bytemuck::bytes_of(row),
                );
            }
        }
        Ok(())
    }

    pub(crate) fn scalar_buffers(
        &self,
    ) -> Option<crate::constraint_rows_gpu::ScalarConstraintBuffers<'_>> {
        self.scalar_export
            .as_ref()
            .map(|e| crate::constraint_rows_gpu::ScalarConstraintBuffers {
                rows: &e.rows,
                jacobians: &e.jacobians,
                capacity: self.owners.len() as u32,
            })
    }

    pub(crate) fn scalar_publication(
        &self,
        engine: &Engine,
        solutions: &wgpu::Buffer,
        status: &wgpu::Buffer,
    ) -> Result<ScalarJointPublication> {
        let export = self
            .scalar_export
            .as_ref()
            .ok_or_else(|| Error::Gpu("scalar row export is not prepared".into()))?;
        ScalarJointPublication::new(
            engine,
            &export.rows,
            &export.directions,
            solutions,
            &self.samples,
            &self.row_forces,
            &self.params,
            status,
            self.owners.len() as u32,
        )
    }

    pub(crate) fn scalar_position_buffers(
        &self,
    ) -> Option<crate::constraint_rows_gpu::ScalarPositionBuffers<'_>> {
        self.scalar_export
            .as_ref()
            .map(|e| crate::constraint_rows_gpu::ScalarPositionBuffers {
                rows: &e.positions,
                jacobians: &e.jacobians,
                capacity: self.owners.len() as u32,
            })
    }

    pub(crate) fn prepare_scalar_rows(
        &self,
        engine: &Engine,
        input_index: usize,
        dt: Real,
        substeps: usize,
        substep: usize,
    ) -> Result<()> {
        let export = self
            .scalar_export
            .as_ref()
            .ok_or_else(|| Error::Gpu("scalar row export is not prepared".into()))?;
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        molla_core::gpu_write_buffer!(
            queue,
            &self.params,
            0,
            bytemuck::cast_slice(&[
                (dt as f32).to_bits(),
                (1.0 / substeps as f32).to_bits(),
                u32::try_from(substep)
                    .map_err(|_| Error::Gpu("distance substep count overflow".into()))?,
                self.island_count,
            ]),
        );
        molla_core::gpu_write_buffer!(
            queue,
            &self.params,
            24,
            bytemuck::bytes_of(&u32::from(export.prediction_displacement.is_some()))
        );
        let mut encoder = device.create_command_encoder(&Default::default());
        if let Some(orientations) = &self.orientations {
            orientations[input_index].encode(&mut encoder);
        }
        for buffer in [
            &export.rows,
            &export.positions,
            &export.jacobians,
            &export.directions,
        ] {
            encoder.clear_buffer(buffer, 0, None);
        }
        if let Some(jacobian) = &self.jacobian[input_index] {
            jacobian.encode(&mut encoder);
        }
        if self.rows.iter().any(|row| {
            matches!(row.kind, 13..=14)
                && row.transverse1[3] == 1.0
                && !matches!(row.enabled[0], 0 | 4)
                && row.angles.iter().any(|v| *v > 0.)
        }) && let Some(mass) = &self.mass[input_index]
        {
            mass.encode(&mut encoder);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&export.pipeline);
            pass.set_bind_group(0, &self.groups[input_index], &[]);
            pass.dispatch_workgroups(self.island_count.div_ceil(64).max(1), 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        }
        queue.submit([encoder.finish()]);
        Ok(())
    }

    pub(crate) fn needs_contact_prediction(&self) -> bool {
        self.scalar_export
            .as_ref()
            .is_some_and(|export| export.verification.is_some())
    }

    pub(crate) fn retain_prediction_displacement(
        &self,
        engine: &Engine,
        source: Option<&wgpu::Buffer>,
    ) -> Result<()> {
        let Some(destination) = self
            .scalar_export
            .as_ref()
            .and_then(|export| export.prediction_displacement.as_ref())
        else {
            return Ok(());
        };
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        if let Some(source) = source {
            if source.size() < destination.size() {
                return Err(Error::Gpu(
                    "contact prediction displacement capacity exceeded".into(),
                ));
            }
            encoder.copy_buffer_to_buffer(source, 0, destination, 0, destination.size());
        } else {
            encoder.clear_buffer(destination, 0, None);
        }
        engine.queue().unwrap().submit([encoder.finish()]);
        Ok(())
    }

    pub(crate) fn verify_contact_prediction(
        &self,
        engine: &Engine,
        input_index: usize,
    ) -> Result<()> {
        let pipeline = self
            .scalar_export
            .as_ref()
            .and_then(|export| export.verification.as_ref())
            .ok_or_else(|| Error::Gpu("contact angular prediction is not prepared".into()))?;
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.groups[input_index], &[]);
            pass.dispatch_workgroups(self.island_count.div_ceil(64).max(1), 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        }
        engine.queue().unwrap().submit([encoder.finish()]);
        Ok(())
    }

    pub fn dispatch(
        &self,
        engine: &Engine,
        input_index: usize,
        dt: Real,
        substeps: usize,
        substep: usize,
    ) -> Result<()> {
        let pipeline = self.pipeline.as_ref().ok_or_else(|| {
            Error::Gpu("standalone distance solve is not prepared for coupled rows".into())
        })?;
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        molla_core::gpu_write_buffer!(
            queue,
            &self.params,
            0,
            bytemuck::cast_slice(&[
                (dt as f32).to_bits(),
                (1.0 / substeps as f32).to_bits(),
                u32::try_from(substep)
                    .map_err(|_| Error::Gpu("distance substep count overflow".into()))?,
                self.island_count,
            ]),
        );
        let mut encoder = device.create_command_encoder(&Default::default());
        if let Some(orientations) = &self.orientations {
            orientations[input_index].encode(&mut encoder);
        }
        encoder.clear_buffer(&self.row_forces, 0, None);
        if let Some(mass) = &self.mass[input_index] {
            mass.encode(&mut encoder);
        }
        if let Some(jacobian) = &self.jacobian[input_index] {
            jacobian.encode(&mut encoder);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.groups[input_index], &[]);
            pass.dispatch_workgroups(self.island_count.div_ceil(64).max(1), 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        }
        queue.submit([encoder.finish()]);
        Ok(())
    }
}
