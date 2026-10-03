//! Retained coupled compliant-loop response and transient pre-predictor loads.

use crate::rigid_islands::RigidIslands;
use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_kernels::{
    CompliantLoopBindings, CompliantLoopDispatch, CompliantLoopKernel, CompliantLoopRequest,
    JacobianDispatch, JacobianKernel, JacobianRequest, MassMatrixDispatch, MassMatrixKernel,
    QuaternionFkBindings, QuaternionFkDispatch, QuaternionFkKernel,
};
use molla_math::{Quat, Real, Vec3};
use molla_sim::joint_control::JointAxis;
use molla_sim::runtime::{JointHandle, RigidScene};
use molla_sim::{JointType, State};
use std::collections::BTreeMap;

const ROWS_PER_AXIS: usize = 3;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Endpoint {
    local: u32,
    width: u32,
    jacobian: u32,
    padding: u32,
}

pub(crate) struct LoopGpu {
    mass: [MassMatrixDispatch; 2],
    jacobians: [Option<JacobianDispatch>; 2],
    geometry: [CompliantLoopDispatch; 2],
    orientations: [QuaternionFkDispatch; 2],
    solve: wgpu::ComputePipeline,
    publish: wgpu::ComputePipeline,
    apply: wgpu::ComputePipeline,
    groups: [wgpu::BindGroup; 2],
    params: wgpu::Buffer,
    request_buffer: wgpu::Buffer,
    requests: Vec<CompliantLoopRequest>,
    pub owners: Vec<JointHandle>,
    pub wrenches: wgpu::Buffer,
    pub parent_wrenches: wgpu::Buffer,
    impulses: wgpu::Buffer,
    persistent: wgpu::Buffer,
    active_buffer: wgpu::Buffer,
    active_upload: Vec<u32>,
    island_articulations: Vec<Vec<usize>>,
    bodies: u32,
}

fn data(device: &wgpu::Device, queue: &wgpu::Queue, bytes: &[u8], minimum: u64) -> wgpu::Buffer {
    let buffer = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some("compliant loop data"),
            size: (bytes.len() as u64).max(minimum).max(4),
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

fn request(
    scene: &RigidScene,
    id: JointHandle,
    axis: JointAxis,
    term: usize,
) -> Result<CompliantLoopRequest> {
    let joint = scene.joint(id).unwrap();
    let parent = joint
        .parent
        .map_or(-1, |body| scene.body_index(body).unwrap().index() as i32);
    let child = scene.body_index(joint.child).unwrap().index() as u32;
    let mut frames = [joint.frame_parent, joint.frame_child];
    if matches!(joint.kind, JointType::Revolute | JointType::Prismatic) {
        let alignment = Quat::from_rotation_arc(Vec3::X, joint.axis.normalize());
        for frame in &mut frames {
            frame.rotation *= alignment;
        }
    }
    let settings = scene.joint_axis_settings(id, axis).unwrap_or_default();
    if term != 0 {
        CompliantLoopRequest::motor(
            parent,
            child,
            frames,
            axis,
            if term == 1 {
                settings.motor.unwrap_or_default()
            } else {
                crate::loop_forces::brake_motor(settings.brake)
            },
        )
    } else {
        let limits = if scene.joint_locked_axes(id)?.contains(axis) {
            [0.0; 2]
        } else {
            settings
                .limits
                .unwrap_or([Real::NEG_INFINITY, Real::INFINITY])
        };
        CompliantLoopRequest::limit(
            parent,
            child,
            frames,
            axis,
            limits,
            scene.joint_softness(id)?.unwrap(),
        )
    }
}

impl LoopGpu {
    pub fn new(
        engine: &Engine,
        scene: &RigidScene,
        states: &[State; 2],
        status: &wgpu::Buffer,
        positions: [&wgpu::Buffer; 2],
        coordinates: [&wgpu::Buffer; 2],
    ) -> Result<Option<Self>> {
        let joints: Vec<_> = scene
            .loop_joints()
            .filter(|(id, joint)| {
                joint.kind != JointType::Distance
                    && scene.joint_softness(*id).unwrap().is_some()
                    && crate::distance_constraints::enabled(scene, *id, joint)
            })
            .map(|(id, _)| id)
            .collect();
        if joints.is_empty() {
            return Ok(None);
        }
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        let model = scene.model();
        let islands = RigidIslands::build(scene, [])?;
        let mut links = BTreeMap::new();
        for id in joints {
            let joint = scene.joint(id).unwrap();
            let group = islands
                .body_island(joint.child)
                .or_else(|| joint.parent.and_then(|body| islands.body_island(body)));
            let arts = group
                .map(|group| group.articulations.clone())
                .unwrap_or_default();
            let separate = arts.is_empty().then_some(id);
            links
                .entry((arts, separate))
                .or_insert_with(Vec::new)
                .push(id);
        }
        let mass_kernel = MassMatrixKernel::new(engine, model)?;
        let jacobian_kernel = JacobianKernel::new(engine, model)?;
        let starts = model.articulation_start.host()?;
        let offsets = model.joint_dof_offset.host()?;
        let children = model.joint_child.host()?;
        let centers = model.body_com.host()?;
        let mut gpu_islands = Vec::<[u32; 8]>::new();
        let mut gpu_arts = Vec::<[u32; 4]>::new();
        let mut projections = Vec::<[Endpoint; 2]>::new();
        let mut requests = Vec::new();
        let mut jacobian_requests = Vec::new();
        let mut owners = Vec::new();
        let mut island_articulations = Vec::new();
        let mut scratch_size = 0u64;
        let mut jacobian_size = 0u32;
        for ((arts, _), joints) in links {
            let art_start = gpu_arts.len() as u32;
            let row_start = requests.len() as u32;
            let mut local = 0;
            let mut endpoints = BTreeMap::new();
            for &art in &arts {
                let begin = starts[art] as usize;
                let end = starts[art + 1] as usize;
                let width = offsets[end] - offsets[begin];
                gpu_arts.push([local, width, mass_kernel.mass_offset(art), 0]);
                for &body in &children[begin..end] {
                    endpoints.insert(body, (art, local, width));
                }
                local += width;
            }
            let mut cache = BTreeMap::new();
            for id in joints {
                let joint = scene.joint(id).unwrap();
                let mut ends = [Endpoint::default(); 2];
                for (index, body) in [joint.parent, Some(joint.child)].into_iter().enumerate() {
                    let Some(body) = body else { continue };
                    let body = scene.body_index(body).unwrap().index() as i32;
                    if let Some(&endpoint) = cache.get(&body) {
                        ends[index] = endpoint;
                        continue;
                    }
                    if let Some(&(art, local, width)) = endpoints.get(&body)
                        && width != 0
                    {
                        let endpoint = Endpoint {
                            local,
                            width,
                            jacobian: jacobian_size,
                            padding: 0,
                        };
                        jacobian_requests.push(JacobianRequest {
                            articulation: art as u32,
                            target_body: body as u32,
                            output_offset: jacobian_size,
                            body_q_offset: 0,
                            point_local: centers[body as usize].as_vec3().extend(0.0).to_array(),
                        });
                        jacobian_size = jacobian_size
                            .checked_add(6 * width)
                            .ok_or_else(|| Error::Gpu("compliant loop Jacobian overflow".into()))?;
                        ends[index] = endpoint;
                        cache.insert(body, endpoint);
                    }
                }
                for axis in JointAxis::ALL {
                    for term in 0..ROWS_PER_AXIS {
                        requests.push(request(scene, id, axis, term)?);
                        projections.push(ends);
                        owners.push(id);
                    }
                }
            }
            let count = requests.len() as u32 - row_start;
            gpu_islands.push([
                art_start,
                arts.len() as u32,
                row_start,
                count,
                local,
                u32::try_from(scratch_size)
                    .map_err(|_| Error::Gpu("compliant loop scratch overflow".into()))?,
                0,
                0,
            ]);
            let n = local as u64;
            let r = count as u64;
            scratch_size += 2 * n * n + 2 * r * n + r * r + 4 * r;
            island_articulations.push(arts);
        }
        let limit = device.limits().max_storage_buffer_binding_size;
        for bytes in [
            scratch_size * 4,
            requests.len() as u64 * std::mem::size_of::<CompliantLoopRequest>() as u64,
            requests.len() as u64 * 96,
            jacobian_size as u64 * 4,
            model.body_count as u64 * 32,
        ] {
            if bytes > limit {
                return Err(Error::Gpu(
                    "compliant loop storage capacity exceeded".into(),
                ));
            }
        }
        if [gpu_islands.len(), requests.len(), model.body_count]
            .into_iter()
            .any(|n| n.div_ceil(64) > device.limits().max_compute_workgroups_per_dimension as usize)
        {
            return Err(Error::Gpu(
                "compliant loop dispatch capacity exceeded".into(),
            ));
        }
        let mass_buffer = mass_kernel.create_output_buffer(device);
        let jacobian_buffer = jacobian_kernel.create_output_buffer(device, jacobian_size as usize);
        let islands_buffer = data(device, queue, bytemuck::cast_slice(&gpu_islands), 32);
        let arts_buffer = data(device, queue, bytemuck::cast_slice(&gpu_arts), 16);
        let projections_buffer = data(device, queue, bytemuck::cast_slice(&projections), 32);
        let request_buffer = data(
            device,
            queue,
            bytemuck::cast_slice(&requests),
            std::mem::size_of::<CompliantLoopRequest>() as u64,
        );
        let rows = data(device, queue, &[], requests.len() as u64 * 96);
        let rounding = data(device, queue, &[], requests.len() as u64 * 4);
        let scratch = data(device, queue, &[], scratch_size * 4);
        let impulses = data(device, queue, &[], requests.len() as u64 * 4);
        let wrenches = data(device, queue, &[], requests.len() as u64 * 32);
        let parent_wrenches = data(device, queue, &[], requests.len() as u64 * 32);
        let persistent = data(device, queue, &[], model.body_count as u64 * 32);
        let active_upload = vec![1u32; gpu_islands.len()];
        let active_buffer = data(device, queue, bytemuck::cast_slice(&active_upload), 4);
        let mut incidents = vec![Vec::new(); model.body_count];
        for (index, request) in requests.iter().enumerate() {
            for (side, body) in request.metadata[..2].iter().enumerate() {
                if *body >= 0 {
                    incidents[*body as usize].push([index as u32, side as u32]);
                }
            }
        }
        let mut incidence_offsets = vec![0u32];
        let mut incidence_entries = Vec::new();
        for entries in incidents {
            incidence_entries.extend(entries);
            incidence_offsets.push(
                u32::try_from(incidence_entries.len())
                    .map_err(|_| Error::Gpu("compliant loop incidence overflow".into()))?,
            );
        }
        let offsets_buffer = data(device, queue, bytemuck::cast_slice(&incidence_offsets), 4);
        let incidence_buffer = data(device, queue, bytemuck::cast_slice(&incidence_entries), 8);
        let params = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("compliant loop parameters"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        let geometry_kernel = CompliantLoopKernel::new(engine)?;
        let orientation_kernel = QuaternionFkKernel::new_with_coordinate_residues(engine, model)?;
        let rotations = std::array::from_fn::<_, 2, _>(|_| {
            data(device, queue, &[], model.body_count as u64 * 32)
        });
        let orientation_rounding = data(device, queue, &[], model.body_count as u64 * 4);
        let orientations = std::array::from_fn::<_, 2, _>(|i| {
            orientation_kernel.prepare(
                device,
                QuaternionFkBindings {
                    coordinates: if model.joint_coord_total == 0 {
                        None
                    } else {
                        Some(states[i].joint_q.device_buffer_current()?)
                    },
                    coordinate_residues: Some(coordinates[i]),
                    poses: states[i].body_q.device_buffer_current()?,
                    rotations: &rotations[i],
                    rounding: &orientation_rounding,
                    status,
                },
            )
        });
        let [orientation0, orientation1] = orientations;
        let orientations = [orientation0?, orientation1?];
        let geometry = std::array::from_fn(|i| {
            geometry_kernel.prepare(
                device,
                CompliantLoopBindings {
                    requests: &request_buffer,
                    poses: states[i].body_q.device_buffer().unwrap(),
                    velocities: states[i].body_qd.device_buffer().unwrap(),
                    centers: model.body_com.device_buffer().unwrap(),
                    position_residues: positions[i],
                    rows: &rows,
                    status,
                    rounding: &rounding,
                    rotations: &rotations[i],
                },
                requests.len() as u32,
            )
        });
        let [geometry0, geometry1] = geometry;
        let geometry = [geometry0?, geometry1?];
        let entries: Vec<_> = (0..16)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 13 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: ![6, 7, 8, 9, 12, 15].contains(&binding),
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compliant loop solve"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compliant loop solve"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("coupled compliant loop solve"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/loop_forces.wgsl").into()),
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let solve = pipeline("solve_loops");
        let publish = pipeline("publish_rows");
        let apply = pipeline("apply_loads");
        let groups = std::array::from_fn(|i| {
            let buffers = [
                &islands_buffer,
                &arts_buffer,
                &projections_buffer,
                &mass_buffer,
                &jacobian_buffer,
                &rows,
                &scratch,
                &impulses,
                &wrenches,
                states[i].body_f.device_buffer().unwrap(),
                &offsets_buffer,
                &incidence_buffer,
                status,
                &params,
                &active_buffer,
                &parent_wrenches,
            ];
            let entries: Vec<_> = buffers
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("compliant loop solve"),
                layout: &layout,
                entries: &entries,
            })
        });
        let mass = [
            mass_kernel.prepare_on_device(device, queue, model, &states[0], &mass_buffer)?,
            mass_kernel.prepare_on_device(device, queue, model, &states[1], &mass_buffer)?,
        ];
        let prepare_jacobians = |i: usize| {
            jacobian_kernel.prepare_with_body_q_on_device(
                device,
                queue,
                model,
                states[i].body_q.device_buffer().unwrap(),
                &jacobian_requests,
                &jacobian_buffer,
            )
        };
        let jacobians = [prepare_jacobians(0)?, prepare_jacobians(1)?];
        Ok(Some(Self {
            mass,
            jacobians,
            geometry,
            orientations,
            solve,
            publish,
            apply,
            groups,
            params,
            request_buffer,
            requests,
            owners,
            wrenches,
            parent_wrenches,
            impulses,
            persistent,
            active_buffer,
            active_upload,
            island_articulations,
            bodies: model.body_count as u32,
        }))
    }

    pub fn bind_groups_created(&self) -> u64 {
        6 + 6 + 4 * self.jacobians.iter().filter(|j| j.is_some()).count() as u64
    }

    pub(crate) fn activation_layout(&self) -> (&wgpu::Buffer, &[Vec<usize>]) {
        (&self.active_buffer, &self.island_articulations)
    }

    pub fn prepare_frame(
        &mut self,
        engine: &Engine,
        scene: &RigidScene,
        states: &[State; 2],
        selected: &[bool],
    ) -> Result<()> {
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        let mut changed = false;
        for (index, (&id, previous)) in self.owners.iter().zip(&mut self.requests).enumerate() {
            let next = request(
                scene,
                id,
                JointAxis::ALL[(index / ROWS_PER_AXIS) % JointAxis::ALL.len()],
                index % ROWS_PER_AXIS,
            )?;
            if bytemuck::bytes_of(previous) != bytemuck::bytes_of(&next) {
                *previous = next;
                changed = true;
            }
        }
        if changed {
            molla_core::gpu_write_buffer!(
                queue,
                &self.request_buffer,
                0,
                bytemuck::cast_slice(&self.requests),
            );
        }
        for (index, arts) in self.island_articulations.iter().enumerate() {
            let count = arts.iter().filter(|&&art| selected[art]).count();
            if count != 0 && count != arts.len() {
                return Err(Error::Solver(
                    "GPU activation splits a compliant loop dependency".into(),
                ));
            }
            self.active_upload[index] = u32::from(arts.is_empty() || count != 0);
        }
        molla_core::gpu_write_buffer!(
            queue,
            &self.active_buffer,
            0,
            bytemuck::cast_slice(&self.active_upload),
        );
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(
            states[0].body_f.device_buffer_current()?,
            0,
            &self.persistent,
            0,
            u64::from(self.bodies) * 32,
        );
        queue.submit([encoder.finish()]);
        Ok(())
    }

    pub fn dispatch(&self, engine: &Engine, input: usize, dt: Real) {
        let queue = engine.queue().unwrap();
        molla_core::gpu_write_buffer!(
            queue,
            &self.params,
            0,
            bytemuck::cast_slice(&[
                (dt as f32).to_bits(),
                self.island_articulations.len() as u32,
                self.requests.len() as u32,
                self.bodies,
            ]),
        );
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        encoder.clear_buffer(&self.impulses, 0, None);
        encoder.clear_buffer(&self.wrenches, 0, None);
        encoder.clear_buffer(&self.parent_wrenches, 0, None);
        self.orientations[input].encode(&mut encoder);
        self.geometry[input].encode(&mut encoder);
        self.mass[input].encode(&mut encoder);
        if let Some(jacobians) = &self.jacobians[input] {
            jacobians.encode(&mut encoder);
        }
        for (pipeline, count) in [
            (&self.solve, self.island_articulations.len() as u32),
            (&self.publish, self.requests.len() as u32),
            (&self.apply, self.bodies),
        ] {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.groups[input], &[]);
            pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        }
        queue.submit([encoder.finish()]);
    }

    pub fn restore(&self, engine: &Engine, states: [&State; 2]) -> Result<()> {
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        for state in states {
            encoder.copy_buffer_to_buffer(
                &self.persistent,
                0,
                state.body_f.device_buffer_current()?,
                0,
                u64::from(self.bodies) * 32,
            );
        }
        engine.queue().unwrap().submit([encoder.finish()]);
        Ok(())
    }
}
