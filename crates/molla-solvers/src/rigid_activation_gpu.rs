use super::super::sleep::SleepingGroups;
use crate::FeatherstoneSolver;
use crate::rigid_contact_activation_gpu::{ContactActivationGpu, NO_GROUP, SelectionMapping};
use crate::rigid_islands::RigidIslands;
use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_sim::runtime::RigidScene;
use molla_storage::BufferElement;
use std::collections::BTreeSet;

pub(super) struct FrameSelection<'a> {
    pub bodies: &'a [bool],
    pub articulations: &'a [bool],
    pub sleeping: &'a SleepingGroups,
}

pub(super) struct RuntimeActivation {
    device: ContactActivationGpu,
    mappings: Vec<SelectionMapping>,
    body_groups: Vec<u32>,
    groups: usize,
    accelerations: wgpu::Buffer,
}

impl RuntimeActivation {
    pub(super) fn new(
        engine: &Engine,
        scene: &RigidScene,
        solver: &FeatherstoneSolver,
        compiled: &super::Compiled,
    ) -> Result<Option<Self>> {
        let Some(contacts) = &compiled.contacts else {
            return Ok(None);
        };
        let graph = RigidIslands::build(scene, [])?;
        let groups = graph.groups().len();
        let mut body_groups = vec![NO_GROUP; scene.model().body_count];
        let mut art_groups = vec![NO_GROUP; scene.model().articulation_count];
        for (index, group) in graph.groups().iter().enumerate() {
            for &body in &group.bodies {
                body_groups[scene.body_index(body).unwrap().index()] = index as u32;
            }
            for &art in &group.articulations {
                art_groups[art] = index as u32;
            }
        }
        let accelerations = molla_core::gpu_create_buffer!(
            engine.device().unwrap(),
            &wgpu::BufferDescriptor {
                label: Some("contact activation boundary accelerations"),
                size: (body_groups.len() as u64 * 32).max(32),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        let velocities: Vec<_> = compiled
            .states
            .iter()
            .map(|s| s.body_qd.device_buffer().unwrap_or(&accelerations))
            .collect();
        let buffers = contacts.activation_buffers(
            solver.gpu_status_buffer().unwrap(),
            &velocities,
            &accelerations,
        );
        let device = ContactActivationGpu::new(
            engine,
            &body_groups,
            groups as u32,
            (buffers.rows.size() / 128) as u32,
            buffers,
        )?;
        let mut mappings = vec![device.selection(
            engine,
            &art_groups,
            solver.gpu_active_articulations_buffer()?,
        )?];
        let island_mapping = |(buffer, islands): (&wgpu::Buffer, &[Vec<usize>])| {
            let ids = islands
                .iter()
                .map(|arts| {
                    let group = arts
                        .iter()
                        .map(|&a| art_groups[a])
                        .find(|&g| g != NO_GROUP)
                        .unwrap_or(NO_GROUP);
                    if arts
                        .iter()
                        .any(|&a| art_groups[a] != NO_GROUP && art_groups[a] != group)
                    {
                        return Err(Error::Gpu("inconsistent contact activation island".into()));
                    }
                    Ok(group)
                })
                .collect::<Result<Vec<_>>>()?;
            device.selection(engine, &ids, buffer)
        };
        if let Some(distance) = &compiled.distance {
            mappings.push(island_mapping(distance.activation_layout())?);
        }
        if let Some(loops) = &compiled.loops {
            mappings.push(island_mapping(loops.activation_layout())?);
        }
        if let Some(reactions) = &compiled.reactions {
            let (arts, selected, owners) = reactions.activation_layout();
            mappings.push(device.selection(engine, &art_groups, arts)?);
            let joint_groups: Vec<_> = owners
                .iter()
                .map(|&owner| {
                    let joint = scene.joint(owner).unwrap();
                    [Some(joint.child), joint.parent]
                        .into_iter()
                        .flatten()
                        .map(|b| body_groups[scene.body_index(b).unwrap().index()])
                        .find(|&g| g != NO_GROUP)
                        .unwrap_or(NO_GROUP)
                })
                .collect();
            mappings.push(device.selection(engine, &joint_groups, selected)?);
        }
        Ok(Some(Self {
            device,
            mappings,
            body_groups,
            groups,
            accelerations,
        }))
    }

    pub(super) fn prepare(
        &self,
        engine: &Engine,
        scene: &RigidScene,
        selection: &FrameSelection<'_>,
        contacts: &super::contacts::RuntimeContacts,
    ) -> Result<()> {
        let mut awake = vec![0u32; self.groups];
        for (&group, &selected) in self.body_groups.iter().zip(selection.bodies) {
            if group != NO_GROUP && selected {
                awake[group as usize] = 1;
            }
        }
        let body_group = |body| {
            scene
                .body_index(body)
                .map(|b| self.body_groups[b.index()])
                .filter(|&g| g != NO_GROUP)
        };
        let mut dependencies = BTreeSet::new();
        let mut watches = BTreeSet::new();
        for (bodies, pairs) in selection.sleeping.gpu_groups() {
            let Some(first) = bodies.iter().find_map(|&b| body_group(b)) else {
                continue;
            };
            for group in bodies.iter().filter_map(|&b| body_group(b)) {
                if first != group {
                    dependencies.insert([first.min(group), first.max(group)]);
                }
            }
            for pair in pairs {
                if let [Some(a), Some(b)] = pair.map(|c| contacts.collider_identity(scene, c)) {
                    watches.insert([first, a, b, 0]);
                } else {
                    awake[first as usize] = 1;
                }
            }
        }
        self.device.prepare(
            engine.queue().unwrap(),
            &awake,
            &dependencies.into_iter().collect::<Vec<_>>(),
            &watches.into_iter().collect::<Vec<_>>(),
        )?;
        let accelerations = scene.model().body_kinematic_acceleration.host()?;
        if !accelerations.is_empty() {
            let data: Vec<_> = accelerations.iter().map(BufferElement::to_gpu).collect();
            molla_core::gpu_write_buffer!(
                engine.queue().unwrap(),
                &self.accelerations,
                0,
                bytemuck::cast_slice(&data),
            );
        }
        Ok(())
    }

    pub(super) fn encode(&self, encoder: &mut wgpu::CommandEncoder, side: usize) -> Result<()> {
        self.device.encode(encoder, side)?;
        for mapping in &self.mappings {
            self.device.encode_selection(encoder, mapping);
        }
        Ok(())
    }

    pub(super) fn body_mask(&self) -> &wgpu::Buffer {
        self.device.body_mask()
    }
}
