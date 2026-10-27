//! Stable-identity device contact episodes and onset penalty suppression.

use molla_core::{Error, Result, ShapeId};
use molla_math::Transform;
use molla_sim::runtime::{ColliderHandle, RigidScene, SceneStamp};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU32, Ordering};

pub const IMPACT_EPISODE_FAULT: u32 = 1 << 29;
static NEXT_IDENTITY: AtomicU32 = AtomicU32::new(1);

#[derive(Clone, PartialEq)]
struct Identity {
    collider: SceneStamp,
    pose: Transform,
    parent: Option<(SceneStamp, SceneStamp)>,
}

/// Model-order collider identities, retaining unchanged handles across dense compaction.
#[derive(Clone)]
pub struct ImpactEpisodeLayout {
    entries: BTreeMap<ColliderHandle, (Identity, u32)>,
    tokens: Vec<u32>,
    published: crate::rigid_impact::PublishedImpactState,
}

impl ImpactEpisodeLayout {
    pub(crate) fn record_published(&mut self, scene: &RigidScene) {
        self.published.record(scene);
    }

    pub(crate) fn preserve_after_detachment(&mut self, scene: &RigidScene) {
        for (handle, (identity, _)) in &mut self.entries {
            if let Some(collider) = scene.collider(*handle)
                && scene.collider_stamp(*handle).as_ref() == Some(&identity.collider)
                && collider.pose == identity.pose
                && let Some(body) = collider.parent
                && let Some((structure, state)) = &mut identity.parent
                && scene.body_state_stamp(body).as_ref() == Some(state)
            {
                *structure = scene.body_contact_episode_stamp(body).unwrap();
            }
        }
    }

    pub fn new(scene: &RigidScene, previous: Option<&Self>) -> Result<Self> {
        let mut entries = BTreeMap::new();
        for (handle, collider) in scene.colliders() {
            let identity = Identity {
                collider: scene.collider_stamp(handle).unwrap(),
                pose: collider.pose,
                parent: collider.parent.map(|body| {
                    (
                        scene.body_contact_episode_stamp(body).unwrap(),
                        scene.body_state_stamp(body).unwrap(),
                    )
                }),
            };
            let retained = previous
                .and_then(|old| old.entries.get(&handle))
                .filter(|(old, _)| {
                    let mut normalized = old.clone();
                    if let Some(body) = collider.parent
                        && old.parent.as_ref().map(|p| &p.0)
                            == identity.parent.as_ref().map(|p| &p.0)
                        && previous
                            .unwrap()
                            .published
                            .matches(scene, scene.state(), body)
                    {
                        normalized.parent = identity.parent.clone();
                    }
                    normalized == identity
                })
                .map(|(_, token)| *token);
            let token = match retained {
                Some(token) => token,
                None => NEXT_IDENTITY
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
                    .map_err(|_| Error::Gpu("impact collider identity space exhausted".into()))?,
            };
            entries.insert(handle, (identity, token));
        }
        let tokens = (0..scene.model().shape_count)
            .map(|shape| {
                let handle = scene
                    .collider_handle(ShapeId::from_index(shape))
                    .ok_or_else(|| Error::Gpu("missing impact collider identity".into()))?;
                Ok(entries[&handle].1)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            entries,
            tokens,
            published: previous.map(|p| p.published.clone()).unwrap_or_default(),
        })
    }

    /// Zero is reserved; compiled pieces of one collider share the same identity.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ImpactEpisodeConfig {
    pub row_capacity: u32,
    pub shape_capacity: u32,
    /// Power-of-two open-addressed pair capacity; exhaustion is a sticky fault.
    pub table_capacity: u32,
    pub approach_speed: f32,
    pub normal_similarity: f32,
}

/// Groups active NormalContactRows by stable collider pair, not contact array position.
/// Each encode requires freshly prepared normal rows. New approaching episodes zero
/// their source penalty forces, preserve geometric depth, and mark every pair row.
/// Impact output retains all active contacts; old episodes use zero restitution.
/// `ResidentImpactResponse::with_episode_onsets` selects connected groups and predicted rows.
pub struct ResidentImpactEpisodes {
    pipelines: [wgpu::ComputePipeline; 6],
    layout: wgpu::BindGroupLayout,
    group: wgpu::BindGroup,
    rows: wgpu::Buffer,
    identities: wgpu::Buffer,
    status: wgpu::Buffer,
    history: wgpu::Buffer,
    checkpoint: wgpu::Buffer,
    next: wgpu::Buffer,
    scratch: wgpu::Buffer,
    onsets: wgpu::Buffer,
    impacts: wgpu::Buffer,
    config: ImpactEpisodeConfig,
}

impl ResidentImpactEpisodes {
    pub fn new(
        device: &wgpu::Device,
        rows: &wgpu::Buffer,
        identities: &wgpu::Buffer,
        status: &wgpu::Buffer,
        config: ImpactEpisodeConfig,
    ) -> Result<Self> {
        let invalid = || Error::Gpu("invalid impact episode capacity or buffers".into());
        let limits = device.limits();
        let largest = config.row_capacity.max(config.table_capacity);
        if config.row_capacity == 0
            || config.shape_capacity == 0
            || !config.table_capacity.is_power_of_two()
            || config.row_capacity == u32::MAX
            || config.table_capacity > u32::MAX / 2
            || !config.approach_speed.is_finite()
            || config.approach_speed < 0.
            || !config.normal_similarity.is_finite()
            || !(-1. ..=1.).contains(&config.normal_similarity)
            || largest.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || limits.max_storage_buffers_per_shader_stage < 8
        {
            return Err(invalid());
        }
        for bytes in [
            u64::from(config.table_capacity) * 32,
            u64::from(config.row_capacity) * 128,
            u64::from(config.shape_capacity) * 4,
        ] {
            if bytes > limits.max_storage_buffer_binding_size || bytes > limits.max_buffer_size {
                return Err(invalid());
            }
        }
        for (buffer, bytes) in [
            (rows, u64::from(config.row_capacity) * 128),
            (identities, u64::from(config.shape_capacity) * 4),
            (status, 4),
        ] {
            if buffer.size() < bytes
                || buffer.size() > limits.max_storage_buffer_binding_size
                || !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
            {
                return Err(invalid());
            }
        }
        let history = storage(
            device,
            "impact episode history",
            u64::from(config.table_capacity) * 32,
        );
        let checkpoint = storage(device, "impact episode checkpoint", history.size());
        let next = storage(device, "impact episode next", history.size());
        let scratch = storage(
            device,
            "impact episode claims",
            u64::from(config.table_capacity) * 8,
        );
        let onsets = storage(
            device,
            "impact episode onsets",
            u64::from(config.row_capacity) * 4,
        );
        let impacts = storage(
            device,
            "impact episode rows",
            u64::from(config.row_capacity) * 128,
        );
        let uniform = params(device, config, config.table_capacity);
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("impact episode layout"),
            entries: &(0..9)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 8 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage {
                                read_only: binding == 1,
                            }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                })
                .collect::<Vec<_>>(),
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("impact episodes"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/impact_episodes.wgsl").into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("impact episode pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipelines = [
            "clear",
            "group_rows",
            "classify",
            "publish_rows",
            "commit",
            "inherit",
        ]
        .map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let group = bindings(
            device,
            &layout,
            &[
                rows, identities, &history, &next, &scratch, &onsets, &impacts, status, &uniform,
            ],
        );
        Ok(Self {
            pipelines,
            layout,
            group,
            rows: rows.clone(),
            identities: identities.clone(),
            status: status.clone(),
            history,
            checkpoint,
            next,
            scratch,
            onsets,
            impacts,
            config,
        })
    }

    /// Record grouping, classification, penalty suppression and status-guarded history commit.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        for phase in 0..5 {
            self.dispatch(
                encoder,
                &self.group,
                phase,
                self.config.row_capacity.max(self.config.table_capacity),
            );
        }
    }

    /// Save episode history at the frame transaction boundary, before any substep.
    pub fn encode_checkpoint(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(&self.history, 0, &self.checkpoint, 0, self.history.size());
    }

    /// Restore frame history and retire transient outputs; shared status is not reset.
    pub fn encode_rollback(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(&self.checkpoint, 0, &self.history, 0, self.history.size());
        for buffer in [&self.next, &self.scratch, &self.onsets, &self.impacts] {
            encoder.clear_buffer(buffer, 0, None);
        }
    }

    /// Rehash prior history after capacity/binding rebuild. Allocates only rebuild bindings.
    /// Shape identity tokens must retain their meanings across the two components.
    pub fn encode_inherit(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        previous: &Self,
    ) -> Result<()> {
        if std::ptr::eq(self, previous) {
            return Err(Error::Gpu("cannot inherit impact episodes in place".into()));
        }
        let uniform = params(device, self.config, previous.config.table_capacity);
        let group = bindings(
            device,
            &self.layout,
            &[
                &self.rows,
                &self.identities,
                &previous.history,
                &self.next,
                &self.scratch,
                &self.onsets,
                &self.impacts,
                &self.status,
                &uniform,
            ],
        );
        self.dispatch(
            encoder,
            &group,
            0,
            self.config.row_capacity.max(self.config.table_capacity),
        );
        self.dispatch(encoder, &group, 5, previous.config.table_capacity);
        self.dispatch(encoder, &self.group, 4, self.config.table_capacity);
        Ok(())
    }

    fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        group: &wgpu::BindGroup,
        phase: usize,
        count: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("impact episodes"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines[phase]);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }

    pub fn onset_buffer(&self) -> &wgpu::Buffer {
        &self.onsets
    }
    pub fn impact_row_buffer(&self) -> &wgpu::Buffer {
        &self.impacts
    }
    pub fn resource_counts(&self) -> [usize; 3] {
        [7, 1, 6]
    }
}

fn params(
    device: &wgpu::Device,
    config: ImpactEpisodeConfig,
    source_capacity: u32,
) -> wgpu::Buffer {
    let b = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some("impact episode parameters"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: true,
        }
    );
    molla_core::gpu_mapped_upload!(b.slice(..)).copy_from_slice(bytemuck::cast_slice(&[
        config.row_capacity,
        config.shape_capacity,
        config.table_capacity,
        source_capacity,
        config.approach_speed.to_bits(),
        config.normal_similarity.to_bits(),
        0,
        0,
    ]));
    b.unmap();
    b
}

fn storage(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
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
}

fn bindings(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffers: &[&wgpu::Buffer],
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("impact episode bindings"),
        layout,
        entries: &buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
    })
}

#[cfg(test)]
#[path = "impact_episodes_gpu_tests.rs"]
mod tests;
