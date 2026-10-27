//! Reusable shared-command scene refits and closest-hit traversal.

use crate::{
    SceneAcceleration, SceneRebuildPolicy, SceneStamp, SensorScene, dispatch::BindingsCache,
};
use molla_compute::Engine;
use molla_core::Result;
use molla_kernels::{SceneRebuildKernel, SceneTraceBindings, SceneTraceKernel};
use molla_sim::State;

pub(crate) struct SceneTrace {
    pub stamp: SceneStamp,
    acceleration: SceneAcceleration,
    kernel: SceneTraceKernel,
    bindings: BindingsCache<SceneTraceBindings>,
    rebuild: Option<SceneRebuildKernel>,
    interval: u32,
}

impl SceneTrace {
    pub fn new(
        engine: &Engine,
        scene: &SensorScene,
        state: &State,
        policy: SceneRebuildPolicy,
    ) -> Result<Self> {
        let seeds = state
            .body_q
            .host()
            .ok()
            .filter(|poses| poses.len() == scene.model.body_count);
        let acceleration = SceneAcceleration::new(engine, scene, seeds, Default::default())?;
        acceleration.preflight(engine, scene, state)?;
        let storage = policy.check(
            acceleration.storage().leaves,
            &engine.device().expect("checked engine").limits(),
        )?;
        let rebuild = if storage.device_bytes != 0 {
            Some(acceleration.query_rebuild_kernel(policy.quality_ratio_percent)?)
        } else {
            None
        };
        debug_assert_eq!(
            rebuild.as_ref().map_or(0, SceneRebuildKernel::device_bytes),
            storage.device_bytes
        );
        let kernel = SceneTraceKernel::new(engine.device().expect("checked engine"))?;
        Ok(Self {
            stamp: scene.stamp(),
            acceleration,
            kernel,
            bindings: BindingsCache::with_capacity(128),
            rebuild,
            interval: policy.interval,
        })
    }

    pub fn check_policy(&self, policy: SceneRebuildPolicy, limits: &wgpu::Limits) -> Result<()> {
        policy.check(self.acceleration.storage().leaves, limits)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &mut self,
        engine: &Engine,
        encoder: &mut wgpu::CommandEncoder,
        scene: &SensorScene,
        state: &State,
        outputs: &[wgpu::Buffer],
        count: u32,
        max_distance: f32,
        refresh_bounds: bool,
        until_rebuild: &mut u32,
    ) -> Result<(u64, u64, u64, bool, bool)> {
        self.acceleration.preflight(engine, scene, state)?;
        let before = self.acceleration.stats();
        if refresh_bounds {
            self.acceleration.encode(encoder, state)?;
        }
        let mut rebuilt = false;
        let mut quality_checked = false;
        let mut rebuild_dispatches = 0;
        if let Some(rebuild) = &self.rebuild {
            if *until_rebuild == 0 {
                rebuild_dispatches += rebuild.encode(encoder);
                self.acceleration.encode(encoder, state)?;
                rebuild_dispatches += rebuild.finish_quality(encoder);
                *until_rebuild = self.interval - 1;
                quality_checked = rebuild.quality_state().is_some();
                rebuilt = !quality_checked;
            } else {
                *until_rebuild -= 1;
            }
        }
        let after = self.acceleration.stats();
        let [topology, lower, upper, poses] = self.acceleration.trace_buffers(state)?;
        let geometry = &scene.geometry;
        let buffers = [
            &geometry.shapes,
            &geometry.meshes,
            topology,
            lower,
            upper,
            poses,
            &outputs[0],
            &outputs[1],
            &outputs[2],
            &outputs[3],
            &outputs[4],
            &outputs[5],
        ];
        let parameters = [
            count,
            max_distance.to_bits(),
            self.acceleration
                .storage()
                .nodes
                .checked_sub(1)
                .unwrap_or(u32::MAX),
            geometry.plane_offset,
            geometry.plane_count,
            0,
            0,
            0,
        ];
        let (bindings, prepared) = self.bindings.get_or_prepare(
            |b| b.matches(buffers, parameters),
            || {
                self.kernel.prepare(
                    engine.device().expect("checked engine"),
                    buffers,
                    parameters,
                )
            },
        );
        self.kernel.encode(encoder, bindings);
        Ok((
            1 + after.kernel_dispatches - before.kernel_dispatches + rebuild_dispatches,
            u64::from(prepared),
            after.binding_preparations - before.binding_preparations,
            rebuilt,
            quality_checked,
        ))
    }
}
