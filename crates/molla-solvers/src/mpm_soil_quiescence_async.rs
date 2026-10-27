//! Bounded exact-rest readbacks with request-time provenance.

use super::*;

pub(super) struct PendingRead {
    receive: std::sync::Mutex<
        std::sync::mpsc::Receiver<std::result::Result<(), wgpu::BufferAsyncError>>,
    >,
    snapshot: Option<MpmSoilQuiescenceSnapshot>,
    bytes: u64,
}

impl MpmSimulation {
    pub(in crate::mpm) fn try_request_soil_quiescence(
        &mut self,
        engine: &Engine,
        max_particles: u32,
    ) -> Result<bool> {
        engine.check_device()?;
        molla_kernels::validate_soil_device(engine.device().unwrap(), &self.device)?;
        let guard = self
            .motion_guard
            .as_ref()
            .ok_or_else(|| Error::Solver("soil quiescence is not prepared".into()))?;
        let observer = guard
            .quiescence
            .as_ref()
            .ok_or_else(|| Error::Solver("soil quiescence is not prepared".into()))?;
        let attempt = observer
            .attempt
            .ok_or_else(|| Error::Solver("soil quiescence observation counter exhausted".into()))?;
        if guard.bounds.is_none()
            || observer.needs_reset
            || self.active_grid_needs_reset
            || self.particle_count > max_particles
        {
            return Err(Error::Solver(
                "soil quiescence is stale, disabled or exceeds read bound".into(),
            ));
        }
        if let Some(error) = &observer.read_failure {
            return Err(Error::Gpu(error.clone()));
        }
        if observer.pending.is_some() {
            return Ok(false);
        }
        let snapshot = MpmSoilQuiescenceSnapshot {
            identity: self.soil_identity_snapshot().cloned(),
            epoch: self.step_epoch.clone(),
            attempt,
            particles: Vec::new(),
        };
        let bytes = u64::from(self.particle_count) * 8;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(
            &self.active_tiles.list,
            self.active_tiles.metadata_offset,
            &observer.staging,
            0,
            16,
        );
        if bytes > 0 {
            encoder.copy_buffer_to_buffer(&observer.records, 0, &observer.staging, 16, bytes);
        }
        engine.queue().unwrap().submit([encoder.finish()]);
        molla_core::gpu_metrics::record_readback(16 + bytes);
        let (send, receive) = std::sync::mpsc::sync_channel(1);
        observer
            .staging
            .slice(..16 + bytes)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = send.send(result);
            });
        self.motion_guard
            .as_mut()
            .unwrap()
            .quiescence
            .as_mut()
            .unwrap()
            .pending = Some(PendingRead {
            receive: std::sync::Mutex::new(receive),
            snapshot: Some(snapshot),
            bytes: 16 + bytes,
        });
        Ok(true)
    }

    pub(in crate::mpm) fn poll_soil_quiescence(
        &mut self,
        engine: &Engine,
    ) -> Result<Option<MpmSoilQuiescenceSnapshot>> {
        self.poll_soil_quiescence_inner(engine, false)
    }

    pub(in crate::mpm) fn cancel_soil_quiescence(&mut self) -> bool {
        self.motion_guard
            .as_mut()
            .and_then(|g| g.quiescence.as_mut())
            .and_then(|o| o.pending.as_mut())
            .is_some_and(|pending| pending.snapshot.take().is_some())
    }

    pub(super) fn poll_soil_quiescence_inner(
        &mut self,
        engine: &Engine,
        wait: bool,
    ) -> Result<Option<MpmSoilQuiescenceSnapshot>> {
        engine.check_device()?;
        molla_kernels::validate_soil_device(engine.device().unwrap(), &self.device)?;
        let observer = self
            .motion_guard
            .as_mut()
            .and_then(|g| g.quiescence.as_mut())
            .ok_or_else(|| Error::Solver("soil quiescence is not prepared".into()))?;
        if let Some(error) = &observer.read_failure {
            return Err(Error::Gpu(error.clone()));
        }
        let Some(pending) = observer.pending.as_mut() else {
            return Ok(None);
        };
        let mapped = match self.device.poll(if wait {
            wgpu::PollType::wait_indefinitely()
        } else {
            wgpu::PollType::Poll
        }) {
            Err(error) => Err(Error::Gpu(error.to_string())),
            Ok(_) => match pending.receive.get_mut().unwrap().try_recv() {
                Ok(result) => result.map_err(|e| Error::Gpu(e.to_string())),
                Err(std::sync::mpsc::TryRecvError::Empty) if !wait => return Ok(None),
                Err(error) => Err(Error::Gpu(error.to_string())),
            },
        };
        let pending = observer.pending.take().unwrap();
        if let Err(error) = mapped {
            observer.staging.unmap();
            observer.read_failure = Some(error.to_string());
            return Err(error);
        }
        let Some(mut snapshot) = pending.snapshot else {
            observer.staging.unmap();
            return Ok(None);
        };
        let mapped = observer.staging.slice(..pending.bytes).get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&mapped);
        let faulted = words[2] != 0;
        if !faulted {
            snapshot.particles = words[4..]
                .chunks_exact(2)
                .map(|r| MpmSoilQuiescence {
                    consecutive_steps: r[0],
                    dt: f32::from_bits(r[1]),
                })
                .collect();
        }
        drop(mapped);
        observer.staging.unmap();
        if faulted {
            return Err(Error::Solver(
                "faulted soil has no valid quiescence snapshot".into(),
            ));
        }
        Ok(Some(snapshot))
    }
}

#[cfg(all(test, feature = "resource-counters"))]
#[path = "mpm_soil_quiescence_async_tests.rs"]
mod tests;
