use super::*;
use std::sync::{Arc, OnceLock};

#[derive(Clone, Debug)]
pub struct SoilSurfaceObservation {
    owner: Arc<()>,
    generation: u64,
    revision: u64,
    surface: SoilSurfaceSnapshot,
}

impl SoilSurfaceObservation {
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn revision(&self) -> u64 {
        self.revision
    }
    pub fn surface(&self) -> &SoilSurfaceSnapshot {
        &self.surface
    }
    pub fn into_surface(self) -> SoilSurfaceSnapshot {
        self.surface
    }
}

struct Pending {
    generation: u64,
    revision: u64,
    completion: Arc<OnceLock<std::result::Result<(), wgpu::BufferAsyncError>>>,
}

#[derive(Default)]
pub(super) struct State {
    owner: Arc<()>,
    generation: u64,
    last_requested: Option<u64>,
    pending: Option<Pending>,
    failure: Option<String>,
}

impl PreparedSoilSurface {
    fn check_observation(&self) -> Result<()> {
        match &self.observations.failure {
            Some(error) => Err(Error::Gpu(error.clone())),
            None => Ok(()),
        }
    }

    pub(super) fn check_readback_idle(&self) -> Result<()> {
        self.check_observation()?;
        if self.snapshot_pending() {
            return Err(invalid("asynchronous snapshot pending"));
        }
        Ok(())
    }

    pub fn snapshot_pending(&self) -> bool {
        self.observations.pending.is_some()
    }

    pub fn snapshot_generation(&self) -> u64 {
        self.observations.generation
    }

    /// Invalidate prior observations without waiting or reusing an in-flight stage.
    pub fn reset_snapshots(&mut self) -> Result<()> {
        self.check_observation()?;
        self.observations.generation = self
            .observations
            .generation
            .checked_add(1)
            .ok_or_else(|| invalid("snapshot generation exhausted"))?;
        self.observations.last_requested = None;
        Ok(())
    }

    /// Checks owner/generation and observed map failures, not external device health.
    pub fn is_current_snapshot(&self, observation: &SoilSurfaceObservation) -> bool {
        self.check_observation().is_ok()
            && Arc::ptr_eq(&self.observations.owner, &observation.owner)
            && self.observations.generation == observation.generation
    }

    /// Export and copy after preceding queue work; false means the single retained stage is busy.
    /// Revisions must increase within a generation. Reset explicitly after source reseeding.
    pub fn try_request_snapshot(&mut self, revision: u64) -> Result<bool> {
        self.check_observation()?;
        if self
            .observations
            .last_requested
            .is_some_and(|last| revision <= last)
        {
            return Err(invalid("snapshot revisions must increase"));
        }
        if self.snapshot_pending() {
            return Ok(false);
        }
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.encode(&mut encoder);
        let cells = u64::from(self.config.cells()?);
        for (buffer, (offset, size)) in self.outputs.iter().zip([
            (0, cells * 4),
            (cells * 4, cells * 4),
            (cells * 8, cells * 16),
        ]) {
            encoder.copy_buffer_to_buffer(buffer, 0, &self.stage, offset, size);
        }
        self.queue.submit([encoder.finish()]);
        let completion = Arc::new(OnceLock::new());
        let signal = completion.clone();
        self.stage
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = signal.set(result);
            });
        self.observations.pending = Some(Pending {
            generation: self.observations.generation,
            revision,
            completion,
        });
        self.observations.last_requested = Some(revision);
        Ok(true)
    }

    fn fail_observation(&mut self, message: String) -> Error {
        self.stage.unmap();
        self.observations.pending = None;
        self.observations.failure = Some(message.clone());
        Error::Gpu(message)
    }

    /// Poll without waiting; obsolete generations are discarded without a CPU payload copy.
    pub fn poll_snapshot(&mut self) -> Result<Option<SoilSurfaceObservation>> {
        self.check_observation()?;
        if !self.snapshot_pending() {
            return Ok(None);
        }
        if let Err(error) = self.device.poll(wgpu::PollType::Poll) {
            return Err(self.fail_observation(format!("surface snapshot poll: {error}")));
        }
        let pending = self.observations.pending.as_ref().unwrap();
        match pending.completion.get() {
            None => return Ok(None),
            Some(Err(error)) => {
                return Err(self.fail_observation(format!("surface snapshot map: {error}")));
            }
            Some(Ok(())) => {}
        }
        let pending = self.observations.pending.take().unwrap();
        if pending.generation != self.observations.generation {
            self.stage.unmap();
            return Ok(None);
        }
        let slice = self.stage.slice(..);
        let data = molla_core::gpu_readback!(slice);
        let result = decode(self.config, self.count, bytemuck::cast_slice(&data));
        drop(data);
        self.stage.unmap();
        result.map(|surface| {
            Some(SoilSurfaceObservation {
                owner: self.observations.owner.clone(),
                generation: pending.generation,
                revision: pending.revision,
                surface,
            })
        })
    }
}
