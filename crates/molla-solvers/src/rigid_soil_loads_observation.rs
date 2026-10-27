use super::*;
use std::sync::{Arc, OnceLock};

/// Frame-integrated impulses from one captured commit and observation generation.
#[derive(Clone, Debug)]
pub struct MpmSoilFrameLoadObservation {
    owner: Arc<()>,
    loads: MpmSoilFrameLoads,
}

impl MpmSoilFrameLoadObservation {
    pub fn loads(&self) -> &MpmSoilFrameLoads {
        &self.loads
    }

    pub fn into_loads(self) -> MpmSoilFrameLoads {
        self.loads
    }
}

struct Pending {
    owner: Arc<()>,
    frame: u64,
    duration: f64,
    completion: Arc<OnceLock<std::result::Result<(), wgpu::BufferAsyncError>>>,
}

#[derive(Default)]
pub(super) struct State {
    owner: Arc<()>,
    pending: Option<Pending>,
    last_requested: Option<u64>,
    failure: Option<String>,
}

impl FrameLoads {
    pub(super) fn check_observation(&self, engine: &Engine, stamp: &SceneStamp) -> Result<()> {
        engine.check_device()?;
        molla_kernels::validate_soil_device(
            engine
                .device()
                .ok_or_else(|| invalid("GPU engine required"))?,
            &self.device,
        )?;
        if !self.matches_stamp(stamp) {
            return Err(invalid("stale frame load body association"));
        }
        if let Some(error) = &self.observations.failure {
            return Err(Error::Gpu(error.clone()));
        }
        Ok(())
    }

    pub(in super::super) fn observation_pending(&self) -> bool {
        self.observations.pending.is_some()
    }

    pub(in super::super) fn is_current_observation(
        &self,
        sample: &MpmSoilFrameLoadObservation,
    ) -> bool {
        self.observations.failure.is_none() && Arc::ptr_eq(&self.observations.owner, &sample.owner)
    }

    pub(in super::super) fn reset_observations(
        &mut self,
        engine: &Engine,
        stamp: &SceneStamp,
    ) -> Result<()> {
        self.check_observation(engine, stamp)?;
        self.observations.owner = Arc::new(());
        self.observations.last_requested = None;
        Ok(())
    }

    pub(in super::super) fn request_observation(
        &mut self,
        engine: &Engine,
        stamp: &SceneStamp,
    ) -> Result<bool> {
        self.check_observation(engine, stamp)?;
        if self.observation_pending() {
            return Ok(false);
        }
        let (index, frame, duration) = self
            .ready
            .ok_or_else(|| invalid("no committed frame load sample"))?;
        if self
            .observations
            .last_requested
            .is_some_and(|old| old >= frame)
        {
            return Err(invalid(
                "frame load revisions must increase; reset after replay",
            ));
        }
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(
            &self.buffers[index],
            0,
            &self.staging,
            0,
            self.staging.size(),
        );
        engine.queue().unwrap().submit([encoder.finish()]);
        let completion = Arc::new(OnceLock::new());
        let ready = completion.clone();
        self.staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = ready.set(result);
            });
        self.observations.pending = Some(Pending {
            owner: self.observations.owner.clone(),
            frame,
            duration,
            completion,
        });
        self.observations.last_requested = Some(frame);
        Ok(true)
    }

    fn fail_observation(&mut self, message: String) -> Error {
        self.staging.unmap();
        self.observations.pending = None;
        self.observations.failure = Some(message.clone());
        Error::Gpu(message)
    }

    pub(in super::super) fn poll_observation(
        &mut self,
        engine: &Engine,
        stamp: &SceneStamp,
    ) -> Result<Option<MpmSoilFrameLoadObservation>> {
        self.check_observation(engine, stamp)?;
        if !self.observation_pending() {
            return Ok(None);
        }
        if let Err(error) = self.device.poll(wgpu::PollType::Poll) {
            return Err(self.fail_observation(format!("frame load poll: {error}")));
        }
        let pending = self.observations.pending.as_ref().unwrap();
        match pending.completion.get() {
            None => return Ok(None),
            Some(Err(error)) => {
                return Err(self.fail_observation(format!("frame load map: {error}")));
            }
            Some(Ok(())) => {}
        }
        let pending = self.observations.pending.take().unwrap();
        if !Arc::ptr_eq(&pending.owner, &self.observations.owner) {
            self.staging.unmap();
            return Ok(None);
        }
        let slice = self.staging.slice(..);
        let data = molla_core::gpu_readback!(slice);
        let result = self.decode(pending.frame, pending.duration, bytemuck::cast_slice(&data));
        drop(data);
        self.staging.unmap();
        result.map(|loads| {
            Some(MpmSoilFrameLoadObservation {
                owner: pending.owner,
                loads,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};

    fn drain(
        loads: &mut FrameLoads,
        engine: &Engine,
        stamp: &SceneStamp,
    ) -> Option<MpmSoilFrameLoadObservation> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let sample = loads.poll_observation(engine, stamp).unwrap();
            if sample.is_some() || !loads.observation_pending() {
                return sample;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "load observation deadline"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    #[test]
    #[ignore = "requires hardware GPU; delayed loads, generations and retained staging"]
    fn frame_load_observation_preserves_capture_and_bounded_lifecycle() {
        let engine = Engine::new_gpu().unwrap();
        assert!(!engine.adapter_info().unwrap().is_software);
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        let mut scene = RigidScene::new().unwrap();
        let bodies: Vec<_> = (0..65)
            .map(|_| scene.insert_body(Default::default()).unwrap())
            .collect();
        let stamp = scene.model_stamp();
        let buffer = |size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("async load fixture"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let reaction = buffer(65 * 32);
        let centers = buffer(65 * 16);
        let new_loads =
            || FrameLoads::new(device, &reaction, &centers, bodies.clone(), stamp.clone()).unwrap();
        let mut loads = new_loads();
        assert_eq!(loads.retained_bytes(), 65 * 144 + 16);
        assert!(loads.request_observation(&engine, &stamp).is_err());
        assert!(loads.poll_observation(&engine, &stamp).unwrap().is_none());
        let rows: Vec<[f32; 12]> = (0..65)
            .map(|i| std::array::from_fn(|j| (i * 12 + j) as f32))
            .collect();
        queue.write_buffer(&loads.buffers[0], 0, bytemuck::cast_slice(&rows));
        loads.ready = Some((0, 5, 0.25));
        let reference = loads
            .read(&engine, &stamp, std::time::Duration::from_secs(5))
            .unwrap();
        let other = new_loads();
        let allocations = DeviceScope::start().unwrap();
        let traffic = TransferScope::start().unwrap();
        assert!(loads.request_observation(&engine, &stamp).unwrap());
        assert!(!loads.request_observation(&engine, &stamp).unwrap());
        assert!(
            loads
                .read(&engine, &stamp, std::time::Duration::from_secs(5))
                .is_err()
        );
        for buffer in &loads.buffers {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&vec![[0.0_f32; 12]; 65]));
        }
        loads.ready = Some((1, 6, 0.5));
        let captured = drain(&mut loads, &engine, &stamp).unwrap();
        assert_eq!(captured.loads().committed_frame, 5);
        assert_eq!(
            captured.loads().simulated_seconds,
            reference.simulated_seconds
        );
        for (a, b) in captured.loads().bodies.iter().zip(&reference.bodies) {
            assert_eq!(a.body, b.body);
            assert_eq!(a.linear, b.linear);
            assert_eq!(a.angular, b.angular);
            assert_eq!(a.world_origin_angular, b.world_origin_angular);
        }
        assert!(loads.is_current_observation(&captured));
        assert!(!other.is_current_observation(&captured));
        let bytes = traffic.finish();
        assert_eq!(bytes.readback_bytes, 65 * 48);
        let observed = allocations.finish();
        assert_eq!(observed.observations, 0);

        let allocations = DeviceScope::start().unwrap();
        let traffic = TransferScope::start().unwrap();
        assert!(loads.request_observation(&engine, &stamp).unwrap());
        loads.reset_observations(&engine, &stamp).unwrap();
        assert!(!loads.is_current_observation(&captured));
        assert!(!loads.request_observation(&engine, &stamp).unwrap());
        assert!(drain(&mut loads, &engine, &stamp).is_none());
        assert_eq!(traffic.finish().readback_bytes, 0);
        assert!(loads.request_observation(&engine, &stamp).unwrap());
        let fresh = drain(&mut loads, &engine, &stamp).unwrap();
        assert_eq!(fresh.loads().committed_frame, 6);
        assert!(fresh.loads().bodies.iter().all(|b| b.linear == [0.0; 3]));
        assert!(loads.request_observation(&engine, &stamp).is_err());
        assert_eq!(allocations.finish().observations, 0);

        let foreign = Engine::new_gpu().unwrap();
        assert!(loads.poll_observation(&foreign, &stamp).is_err());
        assert!(loads.request_observation(&foreign, &stamp).is_err());
        assert!(
            loads
                .read(&foreign, &stamp, std::time::Duration::from_secs(5))
                .is_err()
        );
        scene.remove_body(bodies[0]).unwrap();
        assert!(
            loads
                .request_observation(&engine, &scene.model_stamp())
                .is_err()
        );
        assert!(
            loads
                .poll_observation(&engine, &scene.model_stamp())
                .is_err()
        );
        assert!(
            loads
                .reset_observations(&engine, &scene.model_stamp())
                .is_err()
        );

        loads.reset_observations(&engine, &stamp).unwrap();
        queue.write_buffer(&loads.buffers[1], 0, bytemuck::bytes_of(&f32::NAN));
        assert!(loads.request_observation(&engine, &stamp).unwrap());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            match loads.poll_observation(&engine, &stamp) {
                Err(_) => break,
                Ok(None) => {
                    assert!(std::time::Instant::now() < deadline);
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                Ok(Some(_)) => panic!("nonfinite impulse published"),
            }
        }
        assert!(!loads.observation_pending());
        loads.reset_observations(&engine, &stamp).unwrap();
        queue.write_buffer(&loads.buffers[1], 0, bytemuck::bytes_of(&0.0_f32));
        assert!(loads.request_observation(&engine, &stamp).unwrap());
        assert!(drain(&mut loads, &engine, &stamp).is_some());
        loads.reset_observations(&engine, &stamp).unwrap();
        assert!(loads.request_observation(&engine, &stamp).unwrap());
        drop(loads);
        engine.check_device().unwrap();
        eprintln!(
            "async loads: 65 bodies, captured-buffer overwrite parity, saturation, generation discard, foreign owner/device, stale model, invalid payload recovery, queued drop"
        );
    }
}
