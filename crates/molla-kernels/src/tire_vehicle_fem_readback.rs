use super::*;
use molla_compute::Engine;
use std::sync::{
    Arc,
    atomic::{AtomicU8, Ordering},
};

/// Fresh node/hub/chassis data; no FEM force rows or delayed telemetry.
pub struct TireVehicleFemFrame {
    rows: Vec<[f32; 4]>,
    ranges: Vec<std::ops::Range<usize>>,
    nodes: usize,
    wheels: usize,
    cavity_status: Option<[u32; 4]>,
}

impl TireVehicleFemFrame {
    /// Boundary admission only, not a completed-substep or solver-stability certificate.
    pub fn cavity_status(&self) -> Option<[u32; 4]> {
        self.cavity_status
    }

    pub fn positions(&self, tire: usize) -> Option<&[[f32; 4]]> {
        self.ranges.get(tire).map(|r| &self.rows[r.clone()])
    }
    pub fn velocities(&self, tire: usize) -> Option<&[[f32; 4]]> {
        self.ranges
            .get(tire)
            .map(|r| &self.rows[self.nodes + r.start..self.nodes + r.end])
    }
    pub fn hub_rows(&self) -> &[[f32; 4]] {
        &self.rows[2 * self.nodes..2 * self.nodes + 2 * self.wheels]
    }
    pub fn planar_rows(&self) -> &[[f32; 4]] {
        let start = 2 * self.nodes + 2 * self.wheels;
        &self.rows[start..start + 3]
    }
    pub fn vertical_row(&self) -> [f32; 4] {
        self.rows[2 * self.nodes + 2 * self.wheels + 3]
    }
    pub fn spin_rows(&self) -> &[[f32; 4]] {
        let start = 2 * self.nodes + 2 * self.wheels + 4;
        &self.rows[start..start + self.wheels]
    }
    pub fn suspension_rows(&self) -> &[[f32; 4]] {
        let start = 2 * self.nodes + 3 * self.wheels + 4;
        &self.rows[start..start + self.wheels]
    }
}

/// Fixed-capacity blocking frame bridge with one submit/map/wait and retained storage.
/// Returned data is overwritten on the next read. A mapping/poll failure latches
/// this reader unusable; reconstruct it rather than reusing a pending mapping.
/// Cavity mode includes its admission header and rejects unadmitted/faulted frames.
pub struct PreparedTireVehicleFemReadback {
    stage: wgpu::Buffer,
    source: wgpu::Buffer,
    cavity_source: Option<wgpu::Buffer>,
    completion: Arc<AtomicU8>,
    failed: bool,
    frame: TireVehicleFemFrame,
}

impl PreparedTireVehicleFemReadback {
    pub fn required_bytes(nodes: usize, wheels: usize) -> Option<u64> {
        u64::try_from(nodes)
            .ok()?
            .checked_mul(32)?
            .checked_add(u64::try_from(wheels).ok()?.checked_mul(64)?)?
            .checked_add(64)
    }

    pub fn required_bytes_with_cavity(nodes: usize, wheels: usize) -> Option<u64> {
        Self::required_bytes(nodes, wheels)?.checked_add(16)
    }

    pub fn new(
        engine: &Engine,
        step: &PreparedTireVehicleFemStep,
        owner: &TireResidentVehicleFemAtlasOwner,
        max_staging_bytes: u64,
    ) -> Result<Self> {
        step.validate(engine, owner)?;
        let nodes = owner.fem.total_node_count();
        let wheels = owner.tire_count();
        let cavity_source = step.cavity_status_buffer().cloned();
        let bytes = if cavity_source.is_some() {
            Self::required_bytes_with_cavity(nodes, wheels)
        } else {
            Self::required_bytes(nodes, wheels)
        }
        .ok_or_else(|| Error::Solver("vehicle FEM readback size overflow".into()))?;
        let device = engine.device().unwrap();
        if bytes > max_staging_bytes || bytes > device.limits().max_buffer_size {
            return Err(Error::Solver("vehicle FEM readback staging budget".into()));
        }
        let stage = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("retained vehicle FEM frame staging"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }
        );
        Ok(Self {
            stage,
            source: owner.fem.pos_a.clone(),
            cavity_source,
            completion: Arc::new(AtomicU8::new(0)),
            failed: false,
            frame: TireVehicleFemFrame {
                rows: vec![[0.; 4]; Self::required_bytes(nodes, wheels).unwrap() as usize / 16],
                ranges: owner
                    .fem
                    .atlas
                    .layout
                    .rows
                    .iter()
                    .map(|r| r.node_offset..r.node_offset + r.node_count)
                    .collect(),
                nodes,
                wheels,
                cavity_status: None,
            },
        })
    }

    pub fn read_blocking(
        &mut self,
        engine: &Engine,
        step: &PreparedTireVehicleFemStep,
        owner: &TireResidentVehicleFemAtlasOwner,
    ) -> Result<&TireVehicleFemFrame> {
        step.validate(engine, owner)?;
        if self.failed
            || self.source != owner.fem.pos_a
            || self.cavity_source.as_ref() != step.cavity_status_buffer()
        {
            return Err(Error::Gpu(
                "failed or foreign vehicle FEM frame reader".into(),
            ));
        }
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        let fem = &owner.fem;
        let (pos, vel) = if fem.parity {
            (&fem.pos_b, &fem.vel_b)
        } else {
            (&fem.pos_a, &fem.vel_a)
        };
        let n = self.frame.nodes as u64;
        let w = self.frame.wheels as u64;
        let mut offset = 0;
        for (source, bytes) in [
            (pos, 16 * n),
            (vel, 16 * n),
            (&fem.hub_state_buf, 32 * w),
            (owner.vehicle.planar.current_state_buffer(), 48),
            (owner.vehicle.vertical.current_state_buffer(), 16),
            (owner.vehicle.brush.spin.current_state_buffer(), 16 * w),
            (owner.vehicle.brush.suspension.rows_buffer(), 16 * w),
        ] {
            if bytes != 0 {
                encoder.copy_buffer_to_buffer(source, 0, &self.stage, offset, bytes);
            }
            offset += bytes;
        }
        if let Some(source) = &self.cavity_source {
            encoder.copy_buffer_to_buffer(source, 0, &self.stage, offset, 16);
        }
        queue.submit([encoder.finish()]);
        self.failed = true;
        self.completion.store(0, Ordering::Release);
        let completion = self.completion.clone();
        self.stage
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                completion.store(if r.is_ok() { 1 } else { 2 }, Ordering::Release);
            });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("vehicle FEM frame poll: {e:?}")))?;
        if self.completion.load(Ordering::Acquire) != 1 {
            return Err(Error::Gpu("vehicle FEM frame mapping failed".into()));
        }
        let data = molla_core::gpu_readback!(self.stage.slice(..));
        for (row, bytes) in self.frame.rows.iter_mut().zip(data.chunks_exact(16)) {
            *row = bytemuck::pod_read_unaligned(bytes);
        }
        self.frame.cavity_status = self
            .cavity_source
            .as_ref()
            .map(|_| bytemuck::pod_read_unaligned(&data[offset as usize..offset as usize + 16]));
        drop(data);
        self.stage.unmap();
        self.failed = false;
        if let Some(status) = self.frame.cavity_status {
            if status[0] != 0 || status[1] != 1 || status[2] == 0 || status[3] != 0 {
                return Err(Error::Solver(format!(
                    "vehicle FEM cavity frame not admitted: {status:?}"
                )));
            }
        }
        Ok(&self.frame)
    }
}

#[test]
fn vehicle_fem_frame_layout() {
    assert_eq!(
        PreparedTireVehicleFemReadback::required_bytes(288 * 5, 5),
        Some(46_464)
    );
    let frame = TireVehicleFemFrame {
        rows: (0..28).map(|i| [i as f32; 4]).collect(),
        ranges: vec![0..4, 4..8],
        nodes: 8,
        wheels: 2,
        cavity_status: Some([0, 1, 7, 0]),
    };
    assert_eq!(frame.positions(1).unwrap()[0][0], 4.);
    assert_eq!(frame.velocities(1).unwrap()[0][0], 12.);
    assert_eq!(frame.hub_rows()[0][0], 16.);
    assert_eq!(frame.planar_rows()[0][0], 20.);
    assert_eq!(frame.vertical_row()[0], 23.);
    assert_eq!(frame.spin_rows()[0][0], 24.);
    assert_eq!(frame.suspension_rows()[0][0], 26.);
    assert!(frame.positions(2).is_none());
    assert_eq!(frame.cavity_status(), Some([0, 1, 7, 0]));
    assert_eq!(
        PreparedTireVehicleFemReadback::required_bytes(8, 2),
        Some(448)
    );
    assert_eq!(
        PreparedTireVehicleFemReadback::required_bytes_with_cavity(8, 2),
        Some(464)
    );
    assert_eq!(
        PreparedTireVehicleFemReadback::required_bytes_with_cavity(usize::MAX, 2),
        None
    );
}
