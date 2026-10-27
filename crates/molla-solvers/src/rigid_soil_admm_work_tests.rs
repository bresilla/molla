use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

pub(crate) struct Trace {
    report: wgpu::Buffer,
    sources: [[wgpu::Buffer; 3]; 2],
    next: AtomicUsize,
    capacity: usize,
    dofs: u32,
}

fn validate_shape(dofs: u32, capacity: usize) -> Result<()> {
    if !(1..=32).contains(&dofs) || !(1..=1122).contains(&capacity) {
        return Err(invalid("ADMM work trace shape"));
    }
    Ok(())
}

fn decode(dofs: u32, raw: &[u8]) -> Result<serde_json::Value> {
    if raw.len() % 32 != 0 {
        return Err(invalid("ADMM work trace length"));
    }
    validate_shape(dofs, raw.len() / 32)?;
    let slots = crate::soil_admm_gpu::ExperimentalSoilAdmmGpu::response_cache_slots(dofs)?;
    let mut rows = Vec::new();
    for (substep, bytes) in raw.chunks_exact(32).enumerate() {
        let w: [u32; 8] = bytemuck::pod_read_unaligned(bytes);
        if w[0] > 16384 || w[1] != 0 || w[2] != 64 || w[3] != 1 || w[4] != 0 || w[0] != w[6] {
            return Err(invalid("ADMM work trace failed or inconsistent solve"));
        }
        rows.push(serde_json::json!({"substep":substep,"active":w[0],"cached":w[0]<=slots,"steps":w[2],"factors":w[3]}));
    }
    Ok(serde_json::json!({
        "dofs":dofs,"cache_slots":slots,"substeps":rows.len(),
        "cached_substeps":rows.iter().filter(|r|r["cached"]==true).count(),
        "overflow_substeps":rows.iter().filter(|r|r["cached"]==false).count(),
        "empty_substeps":rows.iter().filter(|r|r["active"]==0).count(),
        "records":rows,"acceptance":false,
        "scope":"test-only copies in existing submission; one read after all measured frames; not performance acceptance"
    }))
}

impl Trace {
    pub(crate) fn new(
        device: &wgpu::Device,
        dofs: u32,
        capacity: usize,
        sources: [[wgpu::Buffer; 3]; 2],
    ) -> Result<Self> {
        validate_shape(dofs, capacity)?;
        if sources.iter().any(|s| {
            s.iter()
                .zip([4, 12, 16])
                .any(|(b, n)| b.size() < n || !b.usage().contains(wgpu::BufferUsages::COPY_SRC))
        }) {
            return Err(invalid("ADMM work trace source"));
        }
        Ok(Self {
            report: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("diagnostic ADMM active-set trace"),
                size: capacity as u64 * 32,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            sources,
            next: AtomicUsize::new(0),
            capacity,
            dofs,
        })
    }

    pub(crate) fn validate_next(&self) -> Result<()> {
        if self.next.load(Ordering::Relaxed) >= self.capacity {
            return Err(invalid("ADMM work trace exhausted"));
        }
        Ok(())
    }

    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder, current: usize) {
        let next = self.next.fetch_add(1, Ordering::Relaxed);
        assert!(next < self.capacity);
        for (i, (offset, bytes)) in [(0, 4), (4, 12), (16, 16)].into_iter().enumerate() {
            encoder.copy_buffer_to_buffer(
                &self.sources[current][i],
                0,
                &self.report,
                next as u64 * 32 + offset,
                bytes,
            );
        }
    }

    pub(crate) fn save(&self, engine: &Engine, root: &std::path::Path) -> Result<()> {
        if self.next.load(Ordering::Relaxed) != self.capacity {
            return Err(invalid("incomplete ADMM work trace"));
        }
        let raw = wall_runtime_read(engine, &self.report);
        std::fs::write(root.join("admm-work.bin"), &raw).unwrap();
        let report = decode(self.dofs, &raw)?;
        std::fs::write(
            root.join("admm-work.json"),
            serde_json::to_vec_pretty(&report).unwrap(),
        )
        .unwrap();
        Ok(())
    }
}

#[test]
fn admm_work_trace_shape_and_decoder() {
    for n in [0, 33, u32::MAX] {
        assert!(validate_shape(n, 102).is_err());
    }
    for count in [0, 1123, usize::MAX] {
        assert!(validate_shape(10, count).is_err());
    }
    for count in [1, 102, 1122] {
        validate_shape(10, count).unwrap();
    }
    let cap = crate::soil_admm_gpu::ExperimentalSoilAdmmGpu::response_cache_slots(10).unwrap();
    let raw: Vec<u32> = [0, cap - 1, cap, cap + 1, 16384]
        .into_iter()
        .flat_map(|n| [n, 0, 64, 1, 0, 0, n, 0])
        .collect();
    let result = decode(10, bytemuck::cast_slice(&raw)).unwrap();
    assert_eq!(result["cached_substeps"], 3);
    assert_eq!(result["overflow_substeps"], 2);
    assert_eq!(result["empty_substeps"], 1);
    assert!(decode(10, &[0; 31]).is_err());
    for (index, value) in [(0, 16385), (1, 1), (2, 63), (3, 0), (4, 1), (6, 1)] {
        let mut bad = raw.clone();
        bad[index] = value;
        assert!(decode(10, bytemuck::cast_slice(&bad)).is_err());
    }
}
