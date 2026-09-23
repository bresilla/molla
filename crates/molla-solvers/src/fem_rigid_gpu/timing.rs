use molla_core::{Error, Result};

const PAGE: u32 = 256;

pub(super) struct SubstepTiming {
    queries: Vec<wgpu::QuerySet>,
    ticks: wgpu::Buffer,
    pub iterations: u32,
    pub ready: bool,
}

impl SubstepTiming {
    pub fn new(device: &wgpu::Device, iterations: u32) -> Result<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY)
            || !(1..=4096).contains(&iterations)
        {
            return Err(Error::Build(
                "FEM timing requires timestamp queries and valid iterations".into(),
            ));
        }
        let count = 2 * iterations + 3;
        let queries = (0..count)
            .step_by(PAGE as usize)
            .map(|start| {
                device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("FEM substep phase timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: (count - start).min(PAGE),
                })
            })
            .collect();
        let ticks = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FEM substep phase ticks"),
            size: u64::from(count) * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Ok(Self {
            queries,
            ticks,
            iterations,
            ready: false,
        })
    }

    pub fn mark(&self, encoder: &mut wgpu::CommandEncoder, slot: u32) {
        let _pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FEM phase boundary"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &self.queries[(slot / PAGE) as usize],
                beginning_of_pass_write_index: Some(slot % PAGE),
                end_of_pass_write_index: None,
            }),
        });
    }

    pub fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let count = 2 * self.iterations + 3;
        for (page, queries) in self.queries.iter().enumerate() {
            let start = page as u32 * PAGE;
            encoder.resolve_query_set(
                queries,
                0..(count - start).min(PAGE),
                &self.ticks,
                u64::from(start) * 8,
            );
        }
        self.ready = true;
    }

    pub fn ticks(&self) -> Option<&wgpu::Buffer> {
        self.ready.then_some(&self.ticks)
    }
}
