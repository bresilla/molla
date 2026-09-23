use molla_core::{Error, Result};

const PAGE: u32 = 256;

pub(super) struct SubstepTiming {
    phases: TimestampPages,
    contacts: TimestampPages,
    pub iterations: u32,
    pub ready: bool,
}

struct TimestampPages {
    queries: Vec<wgpu::QuerySet>,
    ticks: wgpu::Buffer,
    count: u32,
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
        Ok(Self {
            phases: TimestampPages::new(device, 2 * iterations + 3),
            contacts: TimestampPages::new(device, 2 * iterations),
            iterations,
            ready: false,
        })
    }

    pub fn mark(&self, encoder: &mut wgpu::CommandEncoder, slot: u32) {
        self.phases.mark(encoder, slot);
    }

    pub fn mark_contact(&self, encoder: &mut wgpu::CommandEncoder, slot: u32) {
        self.contacts.mark(encoder, slot);
    }

    pub fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.phases.resolve(encoder);
        self.contacts.resolve(encoder);
        self.ready = true;
    }

    pub fn ticks(&self) -> Option<&wgpu::Buffer> {
        self.ready.then_some(&self.phases.ticks)
    }

    pub fn contact_ticks(&self) -> Option<&wgpu::Buffer> {
        self.ready.then_some(&self.contacts.ticks)
    }
}

impl TimestampPages {
    fn new(device: &wgpu::Device, count: u32) -> Self {
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
        Self {
            queries,
            ticks,
            count,
        }
    }

    fn mark(&self, encoder: &mut wgpu::CommandEncoder, slot: u32) {
        let _pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FEM phase boundary"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &self.queries[(slot / PAGE) as usize],
                beginning_of_pass_write_index: Some(slot % PAGE),
                end_of_pass_write_index: None,
            }),
        });
    }

    fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        for (page, queries) in self.queries.iter().enumerate() {
            let start = page as u32 * PAGE;
            encoder.resolve_query_set(
                queries,
                0..(self.count - start).min(PAGE),
                &self.ticks,
                u64::from(start) * 8,
            );
        }
    }
}
