use super::*;

#[derive(Clone, Debug)]
pub struct MpmSoilMachineTiming {
    pub stage: &'static str,
    pub substep: Option<usize>,
    /// GPU offset from the first machine timestamp, not from the first MPM timestamp.
    pub start_ms: f64,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MpmSoilNewtonWork {
    pub substep: usize,
    pub faults: [u32; 4],
    pub steps: u32,
    /// Krylov iterations plus condensed solves, not exclusively Krylov work.
    pub linear_iterations: u32,
    pub trials: u32,
}

#[derive(Default)]
pub(super) struct MachineProfile {
    pub stages: Vec<MpmSoilMachineTiming>,
    pub work: Vec<MpmSoilNewtonWork>,
    pub readback_bytes: u64,
}

fn decode_work(bytes: &[u8]) -> Result<Vec<MpmSoilNewtonWork>> {
    if bytes.len() % 32 != 0 {
        return Err(invalid("invalid Newton work profile size"));
    }
    bytes
        .chunks_exact(32)
        .enumerate()
        .map(|(substep, record)| {
            let words: [u32; 8] = std::array::from_fn(|i| {
                u32::from_le_bytes(record[i * 4..i * 4 + 4].try_into().unwrap())
            });
            if words[7] != 0 {
                return Err(invalid("unsupported Newton work profile padding"));
            }
            Ok(MpmSoilNewtonWork {
                substep,
                faults: words[..4].try_into().unwrap(),
                steps: words[4],
                linear_iterations: words[5],
                trials: words[6],
            })
        })
        .collect()
}

pub(super) struct MachineProfiler {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    staging: wgpu::Buffer,
    entries: Vec<(&'static str, Option<usize>)>,
    capacity: usize,
    work_capacity: usize,
    work_count: usize,
}

impl MachineProfiler {
    pub(super) fn new(device: &wgpu::Device, substeps: usize) -> Result<Self> {
        let capacity = substeps
            .checked_mul(3)
            .and_then(|n| n.checked_add(1))
            .filter(|&n| substeps > 0 && n <= wgpu::QUERY_SET_MAX_QUERIES as usize / 2)
            .ok_or_else(|| invalid("machine profile capacity exceeds query limit"))?;
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return Err(invalid("machine profiling requires timestamp queries"));
        }
        let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("coupled machine profile"),
            ty: wgpu::QueryType::Timestamp,
            count: capacity as u32 * 2,
        });
        let buffer = |label, usage, size| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage,
                    mapped_at_creation: false,
                }
            )
        };
        Ok(Self {
            queries,
            resolve: buffer(
                "machine timestamp resolve",
                wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                capacity as u64 * 16,
            ),
            staging: buffer(
                "machine timestamp readback",
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                capacity as u64 * 16 + substeps as u64 * 32,
            ),
            entries: Vec::with_capacity(capacity),
            capacity,
            work_capacity: substeps,
            work_count: 0,
        })
    }

    pub(super) fn reset(&mut self) {
        self.entries.clear();
        self.work_count = 0;
    }

    pub(super) fn validate_work_slot(&self) -> Result<()> {
        if self.work_count >= self.work_capacity {
            return Err(invalid("Newton work profile capacity exceeded"));
        }
        Ok(())
    }

    pub(super) fn encode_work(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Buffer,
    ) {
        assert!(self.work_count < self.work_capacity);
        encoder.copy_buffer_to_buffer(
            source,
            0,
            &self.staging,
            self.capacity as u64 * 16 + self.work_count as u64 * 32,
            32,
        );
        self.work_count += 1;
    }

    pub(super) fn timestamp(
        &mut self,
        stage: &'static str,
        substep: Option<usize>,
    ) -> Option<(wgpu::QuerySet, [u32; 2])> {
        if self.entries.len() == self.capacity {
            return None;
        }
        let index = self.entries.len() as u32 * 2;
        self.entries.push((stage, substep));
        Some((self.queries.clone(), [index, index + 1]))
    }

    pub(super) fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        let count = self.entries.len() as u32 * 2;
        if count != 0 {
            encoder.resolve_query_set(&self.queries, 0..count, &self.resolve, 0);
            encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.staging, 0, count as u64 * 8);
        }
    }

    pub(super) fn read(
        &mut self,
        engine: &Engine,
        timeout: std::time::Duration,
    ) -> Result<MachineProfile> {
        if self.entries.is_empty() {
            if self.work_count != 0 {
                return Err(invalid("Newton work profile has no machine timestamps"));
            }
            return Ok(MachineProfile::default());
        }
        let device = engine.device().unwrap();
        let completion =
            soil_completion::CompletionDeadline::new(engine.device_health().unwrap(), timeout)?;
        let work_start = self.capacity as u64 * 16;
        let bytes = if self.work_count == 0 {
            self.entries.len() as u64 * 16
        } else {
            work_start + self.work_count as u64 * 32
        };
        let slice = self.staging.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let result = (|| {
            completion.poll(device)?;
            completion
                .receive(&rx)?
                .map_err(|e| Error::Gpu(e.to_string()))?;
            let data = molla_core::gpu_readback!(slice);
            let ticks: &[u64] = bytemuck::cast_slice(&data[..self.entries.len() * 16]);
            let scale = f64::from(engine.queue().unwrap().get_timestamp_period()) * 1e-6;
            let stages = self
                .entries
                .iter()
                .zip(ticks.chunks_exact(2))
                .map(|(&(stage, substep), pair)| {
                    let offset = pair[0]
                        .checked_sub(ticks[0])
                        .ok_or_else(|| invalid("machine timestamp precedes origin"))?;
                    let delta = pair[1]
                        .checked_sub(pair[0])
                        .ok_or_else(|| invalid("machine timestamp end precedes start"))?;
                    Ok(MpmSoilMachineTiming {
                        stage,
                        substep,
                        start_ms: offset as f64 * scale,
                        elapsed_ms: delta as f64 * scale,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let work = if self.work_count == 0 {
                Vec::new()
            } else {
                decode_work(&data[work_start as usize..])?
            };
            Ok(MachineProfile {
                stages,
                work,
                readback_bytes: bytes,
            })
        })();
        self.staging.unmap();
        self.entries.clear();
        self.work_count = 0;
        result
    }
}

#[cfg(all(test, feature = "resource-counters"))]
mod tests {
    use super::*;

    #[test]
    fn newton_work_profile_preserves_counts_and_faults() {
        let words = [1u32, 2, 4, 8, 64, 4097, 97, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let bytes: Vec<_> = words.iter().flat_map(|v| v.to_le_bytes()).collect();
        let work = decode_work(&bytes).unwrap();
        assert_eq!(work.len(), 2);
        assert_eq!(
            work[0],
            MpmSoilNewtonWork {
                substep: 0,
                faults: [1, 2, 4, 8],
                steps: 64,
                linear_iterations: 4097,
                trials: 97
            }
        );
        assert_eq!(work[1].substep, 1);
        assert_eq!(work[1].steps, 0);
        assert!(decode_work(&[]).unwrap().is_empty());
        assert!(decode_work(&bytes[..63]).is_err());
        let mut invalid = bytes;
        invalid[28] = 1;
        assert!(decode_work(&invalid).is_err());
    }

    #[test]
    #[ignore = "requires GPU timestamp queries"]
    fn machine_profile_retains_slots_and_reuses_without_allocations() {
        let engine = Engine::new_gpu_with_features(
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
        )
        .unwrap();
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        assert!(MachineProfiler::new(device, 0).is_err());
        assert!(MachineProfiler::new(device, usize::MAX).is_err());
        let mut profiler = MachineProfiler::new(device, 1).unwrap();
        use wgpu::util::DeviceExt;
        let words = [[0u32, 0, 0, 0, 5, 17, 9, 0], [0u32, 0, 0, 0, 8, 24, 8, 0]];
        let sources = words.map(|words| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Newton work profile fixture"),
                contents: bytemuck::cast_slice(&words),
                usage: wgpu::BufferUsages::COPY_SRC,
            })
        });
        for iteration in 0..3 {
            let with_work = iteration != 1;
            let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
            let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
            for (stage, step) in [
                ("refresh", None),
                ("dynamics", Some(0)),
                ("soil-corrected-drift", Some(0)),
                ("momentum", Some(0)),
            ] {
                let (queries, slots) = profiler.timestamp(stage, step).unwrap();
                let mut encoder = device.create_command_encoder(&Default::default());
                encoder.write_timestamp(&queries, slots[0]);
                encoder.write_timestamp(&queries, slots[1]);
                queue.submit([encoder.finish()]);
            }
            assert!(profiler.timestamp("overflow", Some(1)).is_none());
            let mut encoder = device.create_command_encoder(&Default::default());
            profiler.validate_work_slot().unwrap();
            if with_work {
                profiler.encode_work(&mut encoder, &sources[iteration / 2]);
                assert!(profiler.validate_work_slot().is_err());
            }
            profiler.resolve(&mut encoder);
            queue.submit([encoder.finish()]);
            assert_eq!(allocations.finish().observations, 0);
            assert_eq!(transfers.finish().readback_bytes, 0);
            let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
            let result = profiler
                .read(&engine, std::time::Duration::from_secs(5))
                .unwrap();
            let expected_bytes = if with_work { 96 } else { 64 };
            assert_eq!(transfers.finish().readback_bytes, expected_bytes);
            assert_eq!(result.readback_bytes, expected_bytes);
            assert_eq!(result.work.len(), usize::from(with_work));
            if with_work {
                assert_eq!(
                    result.work,
                    decode_work(bytemuck::cast_slice(&words[iteration / 2])).unwrap()
                );
            }
            profiler.validate_work_slot().unwrap();
            let stages = result.stages;
            assert_eq!(stages.len(), 4);
            assert_eq!(stages[0].start_ms, 0.0);
            assert_eq!(stages[0].stage, "refresh");
            assert_eq!(stages[1].substep, Some(0));
            assert_eq!(stages[2].stage, "soil-corrected-drift");
            assert_eq!(stages[3].stage, "momentum");
            assert!(
                stages
                    .iter()
                    .all(|s| s.elapsed_ms.is_finite() && s.elapsed_ms >= 0.0)
            );
            assert!(stages.windows(2).all(|s| s[0].start_ms <= s[1].start_ms));
            assert!(
                profiler
                    .read(&engine, std::time::Duration::from_secs(5))
                    .unwrap()
                    .stages
                    .is_empty()
            );
        }
        profiler.timestamp("cancelled", None).unwrap();
        profiler.reset();
        assert!(
            profiler
                .read(&engine, std::time::Duration::from_secs(5))
                .unwrap()
                .stages
                .is_empty()
        );
        let mut partial = MachineProfiler::new(device, 2).unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        partial.encode_work(&mut encoder, &sources[1]);
        queue.submit([encoder.finish()]);
        assert!(
            partial
                .read(&engine, std::time::Duration::from_secs(5))
                .is_err()
        );
        partial.reset();
        assert!(
            partial
                .read(&engine, std::time::Duration::from_secs(5))
                .unwrap()
                .work
                .is_empty()
        );
        let (query, slots) = partial.timestamp("partial", Some(0)).unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.write_timestamp(&query, slots[0]);
        encoder.write_timestamp(&query, slots[1]);
        partial.encode_work(&mut encoder, &sources[0]);
        partial.resolve(&mut encoder);
        queue.submit([encoder.finish()]);
        let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
        let result = partial
            .read(&engine, std::time::Duration::from_secs(5))
            .unwrap();
        assert_eq!(result.readback_bytes, 7 * 16 + 32);
        assert_eq!(transfers.finish().readback_bytes, result.readback_bytes);
        assert_eq!(result.stages.len(), 1);
        assert_eq!(
            result.work,
            decode_work(bytemuck::cast_slice(&words[0])).unwrap()
        );
        println!(
            "machine profile: cross-submission slots, bounded capacity, reuse, reset and zero inner allocation/readback pass"
        );
    }
}
