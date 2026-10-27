use super::*;

fn describe(words: [u32; 4]) -> Option<String> {
    if words[2] == 0 {
        return None;
    }
    let mut labels = Vec::new();
    for (mask, name) in [(1, "tile-overflow"), (2, "numerical"), (4, "motion-bound")] {
        if words[2] & mask != 0 {
            labels.push(name);
        }
    }
    if words[2] & !7 != 0 {
        labels.push("unknown");
    }
    Some(format!(
        "soil grid flags={}({}), active-tiles={}, g2p-counter={}",
        words[2],
        labels.join("|"),
        words[0],
        words[3]
    ))
}

pub(super) fn detail(
    soil: &RuntimeSoil,
    engine: &Engine,
    completion: &soil_completion::CompletionDeadline<'_>,
) -> Result<Option<String>> {
    let mut details = Vec::new();
    if let Some((source, offset)) = soil.soil.grid_fault_metadata() {
        let device = engine
            .device()
            .ok_or_else(|| invalid("GPU device required"))?;
        let stage = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("failed soil grid metadata"),
                size: 16,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }
        );
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(source, offset, &stage, 0, 16);
        engine.queue().unwrap().submit([encoder.finish()]);
        map_read_with_completion(device, &stage, Some(completion))?;
        let mapped = molla_core::gpu_readback!(stage.slice(..));
        let words = bytemuck::pod_read_unaligned::<[u32; 4]>(&mapped);
        drop(mapped);
        stage.unmap();
        details.extend(describe(words));
    }
    if let Some(projection) = &soil.projection {
        details.push(projection.fault_detail(engine, completion)?);
    }
    Ok((!details.is_empty()).then(|| details.join("; ")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soil_grid_fault_labels_preserve_flags_and_counter() {
        assert_eq!(
            describe([3, 0, 4, 2]),
            Some("soil grid flags=4(motion-bound), active-tiles=3, g2p-counter=2".into())
        );
        assert_eq!(describe([0, 0, 0, 0]), None);
        assert_eq!(describe([8, 1, 0, u32::MAX]), None);
        assert_eq!(
            describe([8, 0, 15, u32::MAX]),
            Some("soil grid flags=15(tile-overflow|numerical|motion-bound|unknown), active-tiles=8, g2p-counter=4294967295".into())
        );
    }
}
