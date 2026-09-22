use std::collections::VecDeque;
use wgpu::util::DeviceExt;

#[derive(PartialEq, Eq)]
struct BufferKey {
    binding: u32,
    buffer: wgpu::Buffer,
    offset: u64,
    size: Option<wgpu::BufferSize>,
}

struct StorageEntry {
    layout: wgpu::BindGroupLayout,
    buffers: Vec<BufferKey>,
    group: wgpu::BindGroup,
}

struct UniformEntry {
    layout: wgpu::BindGroupLayout,
    bytes: Vec<u8>,
    group: wgpu::BindGroup,
}

/// Bounded LRU groups with immutable uniform contents and buffer-identity keys.
pub(super) struct BindingCache {
    capacity: usize,
    storage: VecDeque<StorageEntry>,
    uniforms: VecDeque<UniformEntry>,
}

impl BindingCache {
    pub(super) fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self {
            capacity,
            storage: VecDeque::new(),
            uniforms: VecDeque::new(),
        }
    }

    pub(super) fn storage_group(
        &mut self,
        device: &wgpu::Device,
        descriptor: &wgpu::BindGroupDescriptor<'_>,
    ) -> wgpu::BindGroup {
        if let Some(index) = self.storage.iter().position(|entry| {
            entry.layout == *descriptor.layout
                && entry.buffers.len() == descriptor.entries.len()
                && entry
                    .buffers
                    .iter()
                    .zip(descriptor.entries)
                    .all(|(key, entry)| {
                        matches!(&entry.resource, wgpu::BindingResource::Buffer(buffer)
                        if key.binding == entry.binding && key.buffer == *buffer.buffer
                            && key.offset == buffer.offset && key.size == buffer.size)
                    })
        }) {
            let entry = self.storage.remove(index).unwrap();
            let group = entry.group.clone();
            self.storage.push_back(entry);
            return group;
        }
        let Some(buffers): Option<Vec<_>> = descriptor
            .entries
            .iter()
            .map(|entry| match &entry.resource {
                wgpu::BindingResource::Buffer(buffer) => Some(BufferKey {
                    binding: entry.binding,
                    buffer: buffer.buffer.clone(),
                    offset: buffer.offset,
                    size: buffer.size,
                }),
                _ => None,
            })
            .collect()
        else {
            return device.create_bind_group(descriptor);
        };
        let group = device.create_bind_group(descriptor);
        if self.storage.len() == self.capacity {
            self.storage.pop_front();
        }
        self.storage.push_back(StorageEntry {
            layout: descriptor.layout.clone(),
            buffers,
            group: group.clone(),
        });
        group
    }

    pub(super) fn uniform_group(
        &mut self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        bytes: &[u8],
    ) -> wgpu::BindGroup {
        if let Some(index) = self
            .uniforms
            .iter()
            .position(|entry| entry.layout == *layout && entry.bytes == bytes)
        {
            let entry = self.uniforms.remove(index).unwrap();
            let group = entry.group.clone();
            self.uniforms.push_back(entry);
            return group;
        }
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("molla::vbd_cloth cached uniforms"),
            contents: bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::vbd_cloth cached uniforms bg"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        if self.uniforms.len() == self.capacity {
            self.uniforms.pop_front();
        }
        self.uniforms.push_back(UniformEntry {
            layout: layout.clone(),
            bytes: bytes.to_vec(),
            group: group.clone(),
        });
        group
    }

    #[cfg(test)]
    pub(super) fn entry_counts(&self) -> (usize, usize) {
        (self.storage.len(), self.uniforms.len())
    }
}
