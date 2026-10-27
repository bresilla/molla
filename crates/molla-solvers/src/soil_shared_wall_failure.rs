use super::*;
use std::io::Write;
use std::path::Path;

impl ExperimentalSoilSharedWallFrictionGpu {
    pub(crate) fn capture_failure(
        &self,
        engine: &Engine,
        path: &Path,
        map: impl FnOnce(&wgpu::Buffer) -> Result<()>,
    ) -> Result<()> {
        let invalid = |message: String| Error::Solver(format!("shared failure capture: {message}"));
        if !cfg!(target_endian = "little") {
            return Err(invalid("unsupported byte order".into()));
        }
        let n = u64::from(self.capture_capacity.dofs);
        let c = u64::from(self.contacts.max(1));
        let rates = n * if self.paired_rates { 8 } else { 4 };
        let records = [
            ("mass", &self.capture_inputs[0], 4 * n * n),
            ("free", &self.capture_inputs[1], rates),
            ("rows", &self.capture_inputs[2], 16 * c * n),
            ("contacts", &self.capture_inputs[3], 64 * c),
            ("status", &self.status, self.status.size()),
            ("rates", &self.rates, rates),
            ("outputs", &self.outputs, 80 * c),
            ("conditional", &self.conditional_inputs, 112 * c),
            ("lower", &self.capture_storage[0], 4 * n * n),
            ("response", &self.capture_storage[1], 16 * c * n),
            ("params", &self.capture_storage[2], 16),
            (
                "history",
                &self.capture_storage[3],
                self.capture_storage[3].size(),
            ),
        ];
        let mut total = 0u64;
        for (name, buffer, size) in records {
            if size == 0
                || size % 4 != 0
                || size > buffer.size()
                || !buffer.usage().contains(wgpu::BufferUsages::COPY_SRC)
            {
                return Err(invalid(format!("invalid source buffer: {name}")));
            }
            total = total
                .checked_add(size)
                .ok_or_else(|| invalid("size overflow".into()))?;
        }
        if total > 32 * 1024 * 1024 {
            return Err(invalid("snapshot exceeds 32 MiB".into()));
        }
        let device = engine
            .device()
            .ok_or_else(|| invalid("GPU required".into()))?;
        if total > device.limits().max_buffer_size {
            return Err(invalid("snapshot exceeds device buffer limit".into()));
        }
        std::fs::create_dir(path).map_err(|e| invalid(e.to_string()))?;
        let staging = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("failed shared solve snapshot"),
                size: total,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }
        );
        let mut encoder = device.create_command_encoder(&Default::default());
        let mut offset = 0;
        for (_, buffer, size) in records {
            encoder.copy_buffer_to_buffer(buffer, 0, &staging, offset, size);
            offset += size;
        }
        engine.queue().unwrap().submit([encoder.finish()]);
        map(&staging)?;
        let mapped = molla_core::gpu_readback!(staging.slice(..));
        let bytes = mapped.to_vec();
        drop(mapped);
        staging.unmap();
        let status_offset = (4 * n * n + rates + 16 * c * n + 64 * c) as usize;
        if u32::from_le_bytes(bytes[status_offset..status_offset + 4].try_into().unwrap()) == 0 {
            return Err(invalid("solver has no failure".into()));
        }
        let write = |name: &str, data: &[u8]| -> Result<()> {
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path.join(name))
                .map_err(|e| invalid(e.to_string()))?;
            file.write_all(data).map_err(|e| invalid(e.to_string()))
        };
        let mut files = Vec::new();
        let mut offset = 0;
        for (name, _, size) in records {
            let data = &bytes[offset..offset + size as usize];
            write(&format!("{name}.bin"), data)?;
            files.push(format!(
                r#""{name}":{{"bytes":{size},"blake3":"{}"}}"#,
                blake3::hash(data).to_hex()
            ));
            offset += size as usize;
        }
        let metadata = format!(
            r#"{{"schema":3,"stage":"failed-shared-solve","acceptance":false,"byte_order":"little","dofs":{n},"contacts":{},"rate_encoding":"{}","max_sweeps":{},"local_iterations":{},"staging_bytes":{total},"files":{{{}}}}}"#,
            self.contacts,
            if self.paired_rates { "f32x2" } else { "f32" },
            self.capture_capacity.max_sweeps,
            self.capture_capacity.local_iterations,
            files.join(",")
        );
        write("capture.json", metadata.as_bytes())
    }
}
