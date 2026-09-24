use molla_compute::Engine;
use molla_kernels::soft_rigid_solve::SoftRigidSolveKernel;
use std::path::Path;
use wgpu::util::DeviceExt;

pub struct LocalCoupledPatch {
    groups: Vec<wgpu::BindGroup>,
    pipelines: Vec<wgpu::ComputePipeline>,
    pub input: wgpu::Buffer,
    pub output: wgpu::Buffer,
    pub exclusive: bool,
    pub automatic: bool,
}

impl LocalCoupledPatch {
    pub fn new(
        engine: &Engine,
        directory: &Path,
        material: &[Vec<wgpu::Buffer>],
        contact: &[wgpu::Buffer],
        kernel: &SoftRigidSolveKernel,
        edge: Option<usize>,
    ) -> Self {
        let device = engine.device().unwrap();
        let (nodes, selected) = if let Some(edge) = edge {
            let bytes = std::fs::read(directory.join("contact-contacts.bin")).unwrap();
            let rows: Vec<[u32; 20]> = bytes
                .chunks_exact(80)
                .map(bytemuck::pod_read_unaligned)
                .collect();
            assert_eq!(rows[edge][11], 1);
            assert_eq!(rows[edge][14], 2);
            let nodes = [rows[edge][8], rows[edge][12]];
            assert_ne!(nodes[0], nodes[1]);
            let mut selected = [edge; 3];
            for (i, node) in nodes.iter().enumerate() {
                let matches: Vec<_> = rows
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| {
                        r[11] == 1 && r[14] == 1 && r[8] == *node && r[9..11] == rows[edge][9..11]
                    })
                    .map(|(i, _)| i)
                    .collect();
                assert_eq!(matches.len(), 1);
                selected[i] = matches[0];
            }
            for &row in &selected {
                assert_eq!(rows[row][..3], rows[edge][..3]);
            }
            for (index, row) in rows.iter().enumerate() {
                if row[11] != 1
                    || row[14] != 2
                    || row[9..11] != rows[edge][9..11]
                    || selected.contains(&index)
                {
                    continue;
                }
                let support = [row[8], row[12], row[13]];
                assert!(
                    !support[..row[14] as usize]
                        .iter()
                        .any(|node| nodes.contains(node))
                );
            }
            (nodes, selected)
        } else {
            ([0; 2], [0; 3])
        };
        let exclusive = std::env::var_os("MOLLA_COUPLED_LOCAL_EXCLUSIVE").is_some();
        let mut metadata = vec![0.0_f32; 207];
        metadata[0] = nodes[0] as f32;
        metadata[1] = nodes[1] as f32;
        for i in 0..3 {
            metadata[187 + i] = selected[i] as f32;
        }
        let relaxation = std::env::var("MOLLA_COUPLED_LOCAL_RELAXATION")
            .map(|value| value.parse::<f32>().unwrap())
            .unwrap_or(1.0);
        assert!(relaxation.is_finite() && relaxation > 0.0 && relaxation <= 1.0);
        metadata[190] = relaxation;
        metadata[191] = if edge.is_some() { 1.0 } else { 0.0 };
        let storage = |label, data: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
        };
        let input = storage("live local patch metadata", bytemuck::cast_slice(&metadata));
        let output = storage(
            "live local patch solution",
            bytemuck::cast_slice(&[0.0_f32; 36]),
        );
        let counts = [7, 9, 7, 1, 5, 17, 6];
        let layouts: Vec<_> = counts
            .iter()
            .enumerate()
            .map(|(g, &count)| {
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &(0..count)
                        .map(|b| wgpu::BindGroupLayoutEntry {
                            binding: b,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: if g == 3 {
                                    wgpu::BufferBindingType::Uniform
                                } else {
                                    wgpu::BufferBindingType::Storage {
                                        read_only: !(g == 2
                                            || (g == 5 && [10, 11, 14].contains(&b))
                                            || (g == 6 && [0, 3, 4].contains(&b))),
                                    }
                                },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        })
                        .collect::<Vec<_>>(),
                })
            })
            .collect();
        let mut resources: Vec<Vec<&wgpu::Buffer>> =
            material.iter().map(|v| v.iter().collect()).collect();
        resources[2][5] = &input;
        resources[2][6] = &output;
        resources.push(vec![
            kernel.row_state(),
            kernel.projected_rows(),
            kernel.row_responses(),
            &contact[8],
            &contact[3],
            kernel.active_rows(),
        ]);
        let groups = resources
            .iter()
            .enumerate()
            .map(|(g, buffers)| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &layouts[g],
                    entries: &buffers
                        .iter()
                        .enumerate()
                        .map(|(b, buffer)| wgpu::BindGroupEntry {
                            binding: b as u32,
                            resource: buffer.as_entire_binding(),
                        })
                        .collect::<Vec<_>>(),
                })
            })
            .collect();
        let source = format!(
            "{}\n{}\n{}\n{}",
            include_str!("../../../molla-kernels/shaders/tet_viscosity.wgsl"),
            include_str!("../../../molla-kernels/shaders/vbd_cloth.wgsl"),
            include_str!("../local_coupled_probe.wgsl"),
            include_str!("../local_coupled_apply.wgsl")
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("refreshed local coupled patch"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &layouts.iter().map(Some).collect::<Vec<_>>(),
            immediate_size: 0,
        });
        let pipelines = [
            "patch_refresh",
            "patch_live_solve",
            "patch_apply",
            "patch_disable_split_contacts",
            "patch_restore_contacts",
            "patch_polish_neighbors",
            "patch_select",
        ]
        .iter()
        .map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        })
        .collect();
        println!(
            "COUPLED initial_patch_nodes={nodes:?} rows={selected:?} automatic={} relaxation={relaxation} exclusive={exclusive}; GPU-prepared response and refreshed state",
            edge.is_none()
        );
        Self {
            groups,
            pipelines,
            input,
            output,
            exclusive,
            automatic: edge.is_none(),
        }
    }

    pub fn select(&self, encoder: &mut wgpu::CommandEncoder) {
        if !self.automatic {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipelines[6]);
        for (i, group) in self.groups.iter().enumerate() {
            pass.set_bind_group(i as u32, group, &[]);
        }
        pass.dispatch_workgroups(1, 1, 1);
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        for pipeline in &self.pipelines[..3] {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            for (i, group) in self.groups.iter().enumerate() {
                pass.set_bind_group(i as u32, group, &[]);
            }
            pass.dispatch_workgroups(1, 1, 1);
        }
    }

    pub fn split_contacts(&self, encoder: &mut wgpu::CommandEncoder, enabled: bool) {
        if !self.exclusive {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipelines[if enabled { 4 } else { 3 }]);
        for (i, group) in self.groups.iter().enumerate() {
            pass.set_bind_group(i as u32, group, &[]);
        }
        pass.dispatch_workgroups(1, 1, 1);
    }

    pub fn material_neighbors(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        uniforms: &[wgpu::BindGroup],
        offsets: &[u32],
        sweeps: u32,
    ) {
        assert!(self.exclusive);
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipelines[5]);
        for (i, group) in self.groups.iter().enumerate() {
            pass.set_bind_group(i as u32, group, &[]);
        }
        for _ in 0..sweeps {
            for (color, group) in uniforms.iter().enumerate() {
                pass.set_bind_group(3, group, &[]);
                pass.dispatch_workgroups((offsets[color + 1] - offsets[color]).div_ceil(64), 1, 1);
            }
        }
    }
}
