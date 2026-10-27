#![cfg(all(feature = "gpu", feature = "resource-counters"))]

use molla_compute::Engine;
use molla_core::gpu_metrics::{DeviceScope, DispatchScope, TransferScope};
use molla_solvers::{MpmConfig, MpmSimulation};
use std::{io::Write, time::Duration};

const SIDE: u32 = 512;

struct Renderer {
    source: molla_solvers::MpmParticleRenderSource,
    pipeline: wgpu::RenderPipeline,
    group: wgpu::BindGroup,
    texture: wgpu::Texture,
    staging: wgpu::Buffer,
}

impl Renderer {
    fn new(engine: &Engine, simulation: &MpmSimulation) -> Self {
        let device = engine.device().unwrap();
        let positions = simulation.particle_render_view(engine).unwrap();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MPM resident particle vertex pulling"),
            source: wgpu::ShaderSource::Wgsl(r#"
@group(0) @binding(0) var<storage,read> positions:array<vec4<f32>>;
@vertex fn vertex(@builtin(vertex_index) vertex:u32,@builtin(instance_index) particle:u32)->@builtin(position) vec4<f32> {
    let corners=array<vec2<f32>,6>(vec2<f32>(-1.0,-1.0),vec2<f32>(1.0,-1.0),vec2<f32>(1.0,1.0),vec2<f32>(-1.0,-1.0),vec2<f32>(1.0,1.0),vec2<f32>(-1.0,1.0));
    return vec4<f32>(positions[particle].xy+corners[vertex]*0.02,0.5,1.0);
}
@fragment fn fragment()->@location(0) vec4<f32> {return vec4<f32>(0.7,0.4,0.15,1.0);}
"#.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("resident MPM renderer"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("live MPM position binding"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: positions.positions().as_entire_binding(),
            }],
        });
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("resident MPM image"),
            size: Self::size(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("image-only readback"),
            size: u64::from(SIDE * SIDE * 4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            source: positions.source(),
            pipeline,
            group,
            texture,
            staging,
        }
    }

    fn size() -> wgpu::Extent3d {
        wgpu::Extent3d {
            width: SIDE,
            height: SIDE,
            depth_or_array_layers: 1,
        }
    }

    fn draw(&self, engine: &Engine, simulation: &MpmSimulation, label: &str) -> Vec<u8> {
        let transfer = TransferScope::start().unwrap();
        let allocations = DeviceScope::start().unwrap();
        let dispatches = DispatchScope::start().unwrap();
        let input = simulation.particle_render_view(engine).unwrap();
        assert!(
            input.matches_source(&self.source),
            "render binding belongs to another simulation"
        );
        assert_eq!(
            input.live_position_bytes(),
            u64::from(input.particle_count()) * 16
        );
        let traffic = transfer.finish();
        assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
        assert_eq!(allocations.finish().observations, 0);
        assert_eq!(dispatches.finish().encoded_dispatches, 0);
        let device = engine.device().unwrap();
        let view = self.texture.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.group, &[]);
            pass.draw(0..6, 0..input.particle_count());
        }
        encoder.copy_texture_to_buffer(
            self.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &self.staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(SIDE * 4),
                    rows_per_image: Some(SIDE),
                },
            },
            Self::size(),
        );
        engine.queue().unwrap().submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        self.staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(15)),
            })
            .unwrap();
        rx.recv_timeout(Duration::from_secs(1)).unwrap().unwrap();
        let pixels = self.staging.slice(..).get_mapped_range().to_vec();
        self.staging.unmap();
        if let Some(root) = std::env::var_os("MOLLA_PARTICLE_RENDER_ARTIFACTS") {
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(std::path::Path::new(&root).join(format!("{label}.ppm")))
                .unwrap();
            write!(file, "P6\n{SIDE} {SIDE}\n255\n").unwrap();
            for pixel in pixels.chunks_exact(4) {
                file.write_all(&pixel[..3]).unwrap();
            }
        }
        pixels
    }
}

fn centroid(pixels: &[u8]) -> (f64, usize) {
    let (mut sum, mut count) = (0usize, 0usize);
    for (i, pixel) in pixels.chunks_exact(4).enumerate() {
        if pixel[0] > 100 {
            sum += i % SIDE as usize;
            count += 1;
        }
    }
    (sum as f64 / count.max(1) as f64, count)
}

#[test]
#[ignore = "requires GPU; rendered resident MPM buffer and lifecycle"]
fn mpm_resident_particle_rendering_and_lifecycle() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 16;
    cfg.grid_origin = [-1.0; 3];
    cfg.cell_size = 0.125;
    cfg.max_particles = 50_000;
    cfg.gravity_y = 0.0;
    let mut simulation = MpmSimulation::new(&engine, cfg).unwrap();
    simulation
        .set_particles_with_velocity(
            engine.queue().unwrap(),
            &[[-0.4, -0.2, 0.0], [0.4, 0.2, 0.0]],
            &[[0.5, 0.0, 0.0]; 2],
        )
        .unwrap();
    let renderer = Renderer::new(&engine, &simulation);
    let before = renderer.draw(&engine, &simulation, "before");
    simulation.step_n(&engine, 0.0005, 128).unwrap();
    let after = renderer.draw(&engine, &simulation, "after");
    let (a, n) = centroid(&before);
    let (b, m) = centroid(&after);
    assert!(
        n > 100 && m > 100 && b > a + 5.0,
        "rendered movement {a}->{b}, pixels {n}/{m}"
    );
    assert!(simulation.particle_render_view(&Engine::new_cpu()).is_err());
    let foreign = Engine::new_gpu().unwrap();
    assert!(simulation.particle_render_view(&foreign).is_err());
    let mut replacement_config = cfg;
    replacement_config.max_particles = 1;
    let replacement = MpmSimulation::new(&engine, replacement_config).unwrap();
    assert!(
        !replacement
            .particle_render_view(&engine)
            .unwrap()
            .matches_source(&renderer.source)
    );
    simulation
        .set_particles(engine.queue().unwrap(), &[])
        .unwrap();
    assert_eq!(centroid(&renderer.draw(&engine, &simulation, "empty")).1, 0);
    let positions: Vec<_> = (0..50_000)
        .map(|i| {
            [
                ((i % 100) as f32 - 49.5) / 64.0,
                (((i / 100) % 100) as f32 - 49.5) / 64.0,
                (i / 10_000) as f32 * 0.01,
            ]
        })
        .collect();
    simulation
        .set_particles(engine.queue().unwrap(), &positions)
        .unwrap();
    assert_eq!(
        simulation
            .particle_render_view(&engine)
            .unwrap()
            .particle_count(),
        50_000
    );
    let full = renderer.draw(&engine, &simulation, "50k");
    assert!(centroid(&full).1 > 150_000);
    eprintln!(
        "resident render: two particles advance {a}->{b} pixels; empty clears; 50k reseed uses same render binding; only RGBA images mapped"
    );
}
