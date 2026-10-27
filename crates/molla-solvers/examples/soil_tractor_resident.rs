use molla_compute::Engine;
use molla_solvers::{
    MpmParticleRenderSource,
    mpm::MpmUnilateralConfig,
    soil_tractor::{
        CHASSIS_HALF_EXTENTS, SoilTractor, SoilTractorConfig, TINE_HALF_EXTENTS, WHEEL_HALF_WIDTH,
        WHEEL_RADIUS,
    },
};
use std::{
    io::Write,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};
use wgpu::util::DeviceExt;

const WIDTH: u32 = 1024;
const HEIGHT: u32 = 768;
const FRAMES: u32 = 33;

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
    body: u32,
}

fn triangle(vertices: &mut Vec<Vertex>, points: [glam::Vec3; 3], color: [f32; 3], body: u32) {
    let normal = (points[1] - points[0])
        .cross(points[2] - points[0])
        .normalize()
        .to_array();
    vertices.extend(points.map(|p| Vertex {
        position: p.to_array(),
        normal,
        color,
        body,
    }));
}

fn box_mesh(vertices: &mut Vec<Vertex>, half: glam::Vec3, color: [f32; 3], body: u32) {
    let p = (0..8)
        .map(|i| {
            half * glam::Vec3::new(
                if i & 1 == 0 { -1.0 } else { 1.0 },
                if i & 2 == 0 { -1.0 } else { 1.0 },
                if i & 4 == 0 { -1.0 } else { 1.0 },
            )
        })
        .collect::<Vec<_>>();
    for [a, b, c, d] in [
        [0, 2, 3, 1],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 6, 7, 3],
        [0, 4, 6, 2],
        [1, 3, 7, 5],
    ] {
        triangle(vertices, [p[a], p[b], p[c]], color, body);
        triangle(vertices, [p[a], p[c], p[d]], color, body);
    }
}

fn geometry(tractor: &SoilTractor) -> Vec<Vertex> {
    let scene = &tractor.world().scene;
    let handles = tractor.handles();
    let index = |h| scene.body_index(h).unwrap().index() as u32;
    let mut out = Vec::new();
    box_mesh(
        &mut out,
        CHASSIS_HALF_EXTENTS.as_vec3(),
        [0.15, 0.6, 0.22],
        index(handles.chassis),
    );
    box_mesh(
        &mut out,
        TINE_HALF_EXTENTS.as_vec3(),
        [0.9, 0.16, 0.08],
        index(handles.tine),
    );
    for wheel in handles.wheels {
        for i in 0..32 {
            let angle = i as f32 * std::f32::consts::TAU / 32.0;
            let next = (i + 1) as f32 * std::f32::consts::TAU / 32.0;
            let radial = |a: f32, y: f32| {
                glam::Vec3::new(
                    WHEEL_RADIUS as f32 * a.cos(),
                    y,
                    WHEEL_RADIUS as f32 * a.sin(),
                )
            };
            let y = WHEEL_HALF_WIDTH as f32;
            let [a, b, c, d] = [
                radial(angle, -y),
                radial(next, -y),
                radial(next, y),
                radial(angle, y),
            ];
            for points in [
                [a, b, c],
                [a, c, d],
                [glam::Vec3::new(0.0, -y, 0.0), b, a],
                [glam::Vec3::new(0.0, y, 0.0), d, c],
            ] {
                triangle(&mut out, points, [0.06, 0.07, 0.08], index(wheel));
            }
        }
    }
    out
}

struct Renderer {
    particle: wgpu::RenderPipeline,
    machine: wgpu::RenderPipeline,
    groups: [wgpu::BindGroup; 2],
    vertices: wgpu::Buffer,
    vertex_count: u32,
    source: MpmParticleRenderSource,
    body_sources: [wgpu::Buffer; 2],
    color: wgpu::Texture,
    _depth: wgpu::Texture,
    color_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    images: [wgpu::Buffer; 2],
    queries: wgpu::QuerySet,
    query_resolve: wgpu::Buffer,
    query_staging: wgpu::Buffer,
}

impl Renderer {
    fn new(engine: &Engine, tractor: &SoilTractor) -> Self {
        let device = engine.device().unwrap();
        let (_, view) = tractor.soil_particle_render_view().unwrap();
        let bodies = tractor.world().gpu_body_render_view().unwrap();
        let body_sources = bodies.buffers().map(Clone::clone);
        let storage = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("resident tractor"),
            entries: &[
                storage(0),
                storage(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let camera = glam::Mat4::perspective_rh(
            55f32.to_radians(),
            WIDTH as f32 / HEIGHT as f32,
            0.05,
            50.0,
        ) * glam::Mat4::look_at_rh(
            glam::Vec3::new(5.8, 3.8, 6.5),
            glam::Vec3::new(0.0, 0.15, 0.0),
            glam::Vec3::Y,
        );
        let camera = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("static camera"),
            contents: bytemuck::cast_slice(&camera.to_cols_array()),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let groups = std::array::from_fn(|side| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: view.positions().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: body_sources[side].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: camera.as_entire_binding(),
                    },
                ],
            })
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("resident tractor render"),
            source: wgpu::ShaderSource::Wgsl(include_str!("soil_tractor_resident.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let attributes = wgpu::vertex_attr_array![0=>Float32x3,1=>Float32x3,2=>Float32x3,3=>Uint32];
        let vertex_layout = [wgpu::VertexBufferLayout {
            array_stride: 40,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &attributes,
        }];
        let pipeline = |entry, mesh| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    buffers: if mesh { &vertex_layout } else { &[] },
                },
                primitive: Default::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fragment"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8UnormSrgb,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let mesh = geometry(tractor);
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("analytic fixture meshes"),
            contents: bytemuck::cast_slice(&mesh),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let texture = |format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: Self::size(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let buffer = |size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let color = texture(
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let depth = texture(
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let color_view = color.create_view(&Default::default());
        let depth_view = depth.create_view(&Default::default());
        Self {
            particle: pipeline("particle", false),
            machine: pipeline("machine", true),
            groups,
            vertices,
            vertex_count: mesh.len() as u32,
            source: view.source(),
            body_sources,
            color,
            _depth: depth,
            color_view,
            depth_view,
            images: std::array::from_fn(|_| {
                buffer(
                    u64::from(WIDTH * HEIGHT * 4),
                    wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                )
            }),
            queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: None,
                ty: wgpu::QueryType::Timestamp,
                count: 2 * (FRAMES + 1),
            }),
            query_resolve: buffer(
                u64::from(16 * (FRAMES + 1)),
                wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            ),
            query_staging: buffer(
                u64::from(16 * (FRAMES + 1)),
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            ),
        }
    }
    fn size() -> wgpu::Extent3d {
        wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        }
    }
    fn draw(&self, engine: &Engine, tractor: &SoilTractor, frame: u32) {
        let (committed, view) = tractor.soil_particle_render_view().unwrap();
        assert_eq!(committed, u64::from(frame));
        assert_eq!(view.particle_count(), 50_000);
        assert!(view.matches_source(&self.source));
        let bodies = tractor.world().gpu_body_render_view().unwrap();
        assert_eq!(bodies.committed_frame(), committed);
        assert_eq!(bodies.body_count(), 6);
        assert_eq!(
            bodies.buffers(),
            [&self.body_sources[0], &self.body_sources[1]]
        );
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("resident tractor frame"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.10,
                            g: 0.13,
                            b: 0.17,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: Some(wgpu::RenderPassTimestampWrites {
                    query_set: &self.queries,
                    beginning_of_pass_write_index: Some(frame * 2),
                    end_of_pass_write_index: Some(frame * 2 + 1),
                }),
                ..Default::default()
            });
            pass.set_bind_group(0, &self.groups[bodies.active_side()], &[]);
            pass.set_pipeline(&self.particle);
            pass.draw(0..12, 0..view.particle_count());
            pass.set_pipeline(&self.machine);
            pass.set_vertex_buffer(0, self.vertices.slice(..));
            pass.draw(0..self.vertex_count, 0..1);
        }
        if frame == 0 || frame == FRAMES {
            encoder.copy_texture_to_buffer(
                self.color.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.images[usize::from(frame != 0)],
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(WIDTH * 4),
                        rows_per_image: Some(HEIGHT),
                    },
                },
                Self::size(),
            );
        }
        if frame == FRAMES {
            encoder.resolve_query_set(&self.queries, 0..2 * (FRAMES + 1), &self.query_resolve, 0);
            encoder.copy_buffer_to_buffer(
                &self.query_resolve,
                0,
                &self.query_staging,
                0,
                self.query_staging.size(),
            );
        }
        engine.queue().unwrap().submit([encoder.finish()]);
    }
}

fn mapped(engine: &Engine, buffer: &wgpu::Buffer) -> Vec<u8> {
    let (tx, rx) = std::sync::mpsc::channel();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    engine
        .device()
        .unwrap()
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(15)),
        })
        .unwrap();
    rx.recv_timeout(Duration::from_secs(1)).unwrap().unwrap();
    let bytes = buffer.slice(..).get_mapped_range().to_vec();
    buffer.unmap();
    bytes
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    if !cfg!(feature = "norm-tree-newton") {
        return Err("requires norm-tree-newton".into());
    }
    if molla_solvers::soil_newton_gpu::diagnostic_inner_limit()?.is_some() {
        return Err("diagnostic inner limits are not allowed for resident capture".into());
    }
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    if args.len() != 1 {
        return Err("usage: soil_tractor_resident NEW_OUTPUT_DIRECTORY".into());
    }
    let output = Path::new(&args[0]);
    std::fs::create_dir(output)?;
    let engine = Arc::new(Engine::new_gpu_with_features(
        wgpu::Features::SHADER_FLOAT32_ATOMIC
            | wgpu::Features::PASSTHROUGH_SHADERS
            | wgpu::Features::TIMESTAMP_QUERY,
    )?);
    if engine.adapter_info().unwrap().is_software {
        return Err("hardware GPU required".into());
    }
    let started = Instant::now();
    eprintln!("resident tractor: constructing 50k coupled workload");
    let mut tractor = SoilTractor::new_native_explicit_with_paired_newton(
        engine.clone(),
        SoilTractorConfig::default(),
        MpmUnilateralConfig {
            node_capacity: 4096,
            search_distance: 0.05,
            max_iterations: 64,
        },
        0.3,
    )?;
    let construction_ms = started.elapsed().as_secs_f64() * 1000.0;
    let setup = Instant::now();
    let renderer = Renderer::new(&engine, &tractor);
    let render_setup_ms = setup.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "resident tractor: physics setup {construction_ms:.3} ms; renderer setup {render_setup_ms:.3} ms"
    );
    let mut frames = Vec::new();
    renderer.draw(&engine, &tractor, 0);
    for frame in 1..=FRAMES {
        let start = Instant::now();
        tractor.step(1.0 / 60.0, [300.0; 4])?;
        let physics_ms = start.elapsed().as_secs_f64() * 1000.0;
        renderer.draw(&engine, &tractor, frame);
        frames.push(serde_json::json!({"committed_frame":frame,"simulated_seconds":f64::from(frame)/60.0,"warmup":frame<=3,"physics_wall_ms":physics_ms}));
        eprintln!("resident tractor: committed frame {frame}/{FRAMES}; step {physics_ms:.3} ms");
    }
    for (name, image) in ["initial", "final"].into_iter().zip(&renderer.images) {
        let rgba = mapped(&engine, image);
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output.join(format!("{name}.ppm")))?;
        write!(file, "P6\n{WIDTH} {HEIGHT}\n255\n")?;
        for p in rgba.chunks_exact(4) {
            file.write_all(&p[..3])?;
        }
    }
    let query = mapped(&engine, &renderer.query_staging);
    let ticks = query
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    let render_ms = ticks
        .chunks_exact(2)
        .map(|pair| {
            pair[1].checked_sub(pair[0]).unwrap() as f64
                * engine.queue().unwrap().get_timestamp_period() as f64
                / 1e6
        })
        .collect::<Vec<_>>();
    if render_ms.len() != (FRAMES + 1) as usize
        || render_ms.iter().any(|ms| !ms.is_finite() || *ms <= 0.0)
    {
        return Err("invalid render timestamp samples".into());
    }
    let method = if cfg!(feature = "stream-block-newton") {
        26
    } else if cfg!(feature = "shared-conditioned-core-newton") {
        25
    } else if cfg!(feature = "conditioned-projection-cache-newton") {
        24
    } else if cfg!(feature = "retained-local-newton") {
        23
    } else if cfg!(feature = "publication-products-newton") {
        22
    } else if cfg!(feature = "active-publication-newton") {
        21
    } else {
        20
    };
    let report = serde_json::json!({"acceptance":false,"physical_calibration":"uncalibrated","adapter":engine.adapter_info().unwrap().name,"particles":50_000,"frames":frames,"render_gpu_ms":render_ms,"construction_ms":construction_ms,"render_setup_ms":render_setup_ms,"render_scope":"single pass: 50k display tetrahedra plus six analytic fixture meshes; direct live particle and body-pose buffers; no CPU particle/body readback or pose upload for rendering","physics_timing_scope":"blocking tractor step; may include waiting for previous queued rendering; not isolated physics cost","observation_scope":"two RGBA images and one timestamp array mapped after all frames; not particle observations","renderer":"same-device wgpu 29 offscreen; not Bevy frontend","width":WIDTH,"height":HEIGHT,"display_radius_m":0.012,"method":method,"machine_execution":"generic","camera_eye":[5.8,3.8,6.5],"camera_target":[0.0,0.15,0.0]});
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!(
        "resident tractor: {FRAMES} committed frames; GPU rendering samples={render_ms:?}; acceptance=false"
    );
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("resident tractor: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renderer_shader_and_host_layouts_match() {
        assert_eq!(std::mem::size_of::<Vertex>(), 40);
        assert_eq!(std::mem::offset_of!(Vertex, body), 36);
        assert_eq!(std::mem::size_of::<molla_math::gpu::GpuTransform>(), 32);
        let module =
            naga::front::wgsl::parse_str(include_str!("soil_tractor_resident.wgsl")).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        let body = module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some("Body"))
            .unwrap()
            .1;
        match &body.inner {
            naga::TypeInner::Struct { members, span } => {
                assert_eq!(*span, 32);
                assert_eq!(
                    members.iter().map(|m| m.offset).collect::<Vec<_>>(),
                    [0, 16]
                );
            }
            _ => panic!("body layout"),
        }
        assert_eq!(module.entry_points.len(), 3);
        assert_eq!(WIDTH * 4 % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT, 0);
        assert_eq!(2 * (FRAMES + 1), 68);
    }

    #[test]
    fn box_geometry_has_finite_normals_and_exact_local_bounds() {
        let half = glam::Vec3::new(0.95, 0.22, 0.42);
        let mut vertices = Vec::new();
        box_mesh(&mut vertices, half, [0.1, 0.2, 0.3], 7);
        assert_eq!(vertices.len(), 36);
        for vertex in vertices {
            assert_eq!(vertex.body, 7);
            assert_eq!(glam::Vec3::from(vertex.position).abs(), half);
            assert!((glam::Vec3::from(vertex.normal).length() - 1.0).abs() < 1e-6);
        }
    }
}
