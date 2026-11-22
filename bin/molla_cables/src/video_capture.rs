use eframe::{egui, egui_wgpu, wgpu};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{Receiver, SyncSender};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const WIDTH: u32 = 1920;
pub const HEIGHT: u32 = 1080;
pub const FPS: u32 = 30;

/// A submitted frame whose pixels are still being copied into `readback[buffer]`.
struct Pending {
    buffer: usize,
    submission: wgpu::SubmissionIndex,
    mapped: Receiver<std::result::Result<(), wgpu::BufferAsyncError>>,
    still: Option<PathBuf>,
}

/// Thread feeding raw frames to the encoder while the next frame renders.
struct Writer {
    frames: SyncSender<Vec<u8>>,
    thread: std::thread::JoinHandle<std::io::Result<()>>,
}

pub struct Capture {
    pub ctx: egui::Context,
    state: egui_wgpu::RenderState,
    target: wgpu::Texture,
    msaa: wgpu::TextureView,
    depth: wgpu::TextureView,
    readback: [wgpu::Buffer; 2],
    pending: Option<Pending>,
    writer: Option<Writer>,
    encoder: Child,
    temporary: PathBuf,
    destination: PathBuf,
    pub frames: u64,
    finished: bool,
}

impl Capture {
    #[allow(dead_code)]
    pub fn new(destination: &Path) -> Result<Self> {
        Self::new_with_title(destination, "Molla | Wood remembers")
    }

    pub fn new_with_title(destination: &Path, title: &str) -> Result<Self> {
        if destination.exists() {
            return Err(format!("output already exists: {}", destination.display()).into());
        }
        if let Some(parent) = destination.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::default(),
            backend_options: Default::default(),
            memory_budget_thresholds: Default::default(),
            display: None,
        });
        let state = pollster::block_on(egui_wgpu::RenderState::create(
            &Default::default(),
            &instance,
            None,
            egui_wgpu::RendererOptions {
                msaa_samples: 4,
                depth_stencil_format: Some(wgpu::TextureFormat::Depth24Plus),
                ..Default::default()
            },
        ))?;
        eprintln!("Video renderer: {:?}", state.adapter.get_info());
        crate::studio::initialize(&state);
        let size = wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        };
        let texture = |format, samples, usage| {
            state.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("showcase framebuffer"),
                size,
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let target = texture(
            state.target_format,
            1,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let msaa = texture(
            state.target_format,
            4,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        )
        .create_view(&Default::default());
        let depth = texture(
            wgpu::TextureFormat::Depth24Plus,
            4,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        )
        .create_view(&Default::default());
        let readback = [0, 1].map(|_| {
            state.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("showcase readback"),
                size: u64::from(WIDTH * HEIGHT * 4),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        });
        let temporary = destination.with_extension(format!("{}.partial.mp4", std::process::id()));
        let log = std::fs::File::create(destination.with_extension("ffmpeg.log"))?;
        let mut encoder = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-nostdin",
                "-n",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgba",
                "-video_size",
                "1920x1080",
                "-framerate",
                "30",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "17",
                "-pix_fmt",
                "yuv420p",
                "-threads",
                "8",
                "-vf",
                "scale=out_color_matrix=bt709:out_range=tv",
                "-movflags",
                "+faststart",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                "-colorspace",
                "bt709",
                "-color_range",
                "tv",
            ])
            .arg("-metadata")
            .arg(format!("title={title}"))
            .arg(&temporary)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::from(log))
            .spawn()?;
        let mut input = encoder.stdin.take().ok_or("video encoder has no input")?;
        let (frames, queue) = std::sync::mpsc::sync_channel::<Vec<u8>>(2);
        let thread = std::thread::spawn(move || {
            for frame in queue {
                input.write_all(&frame)?;
            }
            Ok(())
        });
        let ctx = egui::Context::default();
        ctx.set_visuals(egui::Visuals::dark());
        Ok(Self {
            ctx,
            state,
            target,
            msaa,
            depth,
            readback,
            pending: None,
            writer: Some(Writer { frames, thread }),
            encoder,
            temporary,
            destination: destination.into(),
            frames: 0,
            finished: false,
        })
    }

    pub fn frame(&mut self, draw: impl FnMut(&mut egui::Ui), still: Option<&Path>) -> Result<()> {
        let output = self.ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(WIDTH as f32, HEIGHT as f32),
                )),
                time: Some(self.frames as f64 / FPS as f64),
                predicted_dt: 1.0 / FPS as f32,
                ..Default::default()
            },
            draw,
        );
        let jobs = self.ctx.tessellate(output.shapes, output.pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [WIDTH, HEIGHT],
            pixels_per_point: output.pixels_per_point,
        };
        let device = &self.state.device;
        let queue = &self.state.queue;
        let mut renderer = self.state.renderer.write();
        for (id, delta) in &output.textures_delta.set {
            renderer.update_texture(device, queue, *id, delta);
        }
        let mut commands = device.create_command_encoder(&Default::default());
        let mut buffers = renderer.update_buffers(device, queue, &mut commands, &jobs, &screen);
        {
            let target = self.target.create_view(&Default::default());
            let mut pass = commands
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("showcase render"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.msaa,
                        resolve_target: Some(&target),
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                })
                .forget_lifetime();
            renderer.render(&mut pass, &jobs, &screen);
        }
        let buffer = (self.frames % 2) as usize;
        commands.copy_texture_to_buffer(
            self.target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback[buffer],
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(WIDTH * 4),
                    rows_per_image: Some(HEIGHT),
                },
            },
            self.target.size(),
        );
        buffers.push(commands.finish());
        let submission = queue.submit(buffers);
        let (tx, mapped) = std::sync::mpsc::sync_channel(1);
        self.readback[buffer]
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        for id in &output.textures_delta.free {
            renderer.free_texture(id);
        }
        drop(renderer);
        let previous = self.pending.replace(Pending {
            buffer,
            submission,
            mapped,
            still: still.map(Path::to_path_buf),
        });
        if let Some(previous) = previous {
            self.deliver(previous)?;
        }
        self.frames += 1;
        Ok(())
    }

    /// Wait for one submitted frame, hand its pixels to the encoder and save its still.
    fn deliver(&mut self, frame: Pending) -> Result<()> {
        self.state.device.poll(wgpu::PollType::Wait {
            submission_index: Some(frame.submission),
            timeout: Some(std::time::Duration::from_secs(30)),
        })?;
        frame
            .mapped
            .recv_timeout(std::time::Duration::from_secs(30))??;
        let buffer = &self.readback[frame.buffer];
        let rgba = buffer.slice(..).get_mapped_range().to_vec();
        buffer.unmap();
        if let Some(path) = &frame.still {
            image::save_buffer(path, &rgba, WIDTH, HEIGHT, image::ColorType::Rgba8)?;
        }
        self.writer
            .as_ref()
            .ok_or("closed video encoder")?
            .frames
            .send(rgba)
            .map_err(|_| "video encoder input closed")?;
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        if let Some(last) = self.pending.take() {
            self.deliver(last)?;
        }
        if let Some(writer) = self.writer.take() {
            drop(writer.frames);
            writer.thread.join().map_err(|_| "video writer panicked")??;
        }
        let status = self.encoder.wait()?;
        if !status.success() || self.frames == 0 {
            return Err(format!("video encoder failed: {status}").into());
        }
        std::fs::hard_link(&self.temporary, &self.destination)?;
        std::fs::remove_file(&self.temporary)?;
        self.finished = true;
        eprintln!(
            "VIDEO {}: {} frames, {:.3} seconds, 1920x1080 at 30 fps",
            self.destination.display(),
            self.frames,
            self.frames as f64 / FPS as f64
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_1080p_rows_satisfy_wgpu_copy_alignment() {
        assert_eq!((WIDTH, HEIGHT, FPS), (1920, 1080, 30));
        assert_eq!((WIDTH * 4) % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT, 0);
    }

    #[test]
    fn existing_movie_is_rejected_without_overwriting() {
        let path =
            std::env::temp_dir().join(format!("molla-preserve-movie-{}.mp4", std::process::id()));
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.write_all(b"existing movie").unwrap();
        assert!(Capture::new(&path).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"existing movie");
        std::fs::remove_file(path).unwrap();
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        if !self.finished {
            self.writer.take();
            let _ = self.encoder.kill();
            let _ = self.encoder.wait();
            let _ = std::fs::remove_file(&self.temporary);
        }
    }
}
