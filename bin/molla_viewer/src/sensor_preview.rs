use super::sensor_timing::SensorGpuTimer;
use super::{Engine, FIXED_DT, SUBSTEPS_PER_FRAME, Scene, SensorCommandBatch, egui, readout};
use molla::core::{Error, Result};
use molla::sensors::readback::{
    ReadbackConfig, ReadbackFrame, ReadbackRing, ReadbackSource, ReadbackStamp, ReadbackTicket,
};
use molla::sensors::{SensorFramePool, SensorFramePoolConfig};
use std::{collections::VecDeque, time::Instant};

pub(super) const PREVIEW_SLOTS: usize = 3;

#[derive(Clone, Copy, Default)]
pub(super) struct ViewerSensorClock {
    pub revision: u64,
    pub tick: u64,
}

impl ViewerSensorClock {
    pub fn reset(&mut self) {
        self.revision = self
            .revision
            .checked_add(1)
            .expect("sensor revision exhausted");
        self.tick = 0;
    }

    pub fn advance(&mut self) {
        self.tick = self.tick.checked_add(1).expect("sensor clock exhausted");
    }

    pub fn time(self) -> f64 {
        self.tick as f64 * FIXED_DT
    }

    pub fn stamp(self, sensor_id: u64) -> ReadbackStamp {
        ReadbackStamp {
            sensor_id,
            sample_id: self.tick,
            scene_revision: self.revision,
            simulation_time: Some(self.time()),
        }
    }
}

pub(super) fn preview_pool_config(bytes_per_frame: u64) -> Result<SensorFramePoolConfig> {
    Ok(SensorFramePoolConfig {
        slots: PREVIEW_SLOTS,
        max_total_bytes: bytes_per_frame
            .checked_mul(PREVIEW_SLOTS as u64)
            .ok_or_else(|| Error::Storage("sensor preview budget overflow".into()))?,
    })
}

pub(super) struct SensorPreview {
    pub timer: Option<SensorGpuTimer>,
    pub pool: SensorFramePool,
    pub readback: ReadbackRing,
    pending: VecDeque<ReadbackTicket>,
    attempted: Option<u64>,
    pub submitted_tick: Option<u64>,
    pub submitted: u64,
    pub deferred: u64,
    pub collected: u64,
    pub latest: Option<ReadbackStamp>,
    pub now: f64,
    pub cpu_submit_us: f64,
    pub cadence_hz: Option<f64>,
}

impl SensorPreview {
    pub fn new(engine: &Engine, pool: SensorFramePool, bytes_per_frame: u64) -> Result<Self> {
        let readback = ReadbackRing::new(
            engine,
            ReadbackConfig {
                slots: PREVIEW_SLOTS,
                bytes_per_slot: bytes_per_frame,
                max_total_bytes: preview_pool_config(bytes_per_frame)?.max_total_bytes,
            },
        )?;
        Ok(Self {
            timer: SensorGpuTimer::new(engine)?,
            pool,
            readback,
            pending: VecDeque::with_capacity(PREVIEW_SLOTS),
            attempted: None,
            submitted_tick: None,
            submitted: 0,
            deferred: 0,
            collected: 0,
            latest: None,
            now: 0.0,
            cpu_submit_us: 0.0,
            cadence_hz: Some(1.0 / (SUBSTEPS_PER_FRAME as f64 * FIXED_DT)),
        })
    }

    pub fn poll_latest(&mut self, clock: ViewerSensorClock) -> Result<Option<ReadbackFrame>> {
        self.now = clock.time();
        if let Some(timer) = &mut self.timer {
            timer.poll(clock.revision)?;
        }
        let mut latest = None;
        while let Some(ticket) = self.pending.front() {
            let Some(frame) = self.readback.poll(ticket)? else {
                break;
            };
            self.pending.pop_front();
            if frame.stamp.scene_revision == clock.revision {
                self.latest = Some(frame.stamp);
                self.collected += 1;
                latest = Some(frame);
            }
        }
        Ok(latest)
    }

    pub fn begin_sample(&mut self, tick: u64) -> Result<bool> {
        if self.attempted == Some(tick) {
            return Ok(false);
        }
        self.attempted = Some(tick);
        if self.pending.len() == PREVIEW_SLOTS
            || self.readback.available_slots() == 0
            || self.pool.available_slots()? == 0
        {
            self.deferred += 1;
            return Ok(false);
        }
        Ok(true)
    }

    pub fn sample_dt(&self, tick: u64) -> f64 {
        self.submitted_tick
            .map_or(SUBSTEPS_PER_FRAME as f64 * FIXED_DT, |previous| {
                (tick - previous) as f64 * FIXED_DT
            })
    }

    pub fn begin_request(&mut self, id: u64) -> Result<bool> {
        if self.attempted == Some(id) {
            return Ok(false);
        }
        if self.pending.len() == PREVIEW_SLOTS
            || self.readback.available_slots() == 0
            || self.pool.available_slots()? == 0
        {
            self.deferred += 1;
            return Ok(false);
        }
        self.attempted = Some(id);
        Ok(true)
    }

    pub fn submitted(&mut self, tick: u64, start: Instant) {
        self.submitted_tick = Some(tick);
        self.submitted += 1;
        self.cpu_submit_us = start.elapsed().as_secs_f64() * 1e6;
    }

    pub fn begin_gpu(&mut self, batch: &mut SensorCommandBatch, engine: &Engine) -> Result<bool> {
        if let Some(timer) = &mut self.timer
            && timer.available()
        {
            timer.begin(batch, engine)?;
            return Ok(true);
        }
        Ok(false)
    }

    pub fn end_gpu(
        &self,
        batch: &mut SensorCommandBatch,
        engine: &Engine,
        timed: bool,
    ) -> Result<()> {
        if timed {
            self.timer
                .as_ref()
                .expect("enabled timer")
                .end(batch, engine)?;
        }
        Ok(())
    }

    pub fn collect_gpu(&mut self, stamp: ReadbackStamp, timed: bool) -> Result<()> {
        if timed {
            self.timer.as_mut().expect("enabled timer").enqueue(stamp)?;
        }
        Ok(())
    }

    pub fn enqueue(
        &mut self,
        sources: &[ReadbackSource<'_>],
        stamp: ReadbackStamp,
    ) -> Result<bool> {
        if let Some(ticket) = self.readback.try_enqueue(sources, stamp)? {
            self.pending.push_back(ticket);
            Ok(true)
        } else {
            self.deferred += 1;
            Ok(false)
        }
    }
}

pub(super) fn sensor_preview_scroll(
    ui: &mut egui::Ui,
    scene: Scene,
    body: impl FnOnce(&mut egui::Ui),
) {
    if scene.sensor_only()
        || matches!(
            scene,
            Scene::Lidar | Scene::Imu | Scene::Contact | Scene::Camera
        )
    {
        let height = (ui.ctx().content_rect().bottom() - ui.cursor().top() - 16.0).max(64.0);
        egui::ScrollArea::vertical()
            .id_salt(("sensor_preview_scroll", scene as usize))
            .max_height(height)
            .auto_shrink([false, true])
            .show(ui, |ui| {
                egui::Frame::new().inner_margin(12).show(ui, body);
            });
    } else {
        body(ui);
    }
}

pub(super) fn sensor_preview_pane(
    ui: &mut egui::Ui,
    preview: Option<&SensorPreview>,
    error: Option<&str>,
) {
    if let Some(error) = error {
        ui.colored_label(egui::Color32::LIGHT_RED, error);
        ui.small("Reset or switch scenes to retry.");
    }
    if let Some(preview) = preview {
        let cadence = preview.cadence_hz.map_or_else(
            || "once per scene/reset".into(),
            |hz| format!("{hz:.1} Hz (sim)"),
        );
        readout(ui, "Sample cadence", &cadence);
        readout(
            ui,
            "Submitted / collected",
            &format!("{} / {}", preview.submitted, preview.collected),
        );
        readout(
            ui,
            "Pending / deferred",
            &format!("{} / {}", preview.pending.len(), preview.deferred),
        );
        if let Some(stamp) = preview.latest {
            let sampled = stamp.simulation_time.unwrap_or(0.0);
            readout(
                ui,
                "Sample / age (sim)",
                &format!("{sampled:.3} / {:.3} s", (preview.now - sampled).max(0.0)),
            );
        }
        readout(
            ui,
            "CPU submit",
            &format!("{:.1} µs", preview.cpu_submit_us),
        );
        if let Some(timer) = &preview.timer {
            if let Some((stamp, us)) = timer.latest {
                readout(
                    ui,
                    "GPU batch",
                    &format!("{us:.1} µs · sample {}", stamp.sample_id),
                );
                readout(
                    ui,
                    "GPU timing age (sim)",
                    &format!(
                        "{:.3} s",
                        (preview.now - stamp.simulation_time.unwrap_or(0.0)).max(0.0)
                    ),
                );
            } else {
                readout(ui, "GPU batch", "waiting for timestamps");
            }
            readout(ui, "Deferred GPU timings", &timer.deferred.to_string());
        } else {
            readout(ui, "GPU batch", "not measured");
            ui.small("Optional timestamps: MOLLA_SENSOR_TIMING=1 (supported devices).");
        }
        ui.small("Async display with bounded pending frames.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sensor_checkbox_remains_clickable_beside_pane_resize_handle() {
        let ctx = egui::Context::default();
        let mut checked = false;
        let mut rect = egui::Rect::NOTHING;
        let mut render = |events| {
            let _ = ctx.run(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(800.0, 600.0),
                    )),
                    events,
                    ..Default::default()
                },
                |ctx| {
                    crate::Pane2::new(
                        "sensor_resize_checkbox",
                        "Sensor",
                        crate::PaneAnchor::RightRail(crate::RailZone::Start),
                        egui::Color32::LIGHT_BLUE,
                    )
                    .resize(crate::PaneResize::BOTH)
                    .show(ctx, |ui| {
                        sensor_preview_scroll(ui, Scene::Camera, |ui| {
                            rect = ui.checkbox(&mut checked, "Physical optics").rect;
                            for _ in 0..30 {
                                ui.label("Sensor metric");
                            }
                        });
                    });
                },
            );
            rect
        };
        render(vec![]);
        let rect = render(vec![]);
        let pos = egui::pos2(rect.left() + 4.0, rect.center().y);
        for pressed in [true, false] {
            render(vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: egui::Modifiers::NONE,
                },
            ]);
        }
        assert!(checked);
    }

    #[test]
    fn sensor_pane_scroll_stays_inside_viewport() {
        let ctx = egui::Context::default();
        for height in [600.0, 1000.0] {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(800.0, height),
                )),
                ..Default::default()
            };
            let _ = ctx.run(input, |ctx| {
                egui::Area::new(egui::Id::new("sensor_scroll_test"))
                    .fixed_pos(egui::pos2(500.0, 50.0))
                    .show(ctx, |ui| {
                        ui.set_width(280.0);
                        sensor_preview_scroll(ui, Scene::SensorLab, |ui| {
                            for index in 0..100 {
                                ui.label(format!("Sensor metric {index}"));
                            }
                        });
                        assert!(ui.min_rect().bottom() <= height - 15.0);
                    });
            });
        }
    }

    #[test]
    fn clock_tracks_substeps_and_invalidates_reset_samples() {
        let mut clock = ViewerSensorClock::default();
        let initial = clock.stamp(6);
        for _ in 0..SUBSTEPS_PER_FRAME {
            clock.advance();
        }
        assert_eq!(clock.tick, SUBSTEPS_PER_FRAME as u64);
        assert_eq!(clock.time(), SUBSTEPS_PER_FRAME as f64 * FIXED_DT);
        let paused = clock.stamp(6);
        assert_eq!(clock.stamp(6), paused);
        clock.reset();
        assert_eq!(clock.tick, 0);
        assert_ne!(clock.revision, initial.scene_revision);
        assert_eq!(clock.stamp(6).sensor_id, 6);
    }

    #[test]
    fn preview_budget_rejects_overflow() {
        assert!(preview_pool_config(u64::MAX).is_err());
        assert_eq!(preview_pool_config(40).unwrap().max_total_bytes, 120);
    }

    #[test]
    #[ignore = "requires GPU"]
    fn preview_readback_is_bounded_and_drains_to_latest_without_waiting() {
        use molla::sensors::{SensorChannelSpec, readback::DeviceBuffer};
        let engine = Engine::new_gpu().unwrap();
        let pool = SensorFramePool::new(
            &engine,
            preview_pool_config(8).unwrap(),
            &[SensorChannelSpec::of::<u32>(2)],
        )
        .unwrap();
        let source = DeviceBuffer::from_host(
            &engine,
            vec![1u32, 2],
            super::super::mp::BufferUsage::Storage,
        )
        .unwrap();
        let mut preview = SensorPreview::new(&engine, pool, 8).unwrap();
        let mut clock = ViewerSensorClock::default();
        for _ in 0..PREVIEW_SLOTS {
            clock.advance();
            assert!(preview.begin_sample(clock.tick).unwrap());
            preview
                .enqueue(&[source.readback_source(0..2).unwrap()], clock.stamp(6))
                .unwrap();
        }
        let last_tick = clock.tick;
        clock.advance();
        assert!(!preview.begin_sample(clock.tick).unwrap());
        assert!(!preview.begin_sample(clock.tick).unwrap());
        assert_eq!(preview.deferred, 1);
        assert_eq!(preview.pending.len(), PREVIEW_SLOTS);
        assert!(!preview.begin_request(99).unwrap());
        engine
            .device()
            .unwrap()
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(30)),
            })
            .unwrap();
        let latest = preview.poll_latest(clock).unwrap().unwrap();
        assert_eq!(latest.stamp.sample_id, last_tick);
        assert_eq!(latest.channel::<u32>(0).unwrap(), [1, 2]);
        assert_eq!(preview.collected, 3);
        assert!(preview.pending.is_empty());
        assert_eq!(preview.readback.stats().blocking_waits, 0);
        assert_eq!(preview.readback.stats().copied_bytes, 24);
        assert_eq!(preview.readback.stats().staging_allocations, 3);
        assert!(preview.begin_request(99).unwrap());
        assert!(!preview.begin_request(99).unwrap());
        clock.advance();
        assert!(preview.begin_sample(clock.tick).unwrap());
    }
}
