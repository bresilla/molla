use super::{button, egui};

pub(super) struct PlaybackControls {
    pub pause: egui::Response,
    pub step: egui::Response,
    pub reset: egui::Response,
}

pub(super) fn playback_controls(
    ui: &mut egui::Ui,
    paused: bool,
    accent: egui::Color32,
) -> PlaybackControls {
    ui.columns(3, |columns| PlaybackControls {
        pause: button(
            &mut columns[0],
            if paused { "Play" } else { "Pause" },
            accent,
        ),
        step: columns[1]
            .add_enabled_ui(paused, |ui| button(ui, "Step", accent))
            .inner,
        reset: button(&mut columns[2], "Reset", accent),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sensor_physics_stats_clear_on_revision_not_on_pause() {
        let mut stats = crate::SimStats::default();
        stats.begin_revision(3);
        stats.smoothed_step_us = 10.0;
        stats.smoothed_solve_us = 6.0;
        stats.smoothed_collide_us = 2.0;
        stats.smoothed_download_us = 2.0;
        stats.last_contacts = 7;
        stats.begin_revision(3);
        assert_eq!(stats.smoothed_step_us, 10.0);
        assert_eq!(stats.last_contacts, 7);
        stats.begin_revision(4);
        assert_eq!(stats.revision, Some(4));
        assert_eq!(stats.smoothed_step_us, 0.0);
        assert_eq!(stats.smoothed_solve_us, 0.0);
        assert_eq!(stats.smoothed_collide_us, 0.0);
        assert_eq!(stats.smoothed_download_us, 0.0);
        assert_eq!(stats.last_contacts, 0);
    }

    fn render(
        ctx: &egui::Context,
        width: f32,
        paused: bool,
        events: Vec<egui::Event>,
    ) -> PlaybackControls {
        let mut controls = None;
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(width, 200.0),
                )),
                events,
                ..Default::default()
            },
            |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    let bounds = ui.available_rect_before_wrap();
                    let row = playback_controls(ui, paused, egui::Color32::LIGHT_BLUE);
                    for response in [&row.pause, &row.step, &row.reset] {
                        assert!(bounds.contains_rect(response.rect));
                        assert!(response.rect.width() >= 50.0);
                    }
                    assert!(row.pause.rect.right() <= row.step.rect.left());
                    assert!(row.step.rect.right() <= row.reset.rect.left());
                    controls = Some(row);
                });
            },
        );
        controls.unwrap()
    }

    #[test]
    fn sensor_playback_controls_fit_and_receive_clicks() {
        for width in [240.0, 320.0] {
            for paused in [false, true] {
                let ctx = egui::Context::default();
                let row = render(&ctx, width, paused, vec![]);
                assert_eq!(row.step.enabled(), paused);
                for (index, pos) in [row.pause.rect, row.step.rect, row.reset.rect]
                    .map(|rect| rect.center())
                    .into_iter()
                    .enumerate()
                {
                    for pressed in [true, false] {
                        let row = render(
                            &ctx,
                            width,
                            paused,
                            vec![
                                egui::Event::PointerMoved(pos),
                                egui::Event::PointerButton {
                                    pos,
                                    button: egui::PointerButton::Primary,
                                    pressed,
                                    modifiers: egui::Modifiers::NONE,
                                },
                            ],
                        );
                        if !pressed {
                            assert_eq!(
                                [row.pause.clicked(), row.step.clicked(), row.reset.clicked()],
                                [index == 0, index == 1 && paused, index == 2]
                            );
                        }
                    }
                }
            }
        }
    }
}
