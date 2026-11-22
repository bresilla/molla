//! Scheduled side wind for the AG mission: a light breeze, an on-screen announcement,
//! then a smooth ramp to a strong crosswind blowing across the rows (+x).
use crate::draw::Camera;
use eframe::egui::{self, Color32, FontId, Pos2, Rect, Stroke};
use molla_core::Result;
use molla_math::Vec3;

/// Seconds between the announcement and the start of the ramp.
pub const WARNING_S: f64 = 2.5;
/// Duration of the ramp from breeze to crosswind.
pub const RAMP_S: f64 = 3.5;
/// Seconds the banner stays up after the ramp completes.
const BANNER_HOLD_S: f64 = 5.0;
/// Gust amplitude per m/s of crosswind.
const GUSTS_PER_SPEED: f64 = 0.2;
const CALM: Color32 = Color32::from_rgb(150, 196, 170);
const STRONG: Color32 = Color32::from_rgb(242, 176, 84);

#[derive(Clone, Copy, Debug)]
pub struct Crosswind {
    pub speed: f64,
    breeze: (Vec3, f64),
    strong: (Vec3, f64),
    pub announced: Option<f64>,
}

fn smooth(u: f64) -> f64 {
    let u = u.clamp(0.0, 1.0);
    u * u * (3.0 - 2.0 * u)
}

/// Speed with one decimal unless it is whole, e.g. "3.5 m/s" or "6 m/s".
pub fn label(speed: f64) -> String {
    let tenths = (speed * 10.0).round();
    if tenths % 10.0 == 0.0 {
        format!("{:.0} m/s", tenths / 10.0)
    } else {
        format!("{:.1} m/s", tenths / 10.0)
    }
}

impl Crosswind {
    /// Crosswind of `speed` m/s along +x that keeps the breeze's along-row component.
    pub fn new(breeze: Vec3, breeze_gusts: f64, speed: f64) -> Result<Self> {
        if !speed.is_finite()
            || !(0.0..=20.0).contains(&speed)
            || !breeze.is_finite()
            || !breeze_gusts.is_finite()
            || breeze_gusts < 0.0
        {
            return Err(molla_core::Error::Build(
                "crosswind must be between 0 and 20 m/s over a finite breeze".into(),
            ));
        }
        Ok(Self {
            speed,
            breeze: (breeze, breeze_gusts),
            strong: (
                Vec3::new(speed, 0.0, breeze.z),
                breeze_gusts.max(GUSTS_PER_SPEED * speed),
            ),
            announced: None,
        })
    }

    /// Start the warning period; later calls keep the first time.
    pub fn announce(&mut self, time: f64) {
        self.announced.get_or_insert(time);
    }

    /// Time the ramp begins.
    pub fn onset(&self) -> Option<f64> {
        self.announced.map(|t| t + WARNING_S)
    }

    /// Ramp weight in [0, 1].
    pub fn level(&self, time: f64) -> f64 {
        self.onset().map_or(0.0, |t| smooth((time - t) / RAMP_S))
    }

    /// Background wind and gust amplitude at `time`.
    pub fn background(&self, time: f64) -> (Vec3, f64) {
        let w = self.level(time);
        (
            self.breeze.0.lerp(self.strong.0, w),
            self.breeze.1 + (self.strong.1 - self.breeze.1) * w,
        )
    }
}

fn mix(a: Color32, b: Color32, t: f64) -> Color32 {
    let t = t.clamp(0.0, 1.0) as f32;
    let channel = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    Color32::from_rgb(channel(a.r(), b.r()), channel(a.g(), b.g()), channel(a.b(), b.b()))
}

fn arrow(painter: &egui::Painter, center: Pos2, direction: egui::Vec2, color: Color32) {
    let tip = center + direction * 24.0;
    let tail = center - direction * 24.0;
    let side = egui::vec2(-direction.y, direction.x);
    let stroke = Stroke::new(4.0_f32, color);
    painter.line_segment([tail, tip - direction * 4.0], stroke);
    painter.add(egui::Shape::convex_polygon(
        vec![tip, tip - direction * 14.0 + side * 9.0, tip - direction * 14.0 - side * 9.0],
        color,
        Stroke::NONE,
    ));
}

/// Wind gauge for the whole capture and a banner announcing the crosswind.
pub fn overlay(painter: &egui::Painter, rect: Rect, camera: &Camera, wind: &Crosswind, time: f64) {
    let (ambient, _) = wind.background(time);
    let horizontal = Vec3::new(ambient.x, 0.0, ambient.z);
    let level = wind.level(time);
    let accent = mix(CALM, STRONG, level);
    let gauge = Rect::from_min_size(
        Pos2::new(rect.left() + 40.0, rect.top() + 36.0),
        egui::vec2(330.0, 100.0),
    );
    painter.rect_filled(gauge, 14.0, Color32::from_rgba_unmultiplied(8, 12, 16, 190));
    let state = if level > 0.5 {
        "strong crosswind"
    } else if wind.announced.is_some() {
        "picking up"
    } else {
        "light breeze"
    };
    let text = |offset: egui::Vec2, text: String, size: f32, color: Color32| {
        painter.text(
            gauge.left_top() + offset,
            egui::Align2::LEFT_TOP,
            text,
            FontId::proportional(size),
            color,
        );
    };
    text(egui::vec2(24.0, 16.0), format!("WIND · {state}"), 16.0, accent);
    text(
        egui::vec2(24.0, 42.0),
        format!("{:.1} m/s", horizontal.length()),
        36.0,
        Color32::WHITE,
    );
    let (a, _) = camera.project(camera.target, rect);
    let (b, _) = camera.project(
        camera.target + horizontal.normalize_or_zero() * (0.1 * camera.distance),
        rect,
    );
    let center = Pos2::new(gauge.right() - 54.0, gauge.center().y);
    painter.circle_stroke(center, 34.0, Stroke::new(1.5_f32, accent.gamma_multiply(0.6)));
    if horizontal.length() > 0.05 && (b - a).length() > 1.0 {
        arrow(painter, center, (b - a).normalized(), accent);
    }
    let Some(start) = wind.announced else {
        return;
    };
    let end = start + WARNING_S + RAMP_S + BANNER_HOLD_S;
    let alpha = smooth((time - start) / 0.6).min(smooth(end - time)) as f32;
    if alpha <= 0.0 {
        return;
    }
    let banner = Rect::from_center_size(
        Pos2::new(rect.center().x, rect.top() + 96.0),
        egui::vec2(980.0, 112.0),
    );
    painter.rect_filled(
        banner,
        16.0,
        Color32::from_rgba_unmultiplied(40, 24, 8, 210).gamma_multiply(alpha),
    );
    let line = |y: f32, text: String, size: f32, color: Color32| {
        painter.text(
            Pos2::new(banner.center().x, banner.top() + y),
            egui::Align2::CENTER_TOP,
            text,
            FontId::proportional(size),
            color.gamma_multiply(alpha),
        );
    };
    line(20.0, "SECOND ROW · STRONG CROSSWIND".into(), 32.0, STRONG);
    let detail = if wind.onset().is_some_and(|t| time < t) {
        format!("the wind picks up to {} across the rows", label(wind.speed))
    } else {
        format!("spraying in a {} crosswind: watch the spray drift", label(wind.speed))
    };
    line(66.0, detail, 22.0, Color32::from_rgb(236, 226, 210));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crosswind_waits_for_the_warning_then_ramps_smoothly_to_full_strength() {
        let breeze = Vec3::new(1.2, 0.0, 0.35);
        let mut wind = Crosswind::new(breeze, 0.65, 6.0).unwrap();
        assert_eq!(wind.background(50.0), (breeze, 0.65));
        wind.announce(10.0);
        wind.announce(12.0);
        assert_eq!(wind.onset(), Some(10.0 + WARNING_S));
        assert_eq!(wind.background(10.0 + WARNING_S), (breeze, 0.65));
        let middle = wind.background(10.0 + WARNING_S + RAMP_S * 0.5);
        assert!((middle.0.x - 3.6).abs() < 1e-12 && middle.0.z == 0.35);
        let full = wind.background(10.0 + WARNING_S + RAMP_S + 1.0);
        assert!(full.0.distance(Vec3::new(6.0, 0.0, 0.35)) < 1e-12 && (full.1 - 1.2).abs() < 1e-12);
        assert!(Crosswind::new(breeze, 0.65, f64::NAN).is_err());
        assert!(Crosswind::new(breeze, -1.0, 6.0).is_err());
        assert!(Crosswind::new(breeze, 0.65, 25.0).is_err());
        assert_eq!(label(3.5), "3.5 m/s");
        assert_eq!(label(6.0), "6 m/s");
        assert_eq!(label(4.96), "5 m/s");
    }
}
