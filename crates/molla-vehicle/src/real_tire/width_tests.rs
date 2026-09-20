use super::*;
use crate::{FlatTerrain, Terrain};

#[test]
fn brush_cells_resolve_both_shoulders_not_only_the_centerline() {
    let params = RealTireParams::default();
    let state = RealWheelState {
        hub_pos: [0.0, params.radius * 0.9, 0.0],
        hub_vel: [1.0, 0.0, 0.0], heading: [1.0, 0.0, 0.0], omega: 4.0,
    };
    let contact = evaluate_brush_tire(&params, &state, &FlatTerrain::new(0.0, 1.0), &SurfaceMaterial::new(1.0, 0.82));
    assert!(contact.cells.iter().any(|cell| cell.local_z < 0.0));
    assert!(contact.cells.iter().any(|cell| cell.local_z > 0.0));
}

struct SplitWidth;
impl Terrain for SplitWidth {
    fn height(&self, _: f64, _: f64) -> f64 { 0.0 }
    fn normal(&self, _: f64, _: f64) -> [f64; 3] { [0.0, 1.0, 0.0] }
    fn friction(&self) -> f64 { 1.0 }
    fn friction_at(&self, _: f64, z: f64) -> f64 { if z < 0.0 { 0.0 } else { 1.0 } }
}

fn wheel(params: &RealTireParams) -> RealWheelState {
    RealWheelState {
        hub_pos: [0.0, params.radius * 0.9, 0.0],
        hub_vel: [1.0, 0.0, 0.0], heading: [1.0, 0.0, 0.0], omega: 8.0,
    }
}

#[test]
fn mixed_shoulders_have_independent_shear_and_the_correct_yaw_moment() {
    let params = RealTireParams { width_segments: 4, ..Default::default() };
    let state = wheel(&params);
    let material = SurfaceMaterial { rolling_resistance: 0.0, ..SurfaceMaterial::new(1.0, 0.82) };
    let mut history = TireBrushState::default();
    for _ in 0..3 {
        let contact = evaluate_planar_brush_tire_persistent(&params, &state, &SplitWidth, &material, &mut history, 0.001).unwrap();
        let mut torque = 0.0;
        for cell in &contact.cells {
            if cell.local_z < 0.0 {
                assert_eq!(cell.tangential_force, 0.0);
                assert_eq!(cell.lateral_force, 0.0);
            } else { assert!(cell.tangential_force > 0.0); }
            torque += cell.local_z * cell.tangential_force - cell.local_x * cell.lateral_force;
        }
        assert!(torque > 0.0);
        assert!((contact.aligning_moment_z - torque).abs() < 1e-10);
    }
}

#[test]
fn width_refinement_preserves_uniform_load_area_and_persistent_forces() {
    let mut baseline: Option<RealTireContact> = None;
    let material = SurfaceMaterial::new(1.0, 0.82);
    for width_segments in [1, 2, 3, 8, 16] {
        let params = RealTireParams { width_segments, ..Default::default() };
        let mut state = wheel(&params);
        state.hub_vel[2] = 0.2;
        let mut history = TireBrushState::default();
        let contact = evaluate_planar_brush_tire_persistent(&params, &state, &FlatTerrain::new(0.0, 1.0), &material, &mut history, 0.001).unwrap();
        assert!(contact.cells.iter().all(|cell| cell.local_z.abs() < params.width * 0.5));
        if let Some(base) = &baseline {
            for (a, b) in contact.force.iter().zip(base.force) { assert!((a - b).abs() < 1e-8); }
            assert!((contact.normal_load - base.normal_load).abs() < 1e-8);
            assert!((contact.aligning_moment_z - base.aligning_moment_z).abs() < 1e-8);
            assert!((contact.available_friction_budget - base.available_friction_budget).abs() < 1e-8);
            let area: f64 = contact.cells.iter().map(|cell| cell.area).sum();
            let base_area: f64 = base.cells.iter().map(|cell| cell.area).sum();
            assert!((area - base_area).abs() < 1e-12);
            assert_eq!(contact.cells.len(), base.cells.len() * width_segments);
        } else { baseline = Some(contact); }
    }
}

#[test]
fn changing_width_resolution_resets_history_and_invalid_steps_are_atomic() {
    let mut params = RealTireParams { width_segments: 1, ..Default::default() };
    let state = wheel(&params);
    let material = SurfaceMaterial::new(1.0, 0.82);
    let mut history = TireBrushState::default();
    evaluate_planar_brush_tire_persistent(&params, &state, &SplitWidth, &material, &mut history, 0.01).unwrap();
    params.width_segments = 4;
    let stored = history.clone();
    assert!(evaluate_planar_brush_tire_persistent(&params, &state, &SplitWidth, &material, &mut history, f64::NAN).is_err());
    assert_eq!(history.geometry, stored.geometry);
    assert_eq!(history.cells, stored.cells);
    let actual = evaluate_planar_brush_tire_persistent(&params, &state, &SplitWidth, &material, &mut history, 0.001).unwrap();
    let fresh = evaluate_planar_brush_tire_persistent(&params, &state, &SplitWidth, &material, &mut TireBrushState::default(), 0.001).unwrap();
    assert_eq!(actual.force, fresh.force);
    assert_eq!(actual.aligning_moment_z, fresh.aligning_moment_z);
}
