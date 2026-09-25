#![cfg(feature = "gpu")]

//! Distance, light, radar and joint position sensors against a Z-up scene
//! whose expected readings follow from its geometry.

use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{Quat, Real, SpatialVector, Transform, Vec3};
use molla_sensors::{
    DistanceKind, DistanceSensors, DistanceSpec, LightEnvironment, LightSensors, LightSource,
    RadarSpec, RadarTarget, Radars, SensorScene, SensorSceneOptions, cone_rays, joint_reading,
};
use molla_sim::{BodyParams, Model, ModelBuilder, State};
use std::sync::Mutex;

static GPU: Mutex<()> = Mutex::new(());

struct Fixture {
    model: Model,
    state: State,
    scene: SensorScene,
    wall: BodyId,
    mover: BodyId,
    hidden: BodyId,
    joint: usize,
}

/// Ground plane z = 0; a wall slab at x = 5 (y ∈ ±2, z ∈ [0, 4]); a sphere
/// at (10, 6, 1) moving +X at 1 m/s; a sphere hidden behind the wall at
/// (10, 0, 1); an overhang over (0, -6, 3); an arm on a revolute joint.
fn fixture(engine: &Engine) -> Fixture {
    let mut b = ModelBuilder::new();
    b.add_shape_plane(BodyId::NONE, Vec3::Z, Transform::IDENTITY)
        .unwrap();
    let body = |b: &mut ModelBuilder, at: Vec3| b.add_body(BodyParams::new().with_position(at));
    let wall = body(&mut b, Vec3::new(5.0, 0.0, 2.0));
    b.add_shape_box(wall, Vec3::new(0.1, 2.0, 2.0), Transform::IDENTITY)
        .unwrap();
    let mover = body(&mut b, Vec3::new(10.0, 6.0, 1.0));
    b.add_shape_sphere(mover, 0.5, Transform::IDENTITY).unwrap();
    let hidden = body(&mut b, Vec3::new(10.0, 0.0, 1.0));
    b.add_shape_sphere(hidden, 0.5, Transform::IDENTITY).unwrap();
    let roof = body(&mut b, Vec3::new(0.0, -6.0, 3.0));
    b.add_shape_box(roof, Vec3::new(1.0, 1.0, 0.1), Transform::IDENTITY)
        .unwrap();
    let base = body(&mut b, Vec3::new(-20.0, 0.0, 1.0));
    let arm = body(&mut b, Vec3::new(-20.0, 0.0, 2.0));
    let joint = b
        .add_joint_revolute(base, arm, Transform::IDENTITY, Transform::IDENTITY, Vec3::Z)
        .unwrap();
    let mut model = b.build().unwrap();
    model.gravity.host_mut().unwrap()[0] = Vec3::ZERO;
    let (device, queue) = (engine.device().unwrap(), engine.queue().unwrap());
    model.upload_to_device(device, queue).unwrap();
    let mut state = model.state().unwrap();
    state.body_qd.host_mut().unwrap()[mover.index()] = SpatialVector::new(Vec3::X, Vec3::ZERO);
    let coord = model.joint_coord_offset.host().unwrap()[joint.index()] as usize;
    let dof = model.joint_dof_offset.host().unwrap()[joint.index()] as usize;
    state.joint_q.host_mut().unwrap()[coord] = 0.7;
    state.joint_qd.host_mut().unwrap()[dof] = -0.3;
    state.upload_to_device(device, queue).unwrap();
    let scene = SensorScene::from_model(engine, &model, SensorSceneOptions::default()).unwrap();
    Fixture {
        model,
        state,
        scene,
        wall,
        mover,
        hidden,
        joint: joint.index(),
    }
}

fn at(x: Real, y: Real, z: Real, rotation: Quat) -> Transform {
    Transform::new(Vec3::new(x, y, z), rotation)
}

#[test]
#[ignore = "requires GPU"]
fn distance_sensors_reduce_their_cones_like_laser_infrared_and_sonar() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let f = fixture(&engine);
    let aperture = 0.6;
    let spec = |kind, rays| DistanceSpec {
        kind,
        rays,
        aperture,
        max_range: 20.0,
        gaussian_width: 1.0,
        lookup: vec![[0.0, 0.0], [10.0, 1000.0]],
    };
    let mut sensors = DistanceSensors::new(
        &engine,
        vec![
            spec(DistanceKind::Laser, 1),
            spec(DistanceKind::InfraRed, 5),
            spec(DistanceKind::Sonar, 5),
            spec(DistanceKind::Laser, 1),
        ],
    )
    .unwrap();
    let forward = at(0.0, 0.0, 1.0, Quat::IDENTITY);
    let skyward = at(0.0, 0.0, 1.0, Quat::from_rotation_y(-std::f64::consts::FRAC_PI_2));
    let readings = sensors
        .update(&engine, &f.scene, &f.state, &[forward, forward, forward, skyward])
        .unwrap();

    let face = 4.9;
    assert!((readings[0].distance - face).abs() < 1e-3, "{readings:?}");
    assert_eq!(readings[0].body, f.wall);
    assert!((readings[0].value - 490.0).abs() < 0.2);

    // Each ray from (0, 0, 1) meets the wall face or, if it points down, the ground first.
    let hit = |r: Vec3| {
        let ground = if r.z < 0.0 { -1.0 / r.z } else { Real::INFINITY };
        (face / r.x).min(ground)
    };
    let ring: Vec<Real> = cone_rays(5, aperture)[1..].iter().map(|r| hit(*r)).collect();
    let w = (-(0.3f64 / aperture).powi(2)).exp();
    let expected = (face + w * ring.iter().sum::<Real>()) / (1.0 + 4.0 * w);
    assert!((readings[1].distance - expected).abs() < 2e-3, "{} vs {expected}", readings[1].distance);
    let nearest = ring.iter().copied().fold(face, Real::min);
    assert!(nearest < face);
    assert!((readings[2].distance - nearest).abs() < 1e-3, "{} vs {nearest}", readings[2].distance);
    assert_eq!(readings[2].body, BodyId::NONE);

    assert!(readings[3].distance.is_infinite());
    assert_eq!(readings[3].body, BodyId::NONE);
    assert_eq!(readings[3].value, 1000.0);
}

#[test]
#[ignore = "requires GPU"]
fn light_sensors_add_unblocked_sun_and_sky_by_facing() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let f = fixture(&engine);
    let up = Quat::from_rotation_y(-std::f64::consts::FRAC_PI_2);
    let environment = LightEnvironment {
        sources: vec![LightSource::Directional {
            towards: Vec3::Z,
            irradiance: 1000.0,
        }],
        sky: 100.0,
        up: Vec3::Z,
    };
    let mut sensors = LightSensors::new(&engine, vec![Vec::new(), Vec::new(), vec![[0.0, 0.0], [2000.0, 1.0]]], 1).unwrap();
    let poses = [
        at(0.0, 0.0, 1.0, up),
        at(0.0, -6.0, 1.0, up),
        at(0.0, 3.0, 1.0, Quat::IDENTITY),
    ];
    let r = sensors.update(&engine, &f.scene, &f.state, &poses, &environment).unwrap();
    assert!((r[0].irradiance - 1100.0).abs() < 1e-6 && r[0].visible == 1, "{r:?}");
    assert!((r[1].direct).abs() < 1e-9 && r[1].visible == 0 && (r[1].sky - 100.0).abs() < 1e-9);
    assert!(r[2].direct.abs() < 1e-9 && r[2].visible == 1 && (r[2].sky - 50.0).abs() < 1e-9);
    assert!((r[2].value - 0.025).abs() < 1e-9);
}

#[test]
#[ignore = "requires GPU"]
fn radar_reports_visible_targets_with_bearing_speed_and_power() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let f = fixture(&engine);
    let spec = RadarSpec {
        min_range: 1.0,
        max_range: 30.0,
        horizontal_fov: 1.5,
        vertical_fov: 0.6,
        ..Default::default()
    };
    let targets = [f.wall, f.mover, f.hidden].map(|body| RadarTarget {
        body,
        cross_section: 1.0,
    });
    let mut radars = Radars::new(&engine, targets.len()).unwrap();
    let pose = at(0.0, 0.0, 1.0, Quat::IDENTITY);
    let seen = radars
        .update(&engine, &f.scene, &f.model, &f.state, &spec, pose, BodyId::NONE, &targets)
        .unwrap();
    assert_eq!(seen.iter().map(|d| d.body).collect::<Vec<_>>(), [f.wall, f.mover]);

    let wall = seen[0];
    assert!((wall.distance - 26f64.sqrt()).abs() < 1e-9);
    assert!(wall.azimuth.abs() < 1e-12 && (wall.elevation - 0.2f64.atan()).abs() < 1e-12);
    assert!(wall.radial_speed.abs() < 1e-12);

    let mover = seen[1];
    let distance = 136f64.sqrt();
    assert!((mover.distance - distance).abs() < 1e-9);
    assert!((mover.azimuth - 0.6f64.atan()).abs() < 1e-12 && mover.elevation.abs() < 1e-12);
    assert!((mover.radial_speed - 10.0 / distance).abs() < 1e-9);
    let lambda = 0.299_792_458 / 24.0;
    let gain = lambda * lambda / ((4.0 * std::f64::consts::PI).powi(3) * distance.powi(4));
    assert!((mover.received_power_dbm - (1.0 + 10.0 * gain.log10())).abs() < 1e-9);

    let narrow = RadarSpec {
        horizontal_fov: 0.4,
        min_radial_speed: 0.5,
        ..spec
    };
    let fast = radars
        .update(&engine, &f.scene, &f.model, &f.state, &narrow, pose, BodyId::NONE, &targets)
        .unwrap();
    assert!(fast.is_empty(), "{fast:?}");
    let seeing_through = RadarSpec {
        occlusion: false,
        ..spec
    };
    let all = radars
        .update(&engine, &f.scene, &f.model, &f.state, &seeing_through, pose, BodyId::NONE, &targets)
        .unwrap();
    assert_eq!(all.len(), 3);
}

#[test]
#[ignore = "requires GPU"]
fn position_sensor_reads_joint_coordinate_and_rate() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let f = fixture(&engine);
    let reading = joint_reading(&f.model, &f.state, f.joint).unwrap();
    assert_eq!((reading.position, reading.velocity), (0.7, -0.3));
    assert!(joint_reading(&f.model, &f.state, f.model.joint_count).is_err());
}
