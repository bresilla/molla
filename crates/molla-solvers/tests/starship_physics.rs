use std::f64::consts::FRAC_PI_2;

use molla_math::{GRAVITY, Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::runtime::BodyDesc;
use molla_sim::{BodyParams, MassProperties};
use molla_solvers::devices::{
    AeroBodySpec, AeroSurfaceKind, AeroSurfaceSpec, BiquadNotchFilter, FlexBodySpec, ImuSpec,
    MARS_GRAVITY, MOON_GRAVITY, PlanetAtmosphere, PropellantTankSpec, RocketEngineCommand,
    RocketEngineSpec, WindField, earth_rotation_at_latitude, free_free_beam_mode, mars_atmosphere,
    us76_atmosphere,
};
use molla_solvers::rigid_world::RigidWorld;

fn dynamic_stage(w: &mut RigidWorld, pos: Vec3, dry_mass: f64, inertia: Vec3) -> molla_sim::runtime::BodyHandle {
    w.scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_pose(Transform::from_translation(pos)),
            additional_mass: Some(MassProperties {
                mass: dry_mass,
                local_com: Vec3::ZERO,
                inertia: Mat3::from_diagonal(inertia),
            }),
            ..BodyDesc::default()
        })
        .unwrap()
}

#[test]
fn us76_atmosphere_matches_standard_tables_from_sea_level_to_100km() {
    let sl = us76_atmosphere(0.0);
    assert!((sl.temperature_k - 288.15).abs() < 1e-6);
    assert!((sl.pressure_pa - 101_325.0).abs() < 1.0);
    assert!((sl.density_kg_m3 - 1.225).abs() < 1e-3);
    assert!((sl.speed_of_sound_mps - 340.29).abs() < 0.2);

    let trop = us76_atmosphere(11_000.0);
    assert!((trop.temperature_k - 216.65).abs() < 0.1);
    assert!((trop.pressure_pa - 22_632.0).abs() < 50.0);

    let strat = us76_atmosphere(50_000.0);
    assert!((strat.temperature_k - 270.65).abs() < 0.5);
    assert!(strat.pressure_pa > 60.0 && strat.pressure_pa < 95.0);

    let karman = us76_atmosphere(100_000.0);
    assert!(karman.density_kg_m3 > 1e-7 && karman.density_kg_m3 < 1e-6);
}

#[test]
fn raptor_nozzle_thrust_varies_with_altitude_and_triggers_schmucker_flow_separation() {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();
    w.devices.rockets.set_flat_ground(Some(0.0), 0.0);

    let stage_sl = dynamic_stage(&mut w, Vec3::new(0.0, 100.0, 0.0), 120_000.0, Vec3::splat(1e7));
    let stage_vac = dynamic_stage(&mut w, Vec3::new(0.0, 120_000.0, 0.0), 120_000.0, Vec3::splat(1e7));
    let up_frame = Transform::from_rotation(Quat::from_rotation_z(FRAC_PI_2));

    let r_sl_low = w
        .devices
        .rockets
        .insert_engine(&w.scene, RocketEngineSpec::raptor3_sea_level(stage_sl, up_frame, None, None, 0.0))
        .unwrap();
    let r_sl_high = w
        .devices
        .rockets
        .insert_engine(&w.scene, RocketEngineSpec::raptor3_sea_level(stage_vac, up_frame, None, None, 0.0))
        .unwrap();
    let r_vac_at_sl = w
        .devices
        .rockets
        .insert_engine(&w.scene, RocketEngineSpec::raptor3_vacuum(stage_sl, up_frame, None, None))
        .unwrap();

    for e in [r_sl_low, r_sl_high, r_vac_at_sl] {
        w.devices
            .rockets
            .command_engine(e, RocketEngineCommand { ignited: true, throttle: 1.0, gimbal_rad: [0.0, 0.0] })
            .unwrap();
    }
    for _ in 0..200 {
        w.step(0.01).unwrap();
    }
    let read_sl = w.devices.rockets.engine_reading(r_sl_low).unwrap();
    let read_vac = w.devices.rockets.engine_reading(r_sl_high).unwrap();
    let read_rvac_sl = w.devices.rockets.engine_reading(r_vac_at_sl).unwrap();

    // Exact nozzle equation: F_vac - F_sl = P_a * A_e!
    let ae = w.devices.rockets.engine_spec(r_sl_low).unwrap().exit_area_m2();
    let expected_diff = us76_atmosphere(100.0).pressure_pa * ae;
    assert!(
        ((read_vac.thrust_n - read_sl.thrust_n) - expected_diff).abs() < 2_500.0,
        "F_vac - F_sl mismatch: got {:.1} vs {:.1}",
        read_vac.thrust_n - read_sl.thrust_n,
        expected_diff
    );
    assert!(read_vac.effective_isp_s > read_sl.effective_isp_s + 15.0);
    // Firing huge RVac bell (P_e = 14.5 kPa) at sea level (P_a = 101 kPa) MUST trigger Schmucker flow separation + side-loads!
    assert!(
        read_rvac_sl.flow_separated && read_rvac_sl.side_load_n > 1_000.0,
        "RVac at sea level must separate and produce side loads: {read_rvac_sl:?}"
    );
    assert!(
        !read_sl.flow_separated,
        "Raptor SL at full throttle should not separate at sea level"
    );
    assert!(read_sl.shock_cell_length_m > 4.0 && read_sl.shock_cell_length_m < 6.5);
}

#[test]
fn multi_tank_drain_shifts_com_and_inertia_and_scales_slosh_with_axial_g() {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::new(0.0, -GRAVITY, 0.0)).unwrap();
    w.devices.rockets.set_flat_ground(Some(0.0), 0.0);

    let stage = dynamic_stage(&mut w, Vec3::new(0.0, 1_000.0, 0.0), 100_000.0, Vec3::splat(2.0e7));
    let lox = w
        .devices
        .rockets
        .insert_tank(
            &mut w.scene,
            PropellantTankSpec {
                body: stage,
                base_local: Vec3::new(0.0, -15.0, 0.0),
                axis_local: Vec3::Y,
                radius_m: 4.5,
                height_m: 14.0,
                liquid_density_kg_m3: 1140.0,
                initial_mass_kg: 700_000.0,
                slosh_mass_fraction: 0.25,
                slosh_damping_ratio: 0.05,
                capillary_omega2: 0.2,
            },
        )
        .unwrap();
    let ch4 = w
        .devices
        .rockets
        .insert_tank(
            &mut w.scene,
            PropellantTankSpec {
                body: stage,
                base_local: Vec3::new(0.0, 0.0, 0.0),
                axis_local: Vec3::Y,
                radius_m: 4.5,
                height_m: 14.0,
                liquid_density_kg_m3: 422.0,
                initial_mass_kg: 200_000.0,
                slosh_mass_fraction: 0.25,
                slosh_damping_ratio: 0.05,
                capillary_omega2: 0.2,
            },
        )
        .unwrap();

    let mp0 = w.scene.body_mass_properties(stage).unwrap();
    assert!((mp0.mass - 1_000_000.0).abs() < 1.0);
    let com_y0 = mp0.local_com.y;

    let up_frame = Transform::new(Vec3::new(0.0, -24.0, 0.0), Quat::from_rotation_z(FRAC_PI_2));
    let mut engines = Vec::new();
    for _ in 0..8 {
        let e = w
            .devices
            .rockets
            .insert_engine(
                &w.scene,
                RocketEngineSpec::raptor3_sea_level(stage, up_frame, Some(lox), Some(ch4), 0.25),
            )
            .unwrap();
        w.devices
            .rockets
            .command_engine(e, RocketEngineCommand { ignited: true, throttle: 1.0, gimbal_rad: [0.0, 0.0] })
            .unwrap();
        engines.push(e);
    }

    for _ in 0..400 {
        w.step(0.02).unwrap();
    }
    let mp1 = w.scene.body_mass_properties(stage).unwrap();
    assert!(mp1.mass < mp0.mass - 40_000.0, "propellant did not drain");
    assert!(
        (mp1.local_com.y - com_y0).abs() > 0.15,
        "COM should shift as tanks drain: y0={com_y0:.3}, y1={:.3}",
        mp1.local_com.y
    );
    // Under ~2g axial acceleration, NASA SP-8009 slosh frequency in R=4.5 m tank is ~0.4–0.5 Hz!
    let tr = w.devices.rockets.tank_reading(lox).unwrap();
    assert!(
        tr.slosh_freq_hz > 0.30 && tr.slosh_freq_hz < 0.75,
        "unexpected SP-8009 slosh freq under ~2g burn: {:.3} Hz",
        tr.slosh_freq_hz
    );
}

#[test]
fn tvc_gimbal_produces_tail_wags_dog_counter_torque_and_hot_staging_blasts_booster() {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();

    let booster = dynamic_stage(&mut w, Vec3::new(0.0, 100.0, 0.0), 275_000.0, Vec3::splat(1e8));
    let ship = dynamic_stage(&mut w, Vec3::new(0.0, 145.0, 0.0), 120_000.0, Vec3::splat(2e7));
    w.devices
        .rockets
        .register_interstage_blast_target(ship, booster, Vec3::new(0.0, 35.0, 0.0), 4.5);

    let up_frame = Transform::new(Vec3::new(0.0, -5.0, 0.0), Quat::from_rotation_z(FRAC_PI_2));
    let eng = w
        .devices
        .rockets
        .insert_engine(
            &w.scene,
            RocketEngineSpec::raptor3_sea_level(ship, up_frame, None, None, 0.26),
        )
        .unwrap();

    // Command rapid TVC gimbal step while unignited -> pure Tail-Wags-Dog inertial torque!
    w.devices
        .rockets
        .command_engine(
            eng,
            RocketEngineCommand {
                ignited: false,
                throttle: 0.0,
                gimbal_rad: [0.22, 0.0],
            },
        )
        .unwrap();
    w.step(0.01).unwrap();
    let r_twd = w.devices.rockets.engine_reading(eng).unwrap();
    assert!(
        r_twd.twd_torque_nm > 10_000.0,
        "expected strong TWD inertial torque from 1,525 kg Raptor slew: {r_twd:?}"
    );

    // Now ignite Ship engine above Booster hot-stage dome -> real downward hot-staging blast force on Booster!
    w.devices
        .rockets
        .command_engine(
            eng,
            RocketEngineCommand {
                ignited: true,
                throttle: 1.0,
                gimbal_rad: [0.0, 0.0],
            },
        )
        .unwrap();
    for _ in 0..60 {
        w.step(0.01).unwrap();
    }
    assert!(
        w.devices.rockets.last_interstage_blast_n() > 200_000.0,
        "expected hot-staging plume blast onto booster dome: {:.1} N",
        w.devices.rockets.last_interstage_blast_n()
    );
    assert!(
        w.scene.body_velocity(booster).unwrap().linear.y < -0.5,
        "booster must be physically pushed aft by hot-staging blast!"
    );
}

#[test]
fn rocket_aero_bellyflop_crossflow_grid_fin_choking_and_srp_drag_loss() {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();
    w.devices.rockets.set_flat_ground(Some(0.0), 0.0);
    w.devices.rocket_aero.set_wind(WindField::calm());

    // 1. Starship in 90° horizontal belly-flop at 85 m/s downward -> strong upward crossflow drag!
    let ship = dynamic_stage(&mut w, Vec3::new(0.0, 1_000.0, 0.0), 120_000.0, Vec3::splat(2e7));
    w.scene
        .set_body_pose(ship, Transform::new(Vec3::new(0.0, 1_000.0, 0.0), Quat::from_rotation_z(FRAC_PI_2)))
        .unwrap();
    w.scene
        .set_body_velocity(ship, SpatialVector::new(Vec3::new(0.0, -85.0, 0.0), Vec3::ZERO))
        .unwrap();
    let ab = w
        .devices
        .rocket_aero
        .insert_body(
            &w.scene,
            AeroBodySpec {
                body: ship,
                axis_local: Vec3::Y,
                diameter_m: 9.0,
                length_m: 50.0,
                cd_axial_nose: 0.32,
                cd_axial_tail: 0.95,
                cd_crossflow_subsonic: 1.20,
                cd_crossflow_supersonic: 1.50,
                cp_crossflow_local: 0.0,
                cp_axial_shift_m: -4.0,
                rot_damping_scale: 1.0,
                srp_drag_loss_gain: 0.9,
            },
        )
        .unwrap();
    w.step(0.01).unwrap();
    let r_bf = w.devices.rocket_aero.body_reading(ab).unwrap();
    assert!((r_bf.aoa_rad - FRAC_PI_2).abs() < 0.02);
    // Terminal belly-flop drag on 9×50 m cylinder at 85 m/s should be ~1.5–2.5 MN upward!
    assert!(
        r_bf.world_force.y > 1_000_000.0,
        "belly-flop crossflow drag too low: {:?}",
        r_bf.world_force
    );

    // 2. Grid fin transonic choking dip around M = 1.12 vs M = 0.6 & M = 2.0!
    let test_fin_at_mach = |mach: f64| -> f64 {
        let mut wf = RigidWorld::new().unwrap();
        wf.scene.set_gravity(Vec3::ZERO).unwrap();
        wf.devices.rockets.set_flat_ground(Some(0.0), 0.0);
        let b = dynamic_stage(&mut wf, Vec3::new(0.0, 12_000.0, 0.0), 275_000.0, Vec3::splat(1e8));
        let a = us76_atmosphere(12_000.0).speed_of_sound_mps;
        wf.scene
            .set_body_velocity(b, SpatialVector::new(Vec3::new(0.0, -mach * a, 0.0), Vec3::ZERO))
            .unwrap();
        let fin = wf
            .devices
            .rocket_aero
            .insert_surface(
                &wf.scene,
                AeroSurfaceSpec {
                    body: b,
                    kind: AeroSurfaceKind::GridFin,
                    center_local: Vec3::new(4.8, 30.0, 0.0),
                    hinge_axis_local: Vec3::X,
                    normal_local: Vec3::Z,
                    area_m2: 6.0,
                    hinge_moment_arm_m: 0.35,
                    min_deflection_rad: -0.35,
                    max_deflection_rad: 0.35,
                    max_rate_rad_s: 5.0,
                    max_hinge_torque_nm: 500_000.0,
                    cn_max: 2.2,
                },
            )
            .unwrap();
        wf.devices.rocket_aero.command_surface(fin, 0.15).unwrap();
        for _ in 0..20 {
            wf.step(0.01).unwrap();
        }
        let f_mag = wf.devices.rocket_aero.surface_reading(fin).unwrap().world_force.length();
        let q = 0.5 * us76_atmosphere(12_000.0).density_kg_m3 * (mach * a).powi(2);
        f_mag / (q * 6.0) // Effective normalized C_N!
    };
    let cn_sub = test_fin_at_mach(0.6);
    let cn_trans = test_fin_at_mach(1.12);
    let cn_sup = test_fin_at_mach(2.0);
    assert!(
        cn_trans < 0.65 * cn_sub && cn_trans < 0.65 * cn_sup,
        "grid fin should show transonic choking dip: sub={cn_sub:.3}, trans={cn_trans:.3}, sup={cn_sup:.3}"
    );

    // 3. Supersonic Retropropulsion (SRP) drag loss when firing into M=2.2 flow!
    let mut ws = RigidWorld::new().unwrap();
    ws.scene.set_gravity(Vec3::ZERO).unwrap();
    ws.devices.rockets.set_flat_ground(Some(0.0), 0.0);
    let b = dynamic_stage(&mut ws, Vec3::new(0.0, 20_000.0, 0.0), 275_000.0, Vec3::splat(1e8));
    let a20 = us76_atmosphere(20_000.0).speed_of_sound_mps;
    ws.scene
        .set_body_velocity(b, SpatialVector::new(Vec3::new(0.0, -2.2 * a20, 0.0), Vec3::ZERO))
        .unwrap();
    let ab_srp = ws
        .devices
        .rocket_aero
        .insert_body(
            &ws.scene,
            AeroBodySpec {
                body: b,
                axis_local: Vec3::Y,
                diameter_m: 9.0,
                length_m: 71.0,
                cd_axial_nose: 0.35,
                cd_axial_tail: 0.95,
                cd_crossflow_subsonic: 1.2,
                cd_crossflow_supersonic: 1.5,
                cp_crossflow_local: 0.0,
                cp_axial_shift_m: 0.0,
                rot_damping_scale: 1.0,
                srp_drag_loss_gain: 0.9,
            },
        )
        .unwrap();
    let up_frame = Transform::from_rotation(Quat::from_rotation_z(FRAC_PI_2));
    let e = ws
        .devices
        .rockets
        .insert_engine(&ws.scene, RocketEngineSpec::raptor3_sea_level(b, up_frame, None, None, 0.0))
        .unwrap();
    ws.step(0.01).unwrap();
    let srp_off = ws.devices.rocket_aero.body_reading(ab_srp).unwrap().srp_drag_factor;
    ws.devices
        .rockets
        .command_engine(e, RocketEngineCommand { ignited: true, throttle: 1.0, gimbal_rad: [0.0, 0.0] })
        .unwrap();
    for _ in 0..80 {
        ws.step(0.01).unwrap();
    }
    let srp_on = ws.devices.rocket_aero.body_reading(ab_srp).unwrap().srp_drag_factor;
    assert!(
        (srp_off - 1.0).abs() < 1e-6 && srp_on < 0.75,
        "SRP retropropulsion must reduce axial body drag factor: off={srp_off:.3}, on={srp_on:.3}"
    );
}

#[test]
fn flex_bending_modes_excite_imu_and_biquad_notch_rejects_bending_and_pogo_couples() {
    // 1. Verify free-free Euler-Bernoulli eigenfunctions at ends and midpoint.
    let (phi1_0, _) = free_free_beam_mode(1, 0.0);
    let (phi1_1, _) = free_free_beam_mode(1, 1.0);
    let (phi2_1, _) = free_free_beam_mode(2, 1.0);
    assert!((phi1_0 - 1.0).abs() < 1e-6 && (phi1_1 - 1.0).abs() < 1e-6);
    assert!((phi2_1 + 1.0).abs() < 1e-6);

    // 2. Excite Super Heavy bending modes with lateral TVC gimbal + axial Pogo with ignition transient!
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();
    w.devices.rockets.set_flat_ground(Some(0.0), 0.0);
    let b = dynamic_stage(&mut w, Vec3::new(0.0, 2_000.0, 0.0), 275_000.0, Vec3::splat(1.2e8));
    let flex_id = w
        .devices
        .rocket_structure
        .insert(&w.scene, FlexBodySpec::super_heavy(b))
        .unwrap();
    let mut ideal_imu = ImuSpec::tactical_fog(b, 42);
    ideal_imu.gyro_bias_init = Vec3::ZERO;
    ideal_imu.gyro_arw = 0.0;
    ideal_imu.gyro_rrw = 0.0;
    ideal_imu.gyro_quantization = 0.0;
    ideal_imu.latency_s = 0.0;
    ideal_imu.sample_rate_hz = 0.0;
    let imu_id = w.devices.space_sensors.insert_imu(&w.scene, ideal_imu).unwrap();
    let up_frame = Transform::new(Vec3::new(0.0, -34.7, 0.0), Quat::from_rotation_z(FRAC_PI_2));
    let eng = w
        .devices
        .rockets
        .insert_engine(&w.scene, RocketEngineSpec::raptor3_sea_level(b, up_frame, None, None, 0.25))
        .unwrap();
    w.devices
        .rockets
        .command_engine(eng, RocketEngineCommand { ignited: true, throttle: 1.0, gimbal_rad: [0.22, 0.0] })
        .unwrap();

    let mut max_tip = 0.0_f64;
    let mut max_imu_bend = 0.0_f64;
    let mut max_imu_flex_diff = 0.0_f64;
    let mut max_pogo_g = 0.0_f64;
    for _ in 0..80 {
        w.step(1.0 / 60.0).unwrap();
        let r = w.devices.rocket_structure.reading(flex_id).unwrap();
        let ir = w.devices.space_sensors.imu_reading(imu_id).unwrap();
        let bp = w.scene.body_pose(b).unwrap();
        let bv = w.scene.body_velocity(b).unwrap();
        let rigid_omega_b = bp.rotation.conjugate() * bv.angular;
        max_imu_flex_diff = max_imu_flex_diff.max((ir.angular_velocity - rigid_omega_b).length());
        max_tip = max_tip.max(r.tip_deflection_m);
        max_imu_bend = max_imu_bend.max(r.imu_bending_rate_body.length());
        max_pogo_g = max_pogo_g.max(r.pogo_accel_g);
    }
    assert!(max_tip > 1e-3, "lateral TVC must excite fuselage bending: {max_tip:.6} m");
    assert!(max_imu_bend > 1e-4, "IMU must pick up elastic bending slope rate: {max_imu_bend:.6} rad/s");
    assert!(
        max_imu_flex_diff > 1e-4,
        "SpaceSensors ideal IMU must automatically include RocketStructure bending feedthrough: {max_imu_flex_diff:.6}"
    );
    assert!(max_pogo_g > 1e-3, "thrust ramp must excite SP-8055 longitudinal Pogo mode: {max_pogo_g:.5} g");

    // 3. Verify digital BiquadNotchFilter kills 1st bending sine (> 25 dB) while passing rigid rate!
    let fs = 60.0;
    let f_bend = 2.8;
    let mut notch_bend = BiquadNotchFilter::new(f_bend, 3.0, fs);
    let mut notch_rigid = BiquadNotchFilter::new(f_bend, 3.0, fs);
    let mut peak_bend = 0.0_f64;
    let mut peak_rigid = 0.0_f64;
    for k in 0..600 {
        let t = k as f64 / fs;
        let yb = notch_bend.filter((std::f64::consts::TAU * f_bend * t).sin());
        let yr = notch_rigid.filter((std::f64::consts::TAU * 0.15 * t).sin());
        if k > 300 {
            peak_bend = peak_bend.max(yb.abs());
            peak_rigid = peak_rigid.max(yr.abs());
        }
    }
    assert!(
        peak_bend < 0.05 && peak_rigid > 0.95,
        "biquad notch must kill bending ({peak_bend:.4}) and pass rigid motion ({peak_rigid:.4})"
    );
}

#[test]
fn ullage_pressurization_cavitation_and_rotating_earth_coriolis() {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();
    w.devices.rockets.set_flat_ground(Some(0.0), 0.0);
    w.devices
        .rockets
        .set_earth_rotation(earth_rotation_at_latitude(26.0));

    let s_ok = dynamic_stage(&mut w, Vec3::new(0.0, 500.0, 0.0), 120_000.0, Vec3::splat(2e7));
    let s_cav = dynamic_stage(&mut w, Vec3::new(100.0, 500.0, 0.0), 120_000.0, Vec3::splat(2e7));
    let t_ok = w
        .devices
        .rockets
        .insert_tank(
            &mut w.scene,
            PropellantTankSpec {
                body: s_ok,
                base_local: Vec3::new(0.0, -10.0, 0.0),
                axis_local: Vec3::Y,
                radius_m: 2.0,
                height_m: 6.0,
                liquid_density_kg_m3: 1140.0,
                initial_mass_kg: 40_000.0,
                slosh_mass_fraction: 0.15,
                slosh_damping_ratio: 0.06,
                capillary_omega2: 0.2,
            },
        )
        .unwrap();
    let t_cav = w
        .devices
        .rockets
        .insert_tank(
            &mut w.scene,
            PropellantTankSpec {
                body: s_cav,
                base_local: Vec3::new(0.0, -10.0, 0.0),
                axis_local: Vec3::Y,
                radius_m: 2.0,
                height_m: 6.0,
                liquid_density_kg_m3: 1140.0,
                initial_mass_kg: 40_000.0,
                slosh_mass_fraction: 0.15,
                slosh_damping_ratio: 0.06,
                capillary_omega2: 0.2,
            },
        )
        .unwrap();
    // Healthy 5.5 bar header vs depressurized 1.15 bar cavitating tank!
    w.devices
        .rockets
        .set_tank_pressurization(t_ok, 550_000.0, 110_000.0, 14.0)
        .unwrap();
    w.devices
        .rockets
        .set_tank_pressurization(t_cav, 115_000.0, 110_000.0, 28.0)
        .unwrap();

    let up_frame = Transform::new(Vec3::new(0.0, -20.0, 0.0), Quat::from_rotation_z(FRAC_PI_2));
    let e_ok = w
        .devices
        .rockets
        .insert_engine(&w.scene, RocketEngineSpec::raptor3_sea_level(s_ok, up_frame, Some(t_ok), None, 0.0))
        .unwrap();
    let e_cav = w
        .devices
        .rockets
        .insert_engine(&w.scene, RocketEngineSpec::raptor3_sea_level(s_cav, up_frame, Some(t_cav), None, 0.0))
        .unwrap();
    for e in [e_ok, e_cav] {
        w.devices
            .rockets
            .command_engine(e, RocketEngineCommand { ignited: true, throttle: 1.0, gimbal_rad: [0.0, 0.0] })
            .unwrap();
    }
    for _ in 0..120 {
        w.step(0.02).unwrap();
    }
    let r_ok = w.devices.rockets.engine_reading(e_ok).unwrap();
    let r_cav = w.devices.rockets.engine_reading(e_cav).unwrap();
    assert!(!r_ok.cavitating && r_cav.cavitating, "low-ullage tank must trigger real pump cavitation");
    assert!(
        r_cav.thrust_n < 0.75 * r_ok.thrust_n,
        "cavitation must degrade thrust: ok={:.0} N, cav={:.0} N",
        r_ok.thrust_n,
        r_cav.thrust_n
    );
    // Ascending rocket at 26°N must deflect Westward (-X) due to exact Coriolis -2m(Ω × v)!
    let v_ok = w.scene.body_velocity(s_ok).unwrap().linear;
    assert!(
        v_ok.y > 20.0 && v_ok.x < -1e-3,
        "ascending rocket must experience Westward Coriolis deflection: v={v_ok:?}"
    );
}

#[test]
fn pifs_cp_shift_transonic_buffet_dryden_turbulence_and_knudsen_bridging() {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();
    w.devices.rockets.set_flat_ground(Some(0.0), 0.0);
    w.devices.rocket_aero.set_wind(WindField::boca_chica_moderate());

    // 1. High-altitude supersonic powered ascent -> underexpanded plume drives forward PIFS CP shift!
    let b_pifs = dynamic_stage(&mut w, Vec3::new(0.0, 22_000.0, 0.0), 275_000.0, Vec3::splat(1e8));
    let a22 = us76_atmosphere(22_000.0).speed_of_sound_mps;
    w.scene
        .set_body_velocity(b_pifs, SpatialVector::new(Vec3::new(0.0, 1.8 * a22, 0.0), Vec3::ZERO))
        .unwrap();
    let ab_pifs = w
        .devices
        .rocket_aero
        .insert_body(
            &w.scene,
            AeroBodySpec {
                body: b_pifs,
                axis_local: Vec3::Y,
                diameter_m: 9.0,
                length_m: 71.0,
                cd_axial_nose: 0.34,
                cd_axial_tail: 0.95,
                cd_crossflow_subsonic: 1.2,
                cd_crossflow_supersonic: 1.5,
                cp_crossflow_local: 0.0,
                cp_axial_shift_m: -4.5,
                rot_damping_scale: 1.0,
                srp_drag_loss_gain: 0.8,
            },
        )
        .unwrap();
    let up_frame = Transform::new(Vec3::new(0.0, -34.0, 0.0), Quat::from_rotation_z(FRAC_PI_2));
    let eng = w
        .devices
        .rockets
        .insert_engine(&w.scene, RocketEngineSpec::raptor3_sea_level(b_pifs, up_frame, None, None, 0.0))
        .unwrap();
    w.devices
        .rockets
        .command_engine(eng, RocketEngineCommand { ignited: true, throttle: 1.0, gimbal_rad: [0.0, 0.0] })
        .unwrap();

    // 2. Transonic Max-Q stage at M = 1.02, 8 km -> real stochastic transonic buffet torque!
    let b_buf = dynamic_stage(&mut w, Vec3::new(200.0, 8_000.0, 0.0), 275_000.0, Vec3::splat(1e8));
    let a8 = us76_atmosphere(8_000.0).speed_of_sound_mps;
    w.scene
        .set_body_velocity(b_buf, SpatialVector::new(Vec3::new(0.0, 1.02 * a8, 0.0), Vec3::ZERO))
        .unwrap();
    let ab_buf = w
        .devices
        .rocket_aero
        .insert_body(
            &w.scene,
            AeroBodySpec {
                body: b_buf,
                axis_local: Vec3::Y,
                diameter_m: 9.0,
                length_m: 71.0,
                cd_axial_nose: 0.34,
                cd_axial_tail: 0.95,
                cd_crossflow_subsonic: 1.2,
                cd_crossflow_supersonic: 1.5,
                cp_crossflow_local: 0.0,
                cp_axial_shift_m: -4.5,
                rot_damping_scale: 1.0,
                srp_drag_loss_gain: 0.8,
            },
        )
        .unwrap();

    // 3. Upper-atmosphere rarefied stage at 105 km -> Knudsen number enters transition regime!
    let b_kn = dynamic_stage(&mut w, Vec3::new(400.0, 105_000.0, 0.0), 120_000.0, Vec3::splat(2e7));
    w.scene
        .set_body_velocity(b_kn, SpatialVector::new(Vec3::new(0.0, -3_000.0, 0.0), Vec3::ZERO))
        .unwrap();
    let ab_kn = w
        .devices
        .rocket_aero
        .insert_body(
            &w.scene,
            AeroBodySpec {
                body: b_kn,
                axis_local: Vec3::Y,
                diameter_m: 9.0,
                length_m: 50.0,
                cd_axial_nose: 0.30,
                cd_axial_tail: 0.90,
                cd_crossflow_subsonic: 1.2,
                cd_crossflow_supersonic: 1.5,
                cp_crossflow_local: 0.0,
                cp_axial_shift_m: -4.0,
                rot_damping_scale: 1.0,
                srp_drag_loss_gain: 0.0,
            },
        )
        .unwrap();

    let mut peak_buffet = 0.0_f64;
    for _ in 0..80 {
        w.step(0.01).unwrap();
        peak_buffet = peak_buffet.max(w.devices.rocket_aero.body_reading(ab_buf).unwrap().buffet_torque_nm);
    }
    let r_pifs = w.devices.rocket_aero.body_reading(ab_pifs).unwrap();
    let r_kn = w.devices.rocket_aero.body_reading(ab_kn).unwrap();
    assert!(
        r_pifs.pifs_cp_shift_m > 1.0,
        "underexpanded supersonic ascent must shift CP forward via PIFS: {:.2} m",
        r_pifs.pifs_cp_shift_m
    );
    assert!(
        peak_buffet > 20_000.0,
        "transonic Max-Q must produce real buffet torque: {peak_buffet:.1} N·m"
    );
    assert!(
        r_kn.knudsen_number > 0.01,
        "105 km flight must enter rarefied Knudsen transition regime: Kn={:.5}",
        r_kn.knudsen_number
    );
}

#[test]
fn mars_co2_atmosphere_and_lunar_vacuum_propulsion_and_aero() {
    let m0 = mars_atmosphere(0.0);
    assert!((m0.pressure_pa - 699.0).abs() < 1.0);
    assert!(m0.density_kg_m3 > 0.012 && m0.density_kg_m3 < 0.018);
    assert!(m0.speed_of_sound_mps > 220.0 && m0.speed_of_sound_mps < 260.0);

    // 1. Mars surface RVac firing: ambient pressure is only 699 Pa -> NO Schmucker flow separation!
    let mut wm = RigidWorld::new().unwrap();
    wm.scene.set_gravity(Vec3::new(0.0, -MARS_GRAVITY, 0.0)).unwrap();
    wm.devices.rockets.set_flat_ground(Some(0.0), 0.0);
    wm.devices.rockets.set_atmosphere(PlanetAtmosphere::MarsCospar);
    let ship_m = dynamic_stage(&mut wm, Vec3::new(0.0, 200.0, 0.0), 120_000.0, Vec3::splat(2e7));
    let up_frame = Transform::new(Vec3::new(0.0, -24.0, 0.0), Quat::from_rotation_z(FRAC_PI_2));
    let rvac = wm
        .devices
        .rockets
        .insert_engine(&wm.scene, RocketEngineSpec::raptor3_vacuum(ship_m, up_frame, None, None))
        .unwrap();
    wm.devices
        .rockets
        .command_engine(rvac, RocketEngineCommand { ignited: true, throttle: 1.0, gimbal_rad: [0.0, 0.0] })
        .unwrap();
    for _ in 0..150 {
        wm.step(0.01).unwrap();
    }
    let rm = wm.devices.rockets.engine_reading(rvac).unwrap();
    assert!(!rm.flow_separated, "RVac must NOT flow-separate in thin 699 Pa Mars CO2!");
    assert!(rm.effective_isp_s > 378.0, "RVac on Mars should retain near-vacuum Isp: {:.1}", rm.effective_isp_s);

    // 2. Lunar Vacuum: zero aero drag & pure vacuum Isp for HLS landing thruster!
    let mut wl = RigidWorld::new().unwrap();
    wl.scene.set_gravity(Vec3::new(0.0, -MOON_GRAVITY, 0.0)).unwrap();
    wl.devices.rockets.set_flat_ground(Some(0.0), 0.0);
    wl.devices.rockets.set_atmosphere(PlanetAtmosphere::Vacuum);
    let ship_l = dynamic_stage(&mut wl, Vec3::new(0.0, 500.0, 0.0), 120_000.0, Vec3::splat(2e7));
    wl.scene
        .set_body_velocity(ship_l, SpatialVector::new(Vec3::new(0.0, -200.0, 0.0), Vec3::ZERO))
        .unwrap();
    let ab_l = wl
        .devices
        .rocket_aero
        .insert_body(
            &wl.scene,
            AeroBodySpec {
                body: ship_l,
                axis_local: Vec3::Y,
                diameter_m: 9.0,
                length_m: 50.0,
                cd_axial_nose: 0.30,
                cd_axial_tail: 0.90,
                cd_crossflow_subsonic: 1.2,
                cd_crossflow_supersonic: 1.5,
                cp_crossflow_local: 0.0,
                cp_axial_shift_m: -4.0,
                rot_damping_scale: 1.0,
                srp_drag_loss_gain: 0.0,
            },
        )
        .unwrap();
    let hls_thr = wl
        .devices
        .rockets
        .insert_engine(&wl.scene, RocketEngineSpec::hls_lunar_landing_thruster(ship_l, up_frame, None, None))
        .unwrap();
    wl.devices
        .rockets
        .command_engine(hls_thr, RocketEngineCommand { ignited: true, throttle: 0.5, gimbal_rad: [0.0, 0.0] })
        .unwrap();
    for _ in 0..100 {
        wl.step(0.01).unwrap();
    }
    let ra_l = wl.devices.rocket_aero.body_reading(ab_l).unwrap();
    let rt_l = wl.devices.rockets.engine_reading(hls_thr).unwrap();
    assert!(ra_l.world_force.length() < 1e-9, "Lunar vacuum must produce strictly zero aero drag");
    assert!((rt_l.effective_isp_s - 335.0).abs() < 0.5, "HLS thruster must run at full 335 s vacuum Isp");
}

#[test]
fn open_loop_offcenter_and_tilted_ignition_spins_without_tvc() {
    // Proves ZERO hidden stabilization in molla-solvers:
    // 1) Igniting an off-center Raptor (r = 1.1 m) with locked gimbal=0° spins & tumbles the stage
    //    at exact Newton-Euler α = (r × F) / I.
    // 2) Pure axial flight (aoa = 0, v = 250 m/s) has zero spurious linear crossflow spin brake.
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();
    w.devices.rockets.set_flat_ground(Some(0.0), 0.0);
    w.devices.rockets.set_atmosphere(PlanetAtmosphere::Vacuum);

    let i_trans = 2.6e7;
    let ship = dynamic_stage(&mut w, Vec3::new(0.0, 1_000.0, 0.0), 120_000.0, Vec3::new(i_trans, 1.5e6, i_trans));
    // Start tilted 45° with zero spin!
    w.scene
        .set_body_pose(ship, Transform::new(Vec3::new(0.0, 1_000.0, 0.0), Quat::from_rotation_z(0.785)))
        .unwrap();
    // Single off-center Raptor at x = +1.10 m, y = -24.2 m, gimbal LOCKED at 0°!
    let frame = Transform::new(Vec3::new(1.10, -24.2, 0.0), Quat::from_rotation_z(FRAC_PI_2));
    let eng = w
        .devices
        .rockets
        .insert_engine(&w.scene, RocketEngineSpec::raptor3_sea_level(ship, frame, None, None, 0.26))
        .unwrap();
    w.devices
        .rockets
        .command_engine(eng, RocketEngineCommand { ignited: true, throttle: 0.90, gimbal_rad: [0.0, 0.0] })
        .unwrap();

    for _ in 0..200 {
        w.step(0.01).unwrap();
    }
    let vel = w.scene.body_velocity(ship).unwrap();
    let pose = w.scene.body_pose(ship).unwrap();
    let tilt_deg = (pose.rotation * Vec3::Y).y.clamp(-1.0, 1.0).acos().to_degrees();
    // Exact Newton-Euler: τ_z ≈ +1.10 * (0.90 * 2.746e6) ≈ +2.72e6 N·m -> α ≈ 0.104 rad/s² -> ω ≈ +0.19 rad/s!
    assert!(
        vel.angular.z > 0.15,
        "off-center open-loop Raptor MUST spin stage at exact Newton-Euler rate: wz={:.4}",
        vel.angular.z
    );
    assert!(
        tilt_deg > 52.0,
        "open-loop tilted rocket must spin further off vertical (never self-straighten!): tilt={tilt_deg:.1}°"
    );

    // Now verify pure axial flight at 250 m/s in Earth air does NOT falsely molasses-brake a small spin!
    let mut wa = RigidWorld::new().unwrap();
    wa.scene.set_gravity(Vec3::ZERO).unwrap();
    wa.devices.rockets.set_flat_ground(Some(0.0), 0.0);
    wa.devices.rockets.set_atmosphere(PlanetAtmosphere::EarthUs76);
    let b = dynamic_stage(&mut wa, Vec3::new(0.0, 500.0, 0.0), 120_000.0, Vec3::new(i_trans, 1.5e6, i_trans));
    wa.scene
        .set_body_velocity(b, SpatialVector::new(Vec3::new(0.0, 250.0, 0.0), Vec3::new(0.0, 0.0, 0.20)))
        .unwrap();
    wa.devices
        .rocket_aero
        .insert_body(
            &wa.scene,
            AeroBodySpec {
                body: b,
                axis_local: Vec3::Y,
                diameter_m: 9.0,
                length_m: 50.0,
                cd_axial_nose: 0.30,
                cd_axial_tail: 0.90,
                cd_crossflow_subsonic: 1.22,
                cd_crossflow_supersonic: 1.54,
                cp_crossflow_local: 0.0,
                cp_axial_shift_m: 0.0,
                rot_damping_scale: 1.0,
                srp_drag_loss_gain: 0.0,
            },
        )
        .unwrap();
    wa.step(0.05).unwrap();
    let wz_after = wa.scene.body_velocity(b).unwrap().angular.z;
    assert!(
        wz_after > 0.185,
        "pure axial flight must NOT apply huge false linear damping: wz_after={wz_after:.4}"
    );
}

#[test]
fn nose_first_aerodynamic_instability_and_rotational_slosh_coupling() {
    // 1. Finless rocket flying nose-first at AoA = 12° with positive cp_axial_shift_m (+9 m ahead of COM)
    //    MUST experience destabilizing pitch-up torque (increases tilt, NEVER self-straightens!).
    let mut wa = RigidWorld::new().unwrap();
    wa.scene.set_gravity(Vec3::ZERO).unwrap();
    wa.devices.rockets.set_flat_ground(Some(0.0), 0.0);
    wa.devices.rockets.set_atmosphere(PlanetAtmosphere::EarthUs76);
    let b = dynamic_stage(&mut wa, Vec3::new(0.0, 2_000.0, 0.0), 120_000.0, Vec3::new(2.6e7, 1.5e6, 2.6e7));
    // Tilt +12° around -Z (nose leaned toward +X), flying straight UP (+Y) at 220 m/s!
    wa.scene
        .set_body_pose(b, Transform::new(Vec3::new(0.0, 2_000.0, 0.0), Quat::from_rotation_z(-0.21)))
        .unwrap();
    wa.scene
        .set_body_velocity(b, SpatialVector::new(Vec3::new(0.0, 220.0, 0.0), Vec3::ZERO))
        .unwrap();
    wa.devices
        .rocket_aero
        .insert_body(
            &wa.scene,
            AeroBodySpec {
                body: b,
                axis_local: Vec3::Y,
                diameter_m: 9.0,
                length_m: 50.0,
                cd_axial_nose: 0.30,
                cd_axial_tail: 0.90,
                cd_crossflow_subsonic: 1.22,
                cd_crossflow_supersonic: 1.54,
                cp_crossflow_local: 0.5,
                cp_axial_shift_m: 8.5,
                rot_damping_scale: 1.0,
                srp_drag_loss_gain: 0.0,
            },
        )
        .unwrap();
    for _ in 0..80 {
        wa.step(0.01).unwrap();
    }
    let tilt_after = (wa.scene.body_pose(b).unwrap().rotation * Vec3::Y)
        .y
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees();
    let wz_after = wa.scene.body_velocity(b).unwrap().angular.z;
    assert!(
        tilt_after > 13.0 && wz_after < -0.02,
        "finless nose-first rocket MUST be aerodynamically unstable (pitch further over!): tilt={tilt_after:.2}°, wz={wz_after:.4}"
    );

    // 2. Pure rotation/flip of a stage with an off-COM nose header tank (y = +16 m) in 0-g MUST excite
    //    real tangential/centripetal slosh (`α × r + ω × (ω × r)`) even with zero COM linear acceleration!
    let mut ws = RigidWorld::new().unwrap();
    ws.scene.set_gravity(Vec3::ZERO).unwrap();
    let s = dynamic_stage(&mut ws, Vec3::ZERO, 120_000.0, Vec3::new(2.6e7, 1.5e6, 2.6e7));
    ws.devices
        .rockets
        .register_stage_dry_mass(
            &ws.scene,
            s,
            MassProperties {
                mass: 120_000.0,
                local_com: Vec3::new(0.0, -3.5, 0.0),
                inertia: Mat3::from_diagonal(Vec3::new(2.6e7, 1.5e6, 2.6e7)),
            },
        )
        .unwrap();
    let nose_tank = ws
        .devices
        .rockets
        .insert_tank(
            &mut ws.scene,
            PropellantTankSpec {
                body: s,
                base_local: Vec3::new(0.0, 14.5, 0.0),
                axis_local: Vec3::Y,
                radius_m: 1.85,
                height_m: 4.5,
                liquid_density_kg_m3: 1_140.0,
                initial_mass_kg: 28_000.0,
                slosh_mass_fraction: 0.25,
                slosh_damping_ratio: 0.05,
                capillary_omega2: 0.35,
            },
        )
        .unwrap();
    // Apply a pure angular spin-up torque (via off-COM step angular velocity change) to mimic flip!
    ws.scene
        .set_body_velocity(s, SpatialVector::new(Vec3::ZERO, Vec3::new(0.0, 0.0, 0.8)))
        .unwrap();
    for _ in 0..40 {
        ws.step(0.01).unwrap();
    }
    let tr = ws.devices.rockets.tank_reading(nose_tank).unwrap();
    assert!(
        tr.slosh_disp_m.length() > 0.02,
        "flipping/rotating nose tank MUST excite real rotational slosh: disp={:?}",
        tr.slosh_disp_m
    );
}

#[test]
fn roll_vectoring_produces_pure_roll_and_free_free_bending_conserves_momentum() {
    // 1. Tangential roll vectoring (`g0 = -r sin a, g1 = +r cos a`) on a 3-Raptor cluster
    //    must generate pure roll torque `+tau_y` with ZERO net lateral force!
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();
    w.devices.rockets.set_atmosphere(PlanetAtmosphere::Vacuum);
    let stage = dynamic_stage(&mut w, Vec3::ZERO, 120_000.0, Vec3::new(2.6e7, 1.5e6, 2.6e7));
    let up_rot = Quat::from_rotation_z(FRAC_PI_2);
    let roll_rad = 0.10;
    for k in 0..3 {
        let a = (k as f64) * (std::f64::consts::TAU / 3.0);
        let frame = Transform::new(Vec3::new(1.10 * a.cos(), -24.0, 1.10 * a.sin()), up_rot);
        let eid = w
            .devices
            .rockets
            .insert_engine(&w.scene, RocketEngineSpec::raptor3_sea_level(stage, frame, None, None, 0.26))
            .unwrap();
        w.devices
            .rockets
            .command_engine(
                eid,
                RocketEngineCommand {
                    ignited: true,
                    throttle: 0.80,
                    gimbal_rad: [-roll_rad * a.sin(), roll_rad * a.cos()],
                },
            )
            .unwrap();
    }
    for _ in 0..100 {
        w.step(0.01).unwrap();
    }
    let v = w.scene.body_velocity(stage).unwrap();
    assert!(
        v.angular.y > 0.35,
        "tangential roll vectoring must spin stage strongly about +Y: wy={:.4}",
        v.angular.y
    );
    // Finite-angle 3D gimbal kinematics (`sin(g) - 2 sin(g/2) = O(g³)`) keeps lateral drift < 0.02% of axial Δv!
    assert!(
        v.linear.x.abs() < 0.01 && v.linear.z.abs() < 0.01,
        "symmetric 3-engine roll vectoring must produce negligible lateral drift: v={:?}",
        v.linear
    );

    // 2. Free-free Euler-Bernoulli bending modes must strictly conserve rigid-body linear momentum in 0-g!
    let mut wf = RigidWorld::new().unwrap();
    wf.scene.set_gravity(Vec3::ZERO).unwrap();
    let fb = dynamic_stage(&mut wf, Vec3::ZERO, 120_000.0, Vec3::new(2.6e7, 1.5e6, 2.6e7));
    wf.devices.rocket_structure.insert(&wf.scene, FlexBodySpec::starship_upper(fb)).unwrap();
    for _ in 0..50 {
        wf.step(0.01).unwrap();
    }
    let vf = wf.scene.body_velocity(fb).unwrap();
    assert!(
        vf.linear.length() < 1e-12 && vf.angular.length() < 1e-12,
        "free-free structure with no external forces must conserve zero momentum: {vf:?}"
    );
}

#[test]
fn spherical_planet_corotating_atmosphere_gives_zero_pad_wind_and_inertial_ascent() {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();
    let r_earth = 6_378_137.0;
    let omega = Vec3::new(0.0, molla_solvers::devices::EARTH_OMEGA_RAD_S, 0.0); // Spinning around +Y North Pole
    w.devices.rockets.set_spherical_planet(Vec3::ZERO, r_earth);
    w.devices.rockets.set_earth_rotation(omega);
    w.devices.rockets.set_atmosphere(PlanetAtmosphere::EarthUs76);

    // Rocket parked on the equator at (+R_earth, 0, 0) moving at exact co-rotating surface velocity Ω × r (~465.1 m/s)!
    let pad_pos = Vec3::new(r_earth, 0.0, 0.0);
    let v_pad = omega.cross(pad_pos);
    let pad_ship = dynamic_stage(&mut w, pad_pos, 120_000.0, Vec3::splat(2.5e7));
    w.scene
        .set_body_velocity(pad_ship, SpatialVector::new(v_pad, Vec3::ZERO))
        .unwrap();
    w.devices
        .rockets
        .register_stage_dry_mass(
            &w.scene,
            pad_ship,
            MassProperties {
                mass: 120_000.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::from_diagonal(Vec3::splat(2.5e7)),
            },
        )
        .unwrap();
    let ab_pad = w
        .devices
        .rocket_aero
        .insert_body(
            &w.scene,
            AeroBodySpec {
                body: pad_ship,
                axis_local: Vec3::X, // Nose pointing radially outward (+X)
                diameter_m: 9.0,
                length_m: 50.0,
                cd_axial_nose: 0.30,
                cd_axial_tail: 0.90,
                cd_crossflow_subsonic: 1.22,
                cd_crossflow_supersonic: 1.54,
                cp_crossflow_local: 0.0,
                cp_axial_shift_m: 8.5,
                rot_damping_scale: 1.0,
                srp_drag_loss_gain: 0.0,
            },
        )
        .unwrap();

    // Second rocket climbing radially at +100 m/s relative to the co-rotating atmosphere!
    let climb_ship = dynamic_stage(&mut w, pad_pos + Vec3::new(50.0, 0.0, 0.0), 120_000.0, Vec3::splat(2.5e7));
    w.scene
        .set_body_velocity(climb_ship, SpatialVector::new(v_pad + Vec3::new(100.0, 0.0, 0.0), Vec3::ZERO))
        .unwrap();
    let ab_climb = w
        .devices
        .rocket_aero
        .insert_body(
            &w.scene,
            AeroBodySpec {
                body: climb_ship,
                axis_local: Vec3::X,
                diameter_m: 9.0,
                length_m: 50.0,
                cd_axial_nose: 0.30,
                cd_axial_tail: 0.90,
                cd_crossflow_subsonic: 1.22,
                cd_crossflow_supersonic: 1.54,
                cp_crossflow_local: 0.0,
                cp_axial_shift_m: 8.5,
                rot_damping_scale: 1.0,
                srp_drag_loss_gain: 0.0,
            },
        )
        .unwrap();

    w.step(0.01).unwrap();
    let r_pad = w.devices.rocket_aero.body_reading(ab_pad).unwrap();
    let r_climb = w.devices.rocket_aero.body_reading(ab_climb).unwrap();
    let v_pad_after = w.scene.body_velocity(pad_ship).unwrap().linear;
    assert!(
        r_pad.world_force.length() < 1e-6 && r_pad.dynamic_pressure_pa < 1e-6,
        "rocket co-rotating on spinning round Earth pad must feel zero wind: q={:.3}, f={:?}",
        r_pad.dynamic_pressure_pa,
        r_pad.world_force
    );
    assert!(
        (v_pad_after - v_pad).length() < 1e-6,
        "inertial spherical-planet mode must NOT inject fictitious flat-frame Coriolis force!"
    );
    let q_expected = 0.5 * us76_atmosphere(50.0).density_kg_m3 * 100.0 * 100.0;
    assert!(
        (r_climb.dynamic_pressure_pa - q_expected).abs() / q_expected < 0.02,
        "climbing rocket q must match relative 100 m/s airspeed ({q_expected:.1} Pa), got {:.1} Pa",
        r_climb.dynamic_pressure_pa
    );
}
