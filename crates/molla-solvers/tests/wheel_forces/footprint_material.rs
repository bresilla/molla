use super::*;
use molla_math::Quat;

#[test]
fn shoulder_friction_difference_produces_a_real_articulated_yaw_moment() {
    let (mut world, chassis, wheel, spin, ground) = fixture(false);
    let mut config = pressure_config();
    config.tire.width_segments = 4;
    world.wheels.configure_pressure_tire(wheel, config).unwrap();
    world
        .wheels
        .register_ground(
            &world.scene,
            ground,
            Some(FrictionGrid {
                origin: [-10.0, -5.0],
                cell_size: [20.0, 5.0],
                cols: 1,
                rows: 2,
                values: vec![0.0, 1.0],
            }),
        )
        .unwrap();
    let offset = world.scene.model().joint_dof_offset.host().unwrap()
        [world.scene.joint_index(spin).unwrap().index()] as usize;
    world.scene.state_mut().joint_qd.host_mut().unwrap()[offset] = 10.0;
    world.solver.substeps = 1;
    world.step(0.0001).unwrap();
    let sample = &world.wheels.samples(&world.scene)[0];
    let pressure = sample.pressure.as_ref().unwrap();
    let mut torque = 0.0;
    for cell in &pressure.cells {
        if cell.local_z < 0.0 {
            assert_eq!(cell.tangential_force, 0.0);
        } else {
            assert!(cell.tangential_force > 0.0);
        }
        torque += cell.local_z * cell.tangential_force - cell.local_x * cell.lateral_force;
    }
    assert!(torque > 0.0);
    assert!((sample.output.aligning_moment - torque).abs() < 1e-10);
    assert!(world.scene.body_velocity(chassis).unwrap().angular.y > 0.0);
    assert!(world.contact_pairs().is_empty());
    for width_segments in [0, 17, usize::MAX] {
        config.tire.width_segments = width_segments;
        assert!(world.wheels.configure_pressure_tire(wheel, config).is_err());
        assert_eq!(
            world
                .wheels
                .pressure_tire(&world.scene, wheel)
                .unwrap()
                .config
                .tire
                .width_segments,
            4
        );
    }
}

#[test]
fn pressure_patch_samples_friction_on_both_sides_of_a_material_boundary() {
    let (mut world, _, wheel, spin, ground) = fixture(true);
    world
        .wheels
        .configure_pressure_tire(wheel, pressure_config())
        .unwrap();
    world
        .wheels
        .register_ground(
            &world.scene,
            ground,
            Some(FrictionGrid {
                origin: [-10.0, -5.0],
                cell_size: [10.0, 10.0],
                cols: 2,
                rows: 1,
                values: vec![0.0, 1.0],
            }),
        )
        .unwrap();
    let offset = world.scene.model().joint_dof_offset.host().unwrap()
        [world.scene.joint_index(spin).unwrap().index()] as usize;
    world.scene.state_mut().joint_qd.host_mut().unwrap()[offset] = 10.0;
    world.solver.substeps = 1;
    world.step(0.0001).unwrap();
    let sample = &world.wheels.samples(&world.scene)[0];
    let pressure = sample.pressure.as_ref().unwrap();
    let mut low = 0;
    let mut high = 0;
    let mut expected_grip = 0.0;
    for cell in &pressure.cells {
        if cell.local_x < 0.0 {
            low += 1;
            assert_eq!(cell.tangential_force, 0.0);
            assert_eq!(cell.lateral_force, 0.0);
        } else {
            high += 1;
            assert!(cell.tangential_force.abs() > 0.0);
            let load_ratio = (cell.normal_load / cell.area / 220_000.0).clamp(0.15, 6.0);
            expected_grip += load_ratio.powf(-0.07).clamp(0.72, 1.12) * cell.normal_load;
        }
    }
    assert!(low > 0 && high > 0);
    assert!((pressure.grip_force - expected_grip).abs() < 1e-8);
}

#[test]
fn tilted_translated_ground_uses_collider_local_cells_and_outside_grid_fallback() {
    for outside in [false, true] {
        let (mut world, chassis, wheel, spin, ground) = fixture(false);
        world
            .wheels
            .configure_pressure_tire(wheel, pressure_config())
            .unwrap();
        let rotation = Quat::from_rotation_y(0.7) * Quat::from_rotation_z(0.2);
        let ground_pose = Transform {
            position: Vec3::new(2.0, 1.0, -3.0),
            rotation,
        };
        world.scene.set_collider_pose(ground, ground_pose).unwrap();
        let original = Transform {
            position: ground_pose.transform_point(Vec3::Y * 0.49),
            rotation,
        };
        world.scene.set_body_pose(chassis, original).unwrap();
        let fallback = world.scene.collider(ground).unwrap().friction;
        let grid = if outside {
            FrictionGrid {
                origin: [-0.02, -5.0],
                cell_size: [0.04, 10.0],
                cols: 1,
                rows: 1,
                values: vec![0.0],
            }
        } else {
            FrictionGrid {
                origin: [-10.0, -5.0],
                cell_size: [10.01, 10.0],
                cols: 2,
                rows: 1,
                values: vec![0.0, 4.0],
            }
        };
        world
            .wheels
            .register_ground(&world.scene, ground, Some(grid))
            .unwrap();
        let offset = world.scene.model().joint_dof_offset.host().unwrap()
            [world.scene.joint_index(spin).unwrap().index()] as usize;
        world.scene.state_mut().joint_qd.host_mut().unwrap()[offset] = 10.0;
        world.solver.substeps = 1;
        world.step(0.0001).unwrap();
        for sleeping in [false, true] {
            if sleeping {
                world.sleep_body(chassis).unwrap();
                let mut moved = original;
                moved.position += rotation * Vec3::X * 0.002;
                world
                    .scene
                    .set_body_pose_with_wake(chassis, moved, false)
                    .unwrap();
                world.step(0.0001).unwrap();
                assert!(world.is_sleeping(chassis));
            }
            let sample = world.wheels.samples(&world.scene).last().unwrap();
            let pressure = sample.pressure.as_ref().unwrap();
            let mut expected = 0.0;
            let mut zero = 0;
            let mut positive = 0;
            for cell in &pressure.cells {
                let x = cell.local_x + if sleeping { 0.002 } else { 0.0 };
                let mu = if outside {
                    if (-0.02..0.02).contains(&x) {
                        0.0
                    } else {
                        fallback
                    }
                } else if x < 0.01 {
                    0.0
                } else {
                    4.0
                };
                if mu == 0.0 {
                    zero += 1;
                    assert_eq!(cell.tangential_force, 0.0);
                    assert_eq!(cell.lateral_force, 0.0);
                } else {
                    positive += 1;
                }
                let load_ratio = (cell.normal_load / cell.area / 220_000.0).clamp(0.15, 6.0);
                expected += mu * load_ratio.powf(-0.07).clamp(0.72, 1.12) * cell.normal_load;
            }
            assert!(
                zero > 0 && positive > 0,
                "outside={outside} sleeping={sleeping} cells={:?}",
                pressure.cells
            );
            assert!(
                (pressure.grip_force - expected).abs() < 1e-8,
                "{} != {expected}",
                pressure.grip_force
            );
            assert_eq!(sample.dt == 0.0, sleeping);
        }
    }
}

#[test]
fn uniform_grid_preserves_scalar_surface_dynamics_and_invalid_replacement_is_atomic() {
    for mu in [0.0, 0.2, 1.2] {
        let mut results = Vec::new();
        for grid in [false, true] {
            let (mut world, chassis, wheel, spin, ground) = fixture(true);
            world
                .wheels
                .configure_pressure_tire(wheel, pressure_config())
                .unwrap();
            let mut material = world.scene.collider(ground).unwrap().clone();
            material.friction = mu;
            world.scene.replace_collider(ground, material).unwrap();
            if grid {
                world
                    .wheels
                    .register_ground(
                        &world.scene,
                        ground,
                        Some(FrictionGrid {
                            origin: [-10.0, -5.0],
                            cell_size: [20.0, 10.0],
                            cols: 1,
                            rows: 1,
                            values: vec![mu],
                        }),
                    )
                    .unwrap();
            }
            let offset = world.scene.model().joint_dof_offset.host().unwrap()
                [world.scene.joint_index(spin).unwrap().index()] as usize;
            world.scene.state_mut().joint_qd.host_mut().unwrap()[offset] = 10.0;
            for _ in 0..4 {
                world.step(0.001).unwrap();
            }
            let before = world.wheels.samples(&world.scene).last().unwrap().clone();
            assert!(
                world
                    .wheels
                    .register_ground(
                        &world.scene,
                        ground,
                        Some(FrictionGrid {
                            origin: [-10.0, -5.0],
                            cell_size: [20.0, 10.0],
                            cols: 1,
                            rows: 1,
                            values: vec![f64::NAN],
                        })
                    )
                    .is_err()
            );
            assert_eq!(
                world
                    .wheels
                    .samples(&world.scene)
                    .last()
                    .unwrap()
                    .output
                    .force,
                before.output.force
            );
            world.step(0.001).unwrap();
            results.push((
                world.scene.body_velocity(chassis).unwrap(),
                world.wheels.samples(&world.scene).last().unwrap().clone(),
            ));
        }
        assert!((results[0].0.linear - results[1].0.linear).length() < 1e-10);
        assert!(
            (Vec3::from_array(results[0].1.output.force)
                - Vec3::from_array(results[1].1.output.force))
            .length()
                < 1e-8
        );
        assert!(
            (results[0].1.pressure.as_ref().unwrap().grip_force
                - results[1].1.pressure.as_ref().unwrap().grip_force)
                .abs()
                < 1e-8
        );
    }
}
