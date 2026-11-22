use crate::{DT, Tree, gpu_limits, step_config};
use molla_compute::Engine;
use molla_core::Result;
use molla_geometry::convex_cut::CutPlane;
use molla_math::Vec3;
use molla_solvers::strands::{GpuStrands, StepConfig, StrandModel};

pub fn grab_matrix(
    engine: &Engine,
    source: &str,
    budget: usize,
    strong: bool,
    selected: Option<u64>,
    selected_axis: Option<usize>,
    grip_force: f64,
) -> Result<()> {
    let initial = Tree::new(engine, source, budget)?;
    let cells: Vec<_> = initial
        .model
        .fibers()
        .iter()
        .filter(|f| f.pin.is_none())
        .enumerate()
        .filter(|(i, f)| selected.map_or(!strong || i % 6 == 0, |id| f.id == id))
        .map(|(_, f)| f.id)
        .filter(|id| selected.is_none_or(|value| value == *id))
        .collect();
    let axes = [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z];
    let names = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"];
    if cells.is_empty() || selected_axis.is_some_and(|a| a >= axes.len()) {
        return Err(molla_core::Error::Solver(
            "invalid grab matrix selection".into(),
        ));
    }
    let mut csv = String::from(
        "cell,direction,strong,peak_step_m,peak_gap_m,stress_snaps,recoveries,pieces,error,grip_force_n\n",
    );
    let mut failures = Vec::new();
    let start = std::time::Instant::now();
    for (index, &id) in cells.iter().enumerate() {
        for (axis, &direction) in axes.iter().enumerate() {
            if selected_axis.is_some_and(|a| a != axis) {
                continue;
            }
            let mut tree = Tree::new(engine, source, budget)?;
            tree.grip_force = grip_force;
            let f = tree.model.fiber(id).unwrap();
            let face = &f.shape.faces()[axis % f.shape.faces().len()];
            let local = face
                .vertices
                .iter()
                .map(|&i| f.shape.vertices()[i as usize])
                .sum::<Vec3>()
                / face.vertices.len() as f64;
            let hit = f.motion.pose.transform_point(local);
            let root = tree.model.fibers()[0].motion.pose;
            let mass: f64 = tree.model.fibers().iter().map(|f| f.mass.mass).sum();
            tree.start_grab(id, hit);
            let mut peak_step: f64 = 0.0;
            let mut peak_gap: f64 = 0.0;
            let loaded = if strong { 300 } else { 120 };
            let outcome = (|| -> Result<()> {
                for step in 0..loaded + 30 {
                    if step < loaded {
                        let t = step as f64 / (loaded - 1) as f64;
                        let displacement = if strong {
                            if axis % 2 == 0 {
                                t
                            } else if t < 0.6 {
                                0.65
                            } else {
                                -0.35
                            }
                        } else {
                            0.025 * (1.0 - (std::f64::consts::TAU * t).cos()) * 0.5
                        };
                        tree.grab.as_mut().unwrap().target = hit + direction * displacement;
                    } else if step == loaded {
                        tree.release_grab();
                    }
                    let previous = tree.model.clone();
                    tree.advance(engine, 2)?;
                    if selected.is_some()
                        && (step == loaded - 1 || step == loaded || step == loaded + 1)
                    {
                        let f = tree.model.fiber(id).unwrap();
                        let kinetic: f64 = tree
                            .model
                            .fibers()
                            .iter()
                            .map(|f| f.motion.momentum(f.mass).unwrap().2)
                            .sum();
                        let elastic: f64 = tree
                            .model
                            .bonds()
                            .iter()
                            .filter(|b| b.intact())
                            .map(|b| {
                                let [a, c] = b.fibers.map(|i| tree.model.fiber(i).unwrap());
                                molla_solvers::strands::bond_rows(b, a, c)
                                    .iter()
                                    .map(|r| 0.5 * r.value * r.value / r.compliance)
                                    .sum::<f64>()
                            })
                            .sum();
                        eprintln!(
                            "RELEASE step={step} point={:?} original={hit:?} handle={:?} velocity={:?} KE={kinetic} elastic={elastic}",
                            f.motion.pose.transform_point(local),
                            tree.handle.map(|h| h.point),
                            f.motion.linear_velocity
                        );
                    }
                    for a in previous.fibers() {
                        if let Some(b) = tree.model.fiber(a.id) {
                            if a.center().distance(b.center()) > if strong { 0.08 } else { 0.02 } {
                                eprintln!(
                                    "JUMP at t={} cell {} mass={} v={} omega={} grabbed={:?} snapped={}",
                                    tree.time,
                                    b.id,
                                    b.mass.mass,
                                    format!(
                                        "{:?}, before={:?}, at={:?}",
                                        b.motion.linear_velocity,
                                        a.motion.linear_velocity,
                                        b.center()
                                    ),
                                    b.motion.angular_velocity.length(),
                                    tree.grab.map(|g| g.fiber),
                                    tree.stress_breaks
                                );
                            }
                            peak_step = peak_step.max(a.center().distance(b.center()));
                        }
                    }
                    for bond in tree.model.bonds().iter().filter(|b| b.intact()) {
                        let [a, b] = bond.fibers.map(|id| tree.model.fiber(id).unwrap());
                        for row in &molla_solvers::strands::bond_rows(bond, a, b)[..3] {
                            peak_gap = peak_gap.max(row.value.abs());
                        }
                    }
                    if !strong && tree.stress_breaks > 0 {
                        return Err(molla_core::Error::Solver(
                            "gentle loading fractured wood".into(),
                        ));
                    }
                    if peak_step > if strong { 0.08 } else { 0.02 } {
                        return Err(molla_core::Error::Solver(format!(
                            "sudden motion {peak_step} m/batch"
                        )));
                    }
                }
                let final_mass: f64 = tree.model.fibers().iter().map(|f| f.mass.mass).sum();
                if (mass - final_mass).abs() > 1e-8
                    || tree.model.fibers()[0].motion.pose != root
                    || tree.clearance() < -0.01
                {
                    return Err(molla_core::Error::Solver(
                        "mass/root/floor invariant".into(),
                    ));
                }
                if tree.grab.is_some() || tree.handle.is_some() {
                    return Err(molla_core::Error::Solver("grab did not release".into()));
                }
                Ok(())
            })();
            let error = outcome.err().map(|e| e.to_string()).unwrap_or_default();
            if !error.is_empty() {
                eprintln!("FAIL matrix cell {id} {}: {error}", names[axis]);
                failures.push(format!("{id}/{}: {error}", names[axis]));
            }
            csv.push_str(&format!(
                "{id},{},{strong},{peak_step:.8},{peak_gap:.8},{},{},{},{},{grip_force}\n",
                names[axis],
                tree.stress_breaks,
                tree.recovered_batches,
                tree.model.components().len(),
                error.replace(',', ";")
            ));
        }
        eprintln!(
            "Grab matrix: {}/{} cells, six directions each, {} failures",
            index + 1,
            cells.len(),
            failures.len()
        );
    }
    let path = if strong {
        "/tmp/molla-grab-matrix-strong.csv"
    } else {
        "/tmp/molla-grab-matrix-gentle.csv"
    };
    let path = if selected.is_some() {
        "/tmp/molla-grab-matrix-repro.csv"
    } else {
        path
    };
    std::fs::write(path, csv).map_err(|e| molla_core::Error::Solver(e.to_string()))?;
    eprintln!(
        "Grab matrix: {} cases, {} failures, {:.2} wall seconds; {path}",
        cells.len() * if selected_axis.is_some() { 1 } else { 6 },
        failures.len(),
        start.elapsed().as_secs_f64()
    );
    if !failures.is_empty() {
        return Err(molla_core::Error::Solver(failures.join("\n")));
    }
    Ok(())
}

pub fn tools(engine: &Engine, source: &str, budget: usize) -> Result<()> {
    let mut rough = Tree::new(engine, source, budget)?;
    let f = rough.model.fibers()[rough.model.fibers().len() / 2].clone();
    let before = rough.model.fibers().len();
    let mass: f64 = rough.model.fibers().iter().map(|f| f.mass.mass).sum();
    rough.section(
        engine,
        CutPlane::new(f.center(), f.motion.pose.rotation * Vec3::Z)?,
        &[f.id],
        true,
    )?;
    assert_eq!(rough.model.fibers().len(), before + 7);
    assert_eq!(rough.model.components().len(), 2);
    assert!(
        (rough
            .model
            .fibers()
            .iter()
            .map(|f| f.mass.mass)
            .sum::<f64>()
            - mass)
            .abs()
            < 1e-9
    );
    let now = std::time::Instant::now();
    for _ in 0..1200 {
        rough.advance(engine, 4)?;
    }
    assert_eq!(rough.model.components().len(), 2);
    assert_eq!(rough.stress_breaks, 0);
    assert!(rough.clearance() > -0.01);
    eprintln!(
        "PASS splintered click: eight local cells, two fragments, conserved mass, floor and 20 s stress in {:.2} wall seconds",
        now.elapsed().as_secs_f64()
    );
    grab_motion(engine, source, budget)
}

pub fn grab_motion(engine: &Engine, source: &str, budget: usize) -> Result<()> {
    let mut tree = Tree::new(engine, source, budget)?;
    let tip = tree.model.fibers().last().unwrap().clone();
    let anchor = tip.center();
    let root = tree.model.fibers()[0].motion.pose;
    tree.start_grab(tip.id, anchor);
    let mut peak_jump: f64 = 0.0;
    let mut peak_gap: f64 = 0.0;
    let mut gap_location = (0, 0, 0.0);
    for i in 0..480 {
        let displacement = if i < 120 {
            0.025 * i as f64 / 119.0
        } else if i < 360 {
            0.025 * ((i - 120) as f64 / 239.0 * std::f64::consts::TAU).cos()
        } else {
            0.025
        };
        tree.grab.as_mut().unwrap().target = anchor + Vec3::X * displacement;
        let previous = tree.model.clone();
        tree.advance(engine, 2)?;
        for (a, b) in previous.fibers().iter().zip(tree.model.fibers()) {
            peak_jump = peak_jump.max(a.center().distance(b.center()));
        }
        for bond in tree.model.bonds().iter().filter(|b| b.intact()) {
            let [a, b] = bond.fibers.map(|id| tree.model.fiber(id).unwrap());
            for row in &molla_solvers::strands::bond_rows(bond, a, b)[..3] {
                if row.value.abs() > peak_gap {
                    gap_location = (i, bond.id, row.compliance);
                }
                peak_gap = peak_gap.max(row.value.abs());
            }
        }
    }
    assert!(
        peak_jump < 0.008,
        "gentle grab jumped {peak_jump} m per batch"
    );
    eprintln!(
        "Grab motion: peak jump {peak_jump}, peak gap {peak_gap}, location {gap_location:?}, stress snaps {}",
        tree.stress_breaks
    );
    assert!(peak_gap < 0.002, "grab opened a joint by {peak_gap} m");
    assert_eq!(
        tree.stress_breaks, 0,
        "a gentle grab must bend without breaking"
    );
    assert_eq!(tree.model.components().len(), 1);
    assert_eq!(tree.model.fibers()[0].motion.pose, root);
    eprintln!(
        "PASS gentle grab/reversal/hold: rooted tree, no fracture; peak movement {peak_jump:.6} m/batch, joint gap {peak_gap:.6} m, {} recoveries",
        tree.recovered_batches
    );
    let mut cpu = tree.model.clone();
    let motion = tree.next_grab(1).unwrap();
    tree.advance(engine, 1)?;
    for substep in 0..tree.last_subdivisions {
        let mut grab = motion.1;
        grab.target = motion.2.lerp(
            grab.target,
            (substep + 1) as f64 / tree.last_subdivisions as f64,
        );
        molla_solvers::strands::step_cpu(
            &mut cpu,
            StepConfig {
                dt: DT / tree.last_subdivisions as f64,
                grab: Some(grab),
                ..step_config()
            },
        )?;
    }
    compare(&tree.model, &cpu, 2e-4);
    for i in 0..480 {
        tree.grab.as_mut().unwrap().target = anchor + Vec3::X * (0.025 + 1.8 * (i as f64 / 479.0));
        tree.advance(engine, 2)?;
        if tree.stress_breaks > 0 {
            break;
        }
    }
    assert!(
        tree.stress_breaks > 0,
        "strong bending must trigger load-driven failure"
    );
    assert!(tree.model.components().len() > 1);
    assert_eq!(
        tree.cuts + tree.breaks,
        0,
        "grab must not invoke the click tools"
    );
    assert!(
        tree.model
            .fibers()
            .iter()
            .any(|f| f.shape.faces().iter().any(|p| matches!(
                p.origin,
                molla_geometry::convex_cut::FaceOrigin::Fracture { .. }
            )))
    );
    let grabbed = tree.grab.unwrap().fiber;
    let height = tree.model.fiber(grabbed).unwrap().center().y;
    tree.grab = None;
    for _ in 0..120 {
        tree.advance(engine, 4)?;
    }
    assert!(
        tree.model.fiber(grabbed).unwrap().center().y < height - 0.01,
        "released piece must fall"
    );
    assert!(tree.clearance() > -0.01);
    eprintln!(
        "PASS strong grab: GPU load-driven failure, physical splinter geometry, release and floor; {} cells, {} pieces",
        tree.model.fibers().len(),
        tree.model.components().len()
    );
    let mut push = Tree::new(engine, source, budget)?;
    let camera = crate::camera();
    let rect = eframe::egui::Rect::from_min_size(
        eframe::egui::Pos2::ZERO,
        eframe::egui::vec2(1280.0, 900.0),
    );
    let pixel = push.visible_limb(&camera, rect);
    push.click(engine, &camera, pixel, rect, false, true)?;
    assert!(push.push_remaining.is_some());
    for _ in 0..120 {
        push.advance(engine, 2)?;
    }
    assert!(push.grab.is_none() && push.handle.is_none() && push.push_remaining.is_none());
    assert_eq!(
        push.stress_breaks, 0,
        "a short push must not tear the branch"
    );
    assert_eq!(push.model.components().len(), 1);
    assert!(push.clearance() > -0.01);
    eprintln!(
        "PASS timed push: applied through the grab solve, released automatically, rooted tree and no fracture"
    );
    Ok(())
}

fn legacy_planar_fixture(
    tree: &mut Tree,
    engine: &Engine,
    plane: CutPlane,
    ids: &[u64],
    fracture: bool,
) -> Result<()> {
    if !fracture {
        return tree.section(engine, plane, ids, false);
    }
    tree.gpu.fracture(
        engine,
        plane,
        &ids.iter().map(|&id| (id, plane)).collect::<Vec<_>>(),
        (tree.cuts + tree.breaks + 1) as u64,
    )?;
    tree.model = tree.gpu.snapshot(engine)?;
    tree.breaks += 1;
    Ok(())
}

fn compare(a: &StrandModel, b: &StrandModel, tolerance: f64) {
    assert_eq!(a.fibers().len(), b.fibers().len());
    assert_eq!(a.bonds().len(), b.bonds().len());
    for (a, b) in a.fibers().iter().zip(b.fibers()) {
        assert_eq!(a.id, b.id);
        assert_eq!(a.shape, b.shape);
        assert_eq!(a.pin, b.pin);
        assert!(
            (a.center() - b.center()).length() < tolerance,
            "fiber {} center error {} tolerance {}; GPU {:?} CPU {:?}",
            a.id,
            a.center().distance(b.center()),
            tolerance,
            a.center(),
            b.center()
        );
        assert!(a.motion.pose.rotation.dot(b.motion.pose.rotation).abs() > 1.0 - tolerance);
        assert!(a.motion.linear_velocity.distance(b.motion.linear_velocity) < tolerance * 1000.0);
    }
}

pub fn recovery(engine: &Engine, source: &str, budget: usize, seconds: f64) -> Result<()> {
    if budget == crate::apple_tree_model::DEFAULT_LIMBS && source == crate::apple_tree_model::SOURCE
    {
        junction_rest(engine, source, budget)?;
    }
    larger_step_recovery(engine, source, budget)?;
    let mut tree = Tree::new(engine, source, budget)?;
    tree.sway(engine)?;
    for _ in 0..30 {
        tree.advance(engine, 4)?;
    }
    for index in [
        tree.model.fibers().len() - 2,
        tree.model.fibers().len() / 2,
        tree.model.fibers().len() / 3,
    ] {
        let f = &tree.model.fibers()[index];
        let id = f.id;
        let plane = CutPlane::new(
            f.center(),
            f.motion.pose.rotation * Vec3::new(0.13, -0.07, 1.0),
        )?;
        let fracture = tree.cuts == 1;
        legacy_planar_fixture(&mut tree, engine, plane, &[id], fracture)?;
        for _ in 0..15 {
            tree.advance(engine, 4)?;
        }
    }
    assert_eq!(tree.model.components().len(), 4);
    let checkpoint = tree.model.clone();
    let start_time = tree.time;
    let counts = (tree.cuts, tree.breaks);
    let mut fail = step_config();
    fail.coupled.as_mut().unwrap().linear_iterations = 1;
    tree.gpu.step(engine, step_config())?;
    tree.gpu.step(engine, fail)?;
    assert_eq!(
        tree.gpu.last_valid_snapshot(engine)?.1,
        4,
        "failure fixture did not reach the convergence guard"
    );
    let before_recoveries = tree.recovered_batches;
    tree.advance(engine, 3)?;
    assert_eq!(tree.recovered_batches, before_recoveries + 1);
    assert_eq!((tree.cuts, tree.breaks), counts);
    assert!((tree.time - start_time - 3.0 * DT).abs() < 1e-12);
    let mut reference = GpuStrands::with_limits(engine, checkpoint.clone(), gpu_limits())?;
    for _ in 0..3 * tree.last_subdivisions {
        reference.step(
            engine,
            StepConfig {
                dt: DT / tree.last_subdivisions as f64,
                ..step_config()
            },
        )?;
    }
    compare(&tree.model, &reference.snapshot(engine)?, 1e-7);
    let mut cpu = checkpoint;
    for _ in 0..3 * tree.last_subdivisions {
        molla_solvers::strands::step_cpu(
            &mut cpu,
            StepConfig {
                dt: DT / tree.last_subdivisions as f64,
                ..step_config()
            },
        )?;
    }
    compare(&tree.model, &cpu, 1e-4);
    eprintln!(
        "PASS recovery: injected sticky PCG failure after a partially advanced batch; exact interval replay, cuts/pins/geometry preserved; fresh GPU and CPU oracles agree"
    );

    let checkpoint = tree.model.clone();
    let start_time = tree.time;
    tree.gpu.step(
        engine,
        StepConfig {
            dt: 1e10,
            gravity: Vec3::splat(1e38),
            iterations: 1,
            air_damping: 0.0,
            ground_height: None,
            ..Default::default()
        },
    )?;
    assert!(
        tree.advance(engine, 2).is_err(),
        "non-convergence faults must not be silently retried"
    );
    assert_eq!(tree.time, start_time);
    compare(&tree.model, &checkpoint, 1e-12);
    compare(&tree.gpu.snapshot(engine)?, &checkpoint, 1e-6);
    tree.advance(engine, 2)?;
    eprintln!(
        "PASS rollback: nonrecoverable fault rejected; healthy checkpoint restored, subsequent step usable"
    );

    let initial_recoveries = tree.recovered_batches;
    let start = std::time::Instant::now();
    let batches = (seconds / (4.0 * DT)).ceil() as usize;
    let root = tree
        .model
        .fibers()
        .iter()
        .find(|f| f.pin.is_some())
        .unwrap()
        .motion
        .pose;
    for batch in 0..batches {
        tree.advance(engine, 4)?;
        if batch % 600 == 599 {
            eprintln!(
                "Stress progress: {:.1} simulated seconds",
                (batch + 1) as f64 * 4.0 * DT
            );
        }
    }
    assert_eq!(
        tree.model
            .fibers()
            .iter()
            .find(|f| f.pin.is_some())
            .unwrap()
            .motion
            .pose,
        root
    );
    assert_eq!(tree.model.components().len(), 4);
    assert_eq!((tree.cuts, tree.breaks), counts);
    assert!(
        tree.clearance() > -0.01,
        "stress floor clearance {}",
        tree.clearance()
    );
    eprintln!(
        "PASS repeated-cut stress: {:.1} simulated seconds / {:.2} wall seconds; {} natural recoveries; four pieces, fixed root, finite motion, floor",
        batches as f64 * 4.0 * DT,
        start.elapsed().as_secs_f64(),
        tree.recovered_batches - initial_recoveries
    );
    Ok(())
}

fn junction_rest(engine: &Engine, source: &str, budget: usize) -> Result<()> {
    let tree = Tree::new(engine, source, budget)?;
    let mut speed = [0.0; 2];
    for hops in 1..=2 {
        let config = StepConfig {
            gravity: Vec3::ZERO,
            contact_exclusion_hops: hops,
            ..step_config()
        };
        let mut gpu = GpuStrands::with_limits(engine, tree.model.clone(), gpu_limits())?;
        gpu.step(engine, config)?;
        let snapshot = gpu.snapshot(engine)?;
        speed[hops as usize - 1] = snapshot
            .fibers()
            .iter()
            .map(|f| f.motion.linear_velocity.length())
            .fold(0.0, f64::max);
        if hops == 2 {
            let mut cpu = tree.model.clone();
            molla_solvers::strands::step_cpu(&mut cpu, config)?;
            compare(&snapshot, &cpu, 1e-5);
        }
    }
    eprintln!(
        "Junction rest test: old contact rule induced {:.6} m/s without gravity or input; junction-aware rule {:.9} m/s",
        speed[0], speed[1]
    );
    assert!(speed[0] > 0.01 && speed[1] < 0.001);
    Ok(())
}

fn larger_step_recovery(engine: &Engine, source: &str, budget: usize) -> Result<()> {
    let mut tree = Tree::new(engine, source, budget)?;
    let camera = crate::draw::Camera {
        yaw: 0.6,
        pitch: 0.16,
        distance: 4.4,
        ..Default::default()
    };
    let rect = eframe::egui::Rect::from_min_size(
        eframe::egui::pos2(298.0, 8.0),
        eframe::egui::vec2(1534.0, 984.0),
    );
    for fracture in [false, true] {
        let pixel = tree.visible_limb(&camera, rect);
        if fracture {
            let (origin, direction) = camera.ray(pixel, rect);
            let (id, hit) = crate::branch::ray_hit(&tree.model, origin, direction).unwrap();
            let plane = CutPlane::new(
                hit,
                tree.model.fiber(id).unwrap().motion.pose.rotation * Vec3::new(0.38, -0.23, 1.0),
            )?;
            legacy_planar_fixture(&mut tree, engine, plane, &[id], true)?;
        } else {
            tree.click(engine, &camera, pixel, rect, false, false)?;
        }
    }
    let config = StepConfig {
        dt: DT * 2.0,
        ..step_config()
    };
    let old_config = StepConfig {
        contact_exclusion_hops: 1,
        ..config
    };
    let mut baseline = GpuStrands::with_limits(engine, tree.model.clone(), gpu_limits())?;
    let mut baseline_failed = false;
    let mut retries = 0;
    let mut checked_oracle = false;
    for batch in 0..240 {
        if !baseline_failed {
            for _ in 0..2 {
                baseline.step(engine, old_config)?;
            }
            let (model, status) = baseline.last_valid_snapshot(engine)?;
            let speed = model
                .fibers()
                .iter()
                .map(|f| f.motion.linear_velocity.length())
                .fold(0.0, f64::max);
            if status != 0 || speed > 100.0 {
                eprintln!(
                    "Unprotected coarse-step replay failed at batch {batch}: status={status:#x}, peak speed={speed}, {:?}",
                    baseline.last_coupled_diagnostics()
                );
                baseline_failed = true;
            }
        }
        let checkpoint = tree.model.clone();
        let result = crate::adaptive_step::advance(
            engine,
            &mut tree.gpu,
            &checkpoint,
            gpu_limits(),
            config,
            2,
        )?;
        if result.halvings > 0 {
            retries += 1;
        }
        if result.halvings > 0 && !checked_oracle {
            let steps = 2usize << result.halvings;
            let fine = StepConfig {
                dt: config.dt / (1u32 << result.halvings) as f64,
                ..config
            };
            let mut cpu = checkpoint;
            for _ in 0..steps {
                molla_solvers::strands::step_cpu(&mut cpu, fine)?;
            }
            compare(&result.model, &cpu, 2e-4);
            checked_oracle = true;
            eprintln!(
                "PASS adaptive angular-resolution CPU oracle at batch {batch}, halvings={}",
                result.halvings
            );
        }
        tree.model = result.model;
        let speed = tree
            .model
            .fibers()
            .iter()
            .map(|f| f.motion.linear_velocity.length())
            .fold(0.0, f64::max);
        assert!(
            speed < 50.0,
            "adaptive replay produced runaway motion: {speed}"
        );
    }
    assert_eq!(tree.model.components().len(), 3);
    eprintln!(
        "PASS coarse-step regression: legacy junction fault={baseline_failed}; adaptive replay completed four seconds with {retries} refined batches, three pieces and bounded motion"
    );
    Ok(())
}
