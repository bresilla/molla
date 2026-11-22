use crate::scene::{NAMES, Scene};
use molla_compute::Engine;

pub fn latency(engine: &Engine) -> molla_core::Result<()> {
    use crate::{branch::{Branch, description}, draw::Camera};
    use eframe::egui::{Pos2, Rect};
    use molla_math::Vec3;
    for (rings, strands) in [(1,7), (2,19), (3,37)] {
        let desc = molla_solvers::strands::BundleDesc {
            rings, strand_radius: 0.025 * (7.0 / strands as f64).sqrt(), ..description()
        };
        let mut branch = Branch::with_description(engine,desc)?;
        branch.advance(engine,4)?;
        for state in ["intact", "cut"] {
            if state == "cut" {
                let camera = Camera::default();
                let rect = Rect::from_min_max(Pos2::ZERO,Pos2::new(900.0,700.0));
                let point = camera.project(Vec3::new(0.3,1.8+(0.3+1.15)/2.3*0.15,0.0),rect).0;
                branch.click_section(engine,&camera,rect,point,false)?;
            }
            let mut times = Vec::new();
            for _ in 0..12 {
                let now = std::time::Instant::now(); branch.advance(engine,2)?;
                times.push(now.elapsed().as_secs_f64()*1000.0);
            }
            times.sort_by(f64::total_cmp);
            println!("BENCH,{strands},{state},{},{:.3},{:.3},submit={:.3},wait_and_decode={:.3}",branch.model.fibers().len(),times[6],times[11],branch.last_enqueue_ms,branch.last_snapshot_ms);
        }
    }
    Ok(())
}

pub fn branches(engine: &Engine) -> molla_core::Result<()> {
    use crate::{branch::Branch, draw::Camera};
    use eframe::egui::{Pos2, Rect};
    use molla_math::Vec3;
    let crate::ActiveScene::Branch(mut soft) = crate::ActiveScene::new(engine, 3)? else {
        unreachable!()
    };
    let crate::ActiveScene::Branch(mut break_preview) = crate::ActiveScene::new(engine, 4)? else {
        unreachable!()
    };
    let mut stiff = Branch::new(engine)?;
    stiff.set_rigidity(engine, 3e7 / crate::branch::description().young_modulus)?;
    soft.pluck(engine, 1.5)?;
    break_preview.pluck(engine, 1.5)?;
    stiff.pluck(engine, 1.5)?;
    let rest = soft.model.fiber(7).unwrap().center();
    let mut peaks = [0.0_f64; 2];
    for _ in 0..48 {
        soft.advance(engine, 2)?;
        break_preview.advance(engine, 2)?;
        stiff.advance(engine, 2)?;
        for (a, b) in soft.model.fibers().iter().zip(break_preview.model.fibers()) {
            assert!(
                a.center().distance(b.center()) < 1e-8,
                "Cut/Break presets changed the dynamics"
            );
            assert!(a.motion.linear_velocity.distance(b.motion.linear_velocity) < 1e-8);
        }
        peaks[0] = peaks[0].max(soft.model.fiber(7).unwrap().center().distance(rest));
        peaks[1] = peaks[1].max(stiff.model.fiber(7).unwrap().center().distance(rest));
    }
    eprintln!("Elasticity comparison: softer/previous-stiff tip travel={peaks:?}");
    assert_eq!(soft.damage_counts(), (0, 0));
    assert_eq!(break_preview.damage_counts(), (0, 0));
    eprintln!(
        "PASS same-branch comparison: {} fibers, identical Cut/Break motion for 96 GPU steps",crate::branch::STRANDS
    );
    assert!(
        peaks[0] > peaks[1] * 1.1,
        "new default did not become visibly more elastic"
    );
    let mut branch = Branch::new(engine)?;
    branch.pluck(engine, 1.5)?;
    branch.advance(engine, 24)?;
    let mass: f64 = branch.model.fibers().iter().map(|f| f.mass.mass).sum();
    let camera = Camera::default();
    let rect = Rect::from_min_max(Pos2::ZERO, Pos2::new(900.0, 700.0));
    for x in [-0.413, 0.537] {
        let a = camera.project(Vec3::new(x, 1.2, 0.0), rect).0;
        let b = camera.project(Vec3::new(x + 0.12, 2.5, 0.0), rect).0;
        branch.cut_stroke(engine, &camera, rect, a, b)?;
    }
    assert_eq!(branch.cuts, 2);
    assert_eq!(branch.model.components().len(), 3);
    assert!(branch.cap_count() >= 4 * crate::branch::STRANDS);
    let after: f64 = branch.model.fibers().iter().map(|f| f.mass.mass).sum();
    assert!((after - mass).abs() < mass * 1e-12);
    for tick in 0..120 {
        branch.advance(engine, 2)?;
        assert!(
            branch.clearance() > -0.005,
            "tick={tick}, clearance={}",
            branch.clearance()
        );
    }
    assert!(
        branch
            .model
            .fibers()
            .iter()
            .any(|f| f.pin.is_none() && f.center().y < 0.5)
    );
    eprintln!(
        "PASS branch: pluck, repeated oblique finite-blade cuts, mass, detached fall, clearance={}",
        branch.clearance()
    );
    let mut stress = Branch::with_description(
        engine,
        molla_solvers::strands::BundleDesc {
            rings: 1,
            strand_radius: 0.025,
            tensile_strength: 3e5,
            shear_strength: 1.5e5,
            damage_rate: 2.0,
        young_modulus: 3e7,
        shear_modulus: 1.2e7,
        damping: 0.6,
            ..crate::branch::description()
        },
    )?;
    stress.pluck(engine, 2.5)?;
    let mut partial = false;
    let mut intact_damaged = false;
    for _ in 0..240 {
        stress.advance(engine, 1)?;
        let (damaged, broken) = stress.damage_counts();
        partial |= damaged > 0;
        intact_damaged |= broken > 0 && stress.model.components().len() == 1;
    }
    eprintln!(
        "Stress diagnostic: partial={partial}, attached_after_break={intact_damaged}, damaged/broken={:?}, components={}",
        stress.damage_counts(),
        stress.model.components().len()
    );
    assert!(partial, "failure did not progress through partial damage");
    assert!(
        intact_damaged,
        "no partially broken attachment was observed"
    );
    assert!(
        stress.damage_counts().1 > 0,
        "loads did not fracture any bonds"
    );
    assert!(
        stress.model.components().len() > 1,
        "stress fracture did not detach a piece"
    );
    assert_eq!(stress.cuts, 0);
    assert_eq!(stress.cap_count(), 0);
    eprintln!("PASS branch: stress-driven fracture without invoking the clean-cut operation");
    Ok(())
}

pub fn cables(engine: &Engine) -> molla_core::Result<()> {
    for (i, name) in NAMES.iter().enumerate() {
        let mut scene = Scene::new(engine, i)?;
        let start = scene.points.clone();
        let mut peak = 1.0f64;
        let mut clearance = f64::INFINITY;
        let mut tube_clearance = f64::INFINITY;
        for tick in 0..180 {
            if tick == 30 {
                scene.pluck(engine)?;
            }
            scene.advance(engine, 4)?;
            peak = peak.max(scene.max_stretch);
            clearance = clearance.min(scene.clearance());
            tube_clearance = tube_clearance.min(scene.tube_clearance());
        }
        assert!(
            scene
                .points
                .iter()
                .zip(&start)
                .any(|(a, b)| a.distance(*b) > 0.1),
            "scene did not move"
        );
        assert!(
            clearance > -0.002,
            "excessive shape penetration: {clearance}"
        );
        assert!(
            tube_clearance > -0.005,
            "excessive tube penetration: {tube_clearance}"
        );
        if i == 1 {
            let tip_y: Vec<_> = scene
                .cables
                .iter()
                .map(|c| scene.points[*c.last().unwrap() as usize].y)
                .collect();
            assert!(
                tip_y[2] > tip_y[0] + 0.3,
                "stiff cable did not resist bending: {tip_y:?}"
            );
            eprintln!("Bending tip heights: {tip_y:?}");
        }
        if i == 2 {
            assert!(
                scene.clearance() < 0.06,
                "cables did not reach the obstacles"
            );
        }
        eprintln!(
            "PASS {name}: 720 steps, max_stretch={peak:.5}, min_clearance={clearance:.5}, tube_clearance={tube_clearance:.5}, time={:.3}",
            scene.time
        );
    }

    Ok(())
}
