use crate::{Tree, apple_tree_model, gpu_limits};
use molla_compute::Engine;
use molla_core::Result;
use molla_geometry::convex_cut::CutPlane;
use molla_math::Vec3;
use molla_solvers::strands::{GpuStrands, StrandModel};

fn restore(engine: &Engine, model: &StrandModel) -> Result<Tree> {
    let mut tree = Tree::new(engine, apple_tree_model::SOURCE, 20)?;
    tree.model = model.clone();
    tree.gpu = GpuStrands::with_limits(engine, model.clone(), gpu_limits())?;
    Ok(tree)
}

pub fn released_sequence(engine: &Engine) -> Result<()> {
    for load_steps in [100, 99, 101, 98] {
        let mut tree = Tree::new(engine, apple_tree_model::SOURCE, 20)?;
        let hit = tree.model.fiber(49).unwrap().center();
        tree.start_grab(49, hit);
        for step in 0..load_steps {
            tree.grab.as_mut().unwrap().target = hit - Vec3::Z * (step as f64 / 240.0);
            tree.advance(engine, 4)?;
        }
        if tree.stress_breaks > 0 {
            continue;
        }
        tree.release_grab();
        let damage = tree
            .model
            .bonds()
            .iter()
            .find(|b| b.id == 48)
            .unwrap()
            .damage;
        let mut checkpoints = Vec::new();
        for batch in 1..=300 {
            tree.advance(engine, 4)?;
            if [15, 30, 60, 90].contains(&batch) {
                checkpoints.push((batch, tree.model.clone()));
            }
            if tree.stress_breaks > 0 {
                break;
            }
        }
        eprintln!(
            "RELEASE CONTROL load={load_steps}, damage={damage}, failures={}",
            tree.stress_breaks
        );
        if tree.stress_breaks > 0 {
            continue;
        }
        for (gap, snapshot) in checkpoints.into_iter().rev() {
            for other in [50, 44, 41, 51, 56] {
                let mut trial = restore(engine, &snapshot)?;
                let f = trial.model.fiber(other).unwrap();
                let plane = CutPlane::new(f.center(), f.motion.pose.rotation * Vec3::Z)?;
                trial.section(engine, plane, &[other], true)?;
                for tick in 1..=180 {
                    trial.advance(engine, 4)?;
                    if trial
                        .model
                        .bonds()
                        .iter()
                        .find(|b| b.id == 48)
                        .is_some_and(|b| !b.intact())
                    {
                        eprintln!(
                            "FOUND RELEASE load={load_steps} gap={gap} break={other} damage={damage} delay={tick}"
                        );
                        std::fs::write("/tmp/molla-showcase-release-scenario.json", serde_json::json!({
                        "load_steps":load_steps, "release_batches":gap, "damage":damage,
                        "break_span":other, "failure_delay_batches":tick, "control_survived_seconds":5
                    }).to_string()).map_err(|e| molla_core::Error::Solver(e.to_string()))?;
                        return Ok(());
                    }
                }
            }
        }
    }
    Err(molla_core::Error::Solver(
        "no released showcase scenario found".into(),
    ))
}

pub fn run(engine: &Engine, selected: Option<u64>, selected_axis: Option<usize>) -> Result<()> {
    let initial = apple_tree_model::build(apple_tree_model::SOURCE, 20)?.model;
    for f in initial.fibers() {
        eprintln!(
            "SPAN {} center={:?} length={:.3} mass={:.4}",
            f.id,
            f.center(),
            f.shape
                .vertices()
                .iter()
                .map(|p| p.z)
                .fold(f64::NEG_INFINITY, f64::max)
                - f.shape
                    .vertices()
                    .iter()
                    .map(|p| p.z)
                    .fold(f64::INFINITY, f64::min),
            f.mass.mass
        );
    }
    for id in [49, 43, 31, 16, 57, 37, 24, 55] {
        if selected.is_some_and(|s| s != id) {
            continue;
        }
        for (axis, direction) in [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z]
            .into_iter()
            .enumerate()
        {
            if selected_axis.is_some_and(|a| a != axis) {
                continue;
            }
            let mut tree = restore(engine, &initial)?;
            let hit = tree.model.fiber(id).unwrap().center();
            tree.start_grab(id, hit);
            let mut threshold = 0.4;
            for step in 0..300 {
                tree.grab.as_mut().unwrap().target = hit + direction * (step as f64 / 240.0);
                if let Err(e) = tree.advance(engine, 4) {
                    eprintln!("PROBE load failed: {id}/{axis} {e}");
                    break;
                }
                if tree.stress_breaks > 0 {
                    break;
                }
                let bond = tree
                    .model
                    .bonds()
                    .iter()
                    .max_by(|a, b| a.damage.total_cmp(&b.damage))
                    .unwrap();
                if bond.damage < threshold {
                    continue;
                }
                let bond_id = bond.id;
                let damage = bond.damage;
                let weak_cells = bond.fibers;
                threshold += 0.15;
                eprintln!(
                    "CANDIDATE grab={id}/{axis} step={step} bond={bond_id} damage={damage:.6} cells={weak_cells:?}"
                );
                let snapshot = tree.model.clone();
                let mut control = restore(engine, &snapshot)?;
                let mut stable = true;
                for _ in 0..180 {
                    if control.advance(engine, 4).is_err() || control.stress_breaks > 0 {
                        stable = false;
                        break;
                    }
                }
                eprintln!("CONTROL grab={id}/{axis} step={step} stable={stable}");
                if !stable {
                    continue;
                }
                for other in [
                    50, 51, 44, 45, 46, 16, 24, 31, 37, 43, 49, 55, 57, 10, 20, 28, 34, 40, 52,
                ] {
                    if weak_cells.contains(&other) || other == id {
                        continue;
                    }
                    let mut trial = restore(engine, &snapshot)?;
                    let f = trial.model.fiber(other).unwrap();
                    let plane = CutPlane::new(f.center(), f.motion.pose.rotation * Vec3::Z)?;
                    if trial.section(engine, plane, &[other], true).is_err() {
                        continue;
                    }
                    for tick in 0..180 {
                        if let Err(e) = trial.advance(engine, 4) {
                            eprintln!("PROBE fracture failed: {e}");
                            break;
                        }
                        if trial.recovered_batches > 12 {
                            break;
                        }
                        if trial
                            .model
                            .bonds()
                            .iter()
                            .find(|b| b.id == bond_id)
                            .is_some_and(|b| !b.intact())
                        {
                            eprintln!(
                                "FOUND grab={id} axis={axis} step={step} damage={damage} bond={bond_id} break={other} delay={:.3}",
                                (tick + 1) as f64 / 60.0
                            );
                            std::fs::write(
                                "/tmp/molla-showcase-scenario.json",
                                serde_json::json!({
                                    "grab": id, "axis": axis, "steps": step+1, "damage": damage,
                                    "bond": bond_id, "break": other, "delay_frames": tick+1,
                                    "control_survived_seconds": 3
                                })
                                .to_string(),
                            )
                            .map_err(|e| molla_core::Error::Solver(e.to_string()))?;
                            return Ok(());
                        }
                    }
                }
            }
        }
    }
    Err(molla_core::Error::Solver(
        "no delayed-failure showcase found".into(),
    ))
}
