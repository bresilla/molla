use crate::{Foliage, apple_tree_sim};
use molla_compute::Engine;
use molla_core::Result;
use molla_geometry::convex_cut::CutPlane;
use molla_math::Vec3;
use molla_solvers::strands::GpuStrands;

fn copy(engine: &Engine, source: &Foliage) -> Result<Foliage> {
    let mut world = Foliage::new(engine)?;
    world.tree.model = source.tree.model.clone();
    world.tree.gpu = GpuStrands::with_limits(
        engine,
        world.tree.model.clone(),
        apple_tree_sim::gpu_limits(),
    )?;
    world.tree.time = source.tree.time;
    world.tree.step_substeps = source.tree.step_substeps;
    world.tree.quiet_batches = source.tree.quiet_batches;
    world.leaves = source.leaves.clone();
    world.particles = source.particles.clone();
    world.gpu.restore(engine, &world.particles)?;
    Ok(world)
}

fn survives(engine: &Engine, world: &mut Foliage, frames: usize) -> Result<bool> {
    for _ in 0..frames {
        world.advance(engine, 4)?;
        if world.tree.stress_breaks > 0 {
            return Ok(false);
        }
    }
    Ok(true)
}

pub fn run(engine: &Engine) -> Result<()> {
    let mut world = Foliage::new(engine)?;
    world.advance(engine, 180)?;
    world.shake_gain = 1.5;
    for (branch, angle) in [(21, 0.0), (11, std::f64::consts::PI)] {
        world.selected = branch;
        world.shake_angle = angle;
        world.start_shake();
        world.advance(engine, 600)?;
    }
    let hit = world.tree.model.fiber(49).unwrap().center();
    world.tree.start_grab(49, hit);
    let mut snapshots = Vec::new();
    for step in 0..89 {
        world.tree.grab.as_mut().unwrap().target = hit - Vec3::Z * (step as f64 / 240.0);
        world.advance(engine, 4)?;
        if step >= 83 {
            snapshots.push((step + 1, copy(engine, &world)?));
        }
    }
    for (steps, mut trial) in snapshots.into_iter().rev() {
        if !survives(engine, &mut trial, 30)? {
            eprintln!("LEAFY PROBE load={steps}: failed release gap");
            continue;
        }
        let mut control = copy(engine, &trial)?;
        if !survives(engine, &mut control, 180)? {
            eprintln!("LEAFY PROBE load={steps}: failed unedited control");
            continue;
        }
        let damage = |w: &Foliage| {
            w.tree
                .model
                .bonds()
                .iter()
                .find(|b| b.id == 48)
                .unwrap()
                .damage
        };
        eprintln!(
            "LEAFY PROBE load={steps}: released damage={}, control damage={}",
            damage(&trial),
            damage(&control)
        );
        for id in [44, 41, 56] {
            let mut edited = copy(engine, &trial)?;
            let f = edited.tree.model.fiber(id).unwrap();
            let plane = CutPlane::new(f.center(), f.motion.pose.rotation * Vec3::Z)?;
            edited.tree.section(engine, plane, &[id], true)?;
            for frame in 1..=180 {
                edited.advance(engine, 4)?;
                if damage(&edited) >= 1.0 {
                    eprintln!(
                        "FOUND leafy release: load={steps}, break={id}, delay_frames={frame}"
                    );
                    std::fs::write("target/leafy-release-scenario.json",serde_json::json!({"load_steps":steps,"break_span":id,"delay_frames":frame,"released_damage":damage(&trial),"control_damage":damage(&control),"control_seconds":3}).to_string()).map_err(|e|molla_core::Error::Solver(e.to_string()))?;
                    return Ok(());
                }
            }
        }
    }
    Err(molla_core::Error::Solver(
        "no released leafy failure scenario found".into(),
    ))
}
