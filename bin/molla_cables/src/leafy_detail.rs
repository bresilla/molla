use super::*;

fn bend(
    movie: &mut Movie,
    engine: &Engine,
    id: u64,
    direction: Vec3,
    force: f64,
) -> video_capture::Result<()> {
    let point = movie
        .world
        .tree
        .model
        .fiber(id)
        .ok_or("bend span missing")?
        .center();
    if !rooted(&movie.world.tree.model).contains(&id) {
        return Err(format!("bend span {id} is already detached").into());
    }
    movie.camera_to(engine, point, 1.75, 0.7, 30)?;
    let hit = movie.world.tree.model.fiber(id).unwrap().center();
    let failures = movie.world.tree.stress_breaks;
    movie.world.tree.grip_force = force;
    movie.world.tree.start_grab(id, hit);
    movie.event("bend_started", serde_json::json!({"span":id,"direction":direction.to_array(),"target_amplitude_m":0.12,"force_limit_n":force}));
    let mut excursion: f64 = 0.0;
    for frame in 0..180 {
        let phase = frame as f64 / 179.0 * std::f64::consts::TAU;
        let offset = direction.normalize() * (0.12 * phase.sin());
        movie
            .world
            .tree
            .grab
            .as_mut()
            .ok_or("bend grab lost")?
            .target = hit + offset;
        let still = match frame {
            45 => Some(format!("bend-{id}-forward")),
            135 => Some(format!("bend-{id}-reverse")),
            _ => None,
        };
        movie.frame(engine, 4, still.as_deref())?;
        let grab = movie.world.tree.grab.unwrap();
        let current = movie
            .world
            .tree
            .model
            .fiber(grab.fiber)
            .unwrap()
            .motion
            .pose
            .transform_point(grab.local_point);
        excursion = excursion.max(current.distance(hit));
    }
    movie.world.tree.release_grab();
    movie.world.tree.grip_force = 4.0;
    movie.event("bend_released", serde_json::json!({"span":id,"peak_excursion_m":excursion,"new_stress_failures":movie.world.tree.stress_breaks-failures}));
    movie.frames(engine, 75)?;
    movie.event(
        "bend_recovery",
        serde_json::json!({"span":id,"free_motion_frames":75}),
    );
    if excursion < 0.025 {
        return Err(format!("bend {id} was not visibly displaced: {excursion}").into());
    }
    movie.check()
}

fn close_break(movie: &mut Movie, engine: &Engine, id: u64) -> video_capture::Result<()> {
    let f = movie
        .world
        .tree
        .model
        .fiber(id)
        .ok_or("close-up span missing")?;
    let point = f.center();
    let axis = f.motion.pose.rotation * Vec3::Z;
    let yaw = axis.x.atan2(axis.z) + 0.7;
    movie.camera_to(engine, point, 0.32, yaw, 45)?;
    movie.event(
        "fracture_closeup",
        serde_json::json!({"span":id,"distance_m":0.32}),
    );
    movie.rough_break(engine, id, true)?;
    movie.frame(engine, 4, Some(&format!("break-{id}-onset")))?;
    movie.frames(engine, 14)?;
    movie.frame(engine, 4, Some(&format!("break-{id}-fibers")))?;
    movie.frames(engine, 30)?;
    movie.camera_to(engine, Vec3::new(0.0, 1.0, 0.0), 4.3, 0.65, 45)?;
    movie.frames(engine, 75)?;
    movie.check()
}

fn path_section(
    initial: &StrandModel,
    ids: &[u64],
    fraction: f64,
) -> video_capture::Result<(Transform, Vec3)> {
    if !fraction.is_finite() || fraction <= 0.0 || fraction >= 1.0 || ids.is_empty() {
        return Err("section fraction must be inside a nonempty material path".into());
    }
    let total: f64 = ids
        .iter()
        .map(|&id| {
            let (a, b) = extent(initial.fiber(id).unwrap());
            b - a
        })
        .sum();
    let mut distance = total * fraction;
    for &id in ids {
        let f = initial.fiber(id).unwrap();
        let (a, b) = extent(f);
        if distance < b - a {
            return Ok((f.rest, Vec3::Z * (a + distance)));
        }
        distance -= b - a;
    }
    Err("section outside material path".into())
}

fn cut(
    movie: &mut Movie,
    engine: &Engine,
    rest: Transform,
    local: Vec3,
    kind: &str,
    oblique: f64,
) -> video_capture::Result<()> {
    if movie.world.tree.grab.is_some() {
        return Err("cut while a branch is held".into());
    }
    movie.cue(engine, rest, local, false, 30)?;
    let f = material_span(&movie.world.tree.model, rest, local).ok_or("cut material detached")?;
    let point = f.motion.pose.transform_point(local);
    let plane = CutPlane::new(
        point,
        (f.motion.pose.rotation * Vec3::Z + Vec3::new(oblique, 0.0, 0.6 * oblique)).normalize(),
    )?;
    let id = f.id;
    let before = rooted(&movie.world.tree.model);
    movie.world.tree.section(engine, plane, &[id], false)?;
    let after = rooted(&movie.world.tree.model);
    let detached: Vec<_> = movie
        .world
        .tree
        .model
        .fibers()
        .iter()
        .filter(|f| !after.contains(&f.id) && (before.contains(&f.id) || f.rest == rest))
        .map(|f| f.id)
        .collect();
    let mass: f64 = detached
        .iter()
        .map(|&id| movie.world.tree.model.fiber(id).unwrap().mass.mass)
        .sum();
    let cells = detached.len();
    movie.event(kind, serde_json::json!({"selected_span":id,"point":point.to_array(),"detached_cells":cells,"detached_mass_kg":mass}));
    movie.marks.clear();
    movie.frames(engine, 210)?;
    movie.frame(
        engine,
        4,
        Some(&format!("cut-{}-landed", movie.world.tree.cuts)),
    )?;
    movie.event("fall_observed", serde_json::json!({"after_cut":movie.world.tree.cuts,"settle_frames":211,"detached_cells":cells}));
    movie.check()
}

fn branch_halves(
    movie: &mut Movie,
    engine: &Engine,
    initial: &StrandModel,
    trunk_ids: &[u64],
) -> video_capture::Result<()> {
    let built = leafy_model::build()?.tree;
    let mut paths = built.limb_cells;
    paths.sort_by(|a, b| {
        let length = |ids: &[u64]| {
            ids.iter()
                .map(|&id| {
                    let (a, b) = extent(initial.fiber(id).unwrap());
                    b - a
                })
                .sum::<f64>()
        };
        length(b).total_cmp(&length(a))
    });
    let ids = paths
        .iter()
        .find(|ids| {
            let root = rooted(&movie.world.tree.model);
            ids.iter().all(|id| root.contains(id))
        })
        .ok_or("no complete large branch remains")?;
    let (rest, local) = path_section(initial, ids, 0.50)?;
    let point = material_span(&movie.world.tree.model, rest, local)
        .unwrap()
        .motion
        .pose
        .transform_point(local);
    movie.camera_to(
        engine,
        Vec3::new(point.x * 0.3, 0.95, point.z * 0.3),
        4.2,
        0.65,
        40,
    )?;
    cut(movie, engine, rest, local, "large_branch_half_cut", 0.2)?;
    let trunk: Vec<_> = trunk_ids
        .iter()
        .map(|&id| initial.fiber(id).unwrap())
        .collect();
    let base = initial.fiber(ids[0]).unwrap();
    let local = Vec3::Z * collar_z(base, &trunk);
    cut(
        movie,
        engine,
        base.rest,
        local,
        "remaining_branch_base_cut",
        0.0,
    )
}

pub(super) fn finish(
    mut movie: Movie,
    engine: &Engine,
    initial: &StrandModel,
    trunk_ids: &[u64],
    shed: usize,
) -> video_capture::Result<()> {
    for (id, direction, force) in [
        (21, Vec3::X, 2.0),
        (11, -Vec3::Z, 2.0),
        (37, Vec3::new(-1.0, 0.35, 0.0), 2.0),
        (55, Vec3::new(0.0, -0.3, 1.0), 4.0),
    ] {
        bend(&mut movie, engine, id, direction, force)?;
    }
    for id in [32, 41, 56] {
        close_break(&mut movie, engine, id)?;
    }
    branch_halves(&mut movie, engine, initial, trunk_ids)?;
    let (rest, local) = path_section(initial, trunk_ids, 0.53)?;
    movie.camera_to(engine, Vec3::new(0.0, 1.1, 0.0), 4.7, 0.95, 40)?;
    cut(&mut movie, engine, rest, local, "crown_trunk_cut", 0.85)?;
    movie.frames(engine, 90)?;
    branch_halves(&mut movie, engine, initial, trunk_ids)?;
    let (rest, local) = path_section(initial, trunk_ids, 0.22)?;
    movie.camera_to(engine, Vec3::new(0.0, 0.65, 0.0), 3.7, 1.0, 40)?;
    cut(&mut movie, engine, rest, local, "lower_trunk_cut", -0.85)?;
    movie.frames(engine, 120)?;
    movie.frame(engine, 4, Some("final-large-pieces"))?;
    let trunk_rest: Vec<_> = trunk_ids
        .iter()
        .map(|&id| initial.fiber(id).unwrap().rest)
        .collect();
    let components = trunk_components(&movie.world.tree.model, &trunk_rest);
    if movie.world.tree.breaks != 4 || components != 3 {
        return Err("detail sequence lost a required break or trunk component".into());
    }
    movie.check()?;
    let report = serde_json::json!({"sequence":"bending-closeups-large-cuts-v2","width":1920,"height":1080,"fps":30,"frames":movie.frame,
        "duration_seconds":movie.frame as f64/30.0,"simulation_seconds":movie.world.tree.time,
        "rough_breaks":movie.world.tree.breaks,"clean_cuts":movie.world.tree.cuts,"stress_breaks":movie.world.tree.stress_breaks,
        "trunk_components":components,"mass_before":movie.initial_mass,"mass_after":movie.world.tree.model.fibers().iter().map(|f|f.mass.mass).sum::<f64>(),
        "floor_clearance_m":movie.world.tree.clearance(),"events":movie.events,"leaves_total":movie.world.leaves.len(),"leaves_attached":movie.world.attached(),"opening_leaves_shed":shed,
        "leaf_strength_multiplier_vs_original":leafy_model::ATTACHMENT_STRENGTH,"rendering":"offline 1080p Vulkan; 0.5x motion; no real-time claim"});
    if let Some(parent) = movie.path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(
        movie.path.with_extension("json"),
        serde_json::to_string_pretty(&report)?,
    )?;
    if let Some(capture) = movie.capture.take() {
        capture.finish()?;
    }
    if let Some(capture) = movie.preview.take() {
        capture.finish()?;
    }
    eprintln!(
        "PASS detailed leafy showcase: {} frames, four extra bends, four rough breaks, {} large cuts",
        movie.frame, movie.world.tree.cuts
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detail_sections_are_inside_material_and_not_node_snapped() {
        let built = leafy_model::build().unwrap().tree;
        for ids in built
            .limb_cells
            .iter()
            .chain(std::iter::once(&built.trunk_cells))
        {
            for fraction in [0.22, 0.50, 0.53] {
                let (rest, point) = path_section(&built.model, ids, fraction).unwrap();
                let f = built
                    .model
                    .fibers()
                    .iter()
                    .find(|f| f.rest == rest)
                    .unwrap();
                let (lo, hi) = extent(f);
                assert!(point.z > lo + 1e-6 && point.z < hi - 1e-6);
                assert!(f.contains_local(point));
            }
        }
    }

    #[test]
    fn section_rejects_empty_paths_and_invalid_fractions() {
        let built = leafy_model::build().unwrap().tree;
        assert!(path_section(&built.model, &[], 0.5).is_err());
        for fraction in [-1.0, 0.0, 1.0, 2.0, f64::NAN] {
            assert!(path_section(&built.model, &built.trunk_cells, fraction).is_err());
        }
    }

    #[test]
    fn bend_targets_have_distal_material() {
        let built = leafy_model::build().unwrap().tree;
        for id in [11, 21, 37, 55] {
            let ids = built
                .limb_cells
                .iter()
                .find(|ids| ids.contains(&id))
                .unwrap();
            assert_ne!(ids[0], id);
            assert!(built.model.fiber(id).unwrap().pin.is_none());
        }
    }

    #[test]
    fn large_cuts_retain_a_rooted_branch_after_the_crown_drop() {
        let built = leafy_model::build().unwrap().tree;
        let mut bonds = built.model.bonds().to_vec();
        bonds.iter_mut().find(|b| b.id == 48).unwrap().damage = 1.0;
        let mut model = StrandModel::new(built.model.fibers().to_vec(), bonds).unwrap();
        for (index, id) in [44, 32, 41, 56].into_iter().enumerate() {
            let f = model.fiber(id).unwrap();
            let plane = CutPlane::new(f.center(), f.motion.pose.rotation * Vec3::Z).unwrap();
            let edit = model.prepare_splinter(id, plane, index as u64 + 1).unwrap();
            model.commit_splinter(edit).unwrap();
        }
        let length = |ids: &[u64]| {
            ids.iter()
                .map(|&id| {
                    let (a, b) = extent(built.model.fiber(id).unwrap());
                    b - a
                })
                .sum::<f64>()
        };
        let mut paths = built.limb_cells.clone();
        paths.sort_by(|a, b| length(b).total_cmp(&length(a)));
        let root = rooted(&model);
        let first = paths
            .iter()
            .find(|ids| ids.iter().all(|id| root.contains(id)))
            .unwrap();
        let (rest, local) = path_section(&built.model, first, 0.5).unwrap();
        let section = |model: &mut StrandModel, rest, local| {
            let f = material_span(model, rest, local).unwrap();
            let plane = CutPlane::new(
                f.motion.pose.transform_point(local),
                f.motion.pose.rotation * Vec3::Z,
            )
            .unwrap();
            let edit = model.prepare_cut(plane, &[f.id], 100).unwrap();
            model.commit_cut(edit).unwrap();
        };
        section(&mut model, rest, local);
        let trunk: Vec<_> = built
            .trunk_cells
            .iter()
            .map(|&id| built.model.fiber(id).unwrap())
            .collect();
        let base = built.model.fiber(first[0]).unwrap();
        section(&mut model, base.rest, Vec3::Z * collar_z(base, &trunk));
        let (rest, local) = path_section(&built.model, &built.trunk_cells, 0.53).unwrap();
        section(&mut model, rest, local);
        let root = rooted(&model);
        let remaining: Vec<_> = paths
            .iter()
            .filter(|ids| ids.iter().all(|id| root.contains(id)))
            .collect();
        eprintln!("first large branch={first:?}; after crown cut={remaining:?}");
        assert!(!remaining.is_empty());
    }

    #[test]
    #[ignore = "requires Vulkan"]
    fn distal_bend_moves_visibly_without_fracture() {
        let engine = Engine::new_gpu().unwrap();
        let mut world = Foliage::new(&engine).unwrap();
        world.advance(&engine, 180).unwrap();
        let initial_mass = world.tree.model.fibers().iter().map(|f| f.mass.mass).sum();
        let mut movie = Movie {
            world,
            camera: camera(),
            capture: None,
            preview: None,
            path: "target/bend-check.json".into(),
            frame: 0,
            marks: Vec::new(),
            events: Vec::new(),
            initial_mass,
        };
        bend(&mut movie, &engine, 55, Vec3::new(0.0, -0.3, 1.0), 4.0).unwrap();
        assert_eq!(movie.world.tree.stress_breaks, 0);
    }
}
