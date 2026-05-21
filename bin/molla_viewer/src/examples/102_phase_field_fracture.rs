fn draw_phase_field_fracture_gizmos(sim: Res<Sim>, mut gizmos: Gizmos) {
    if sim.scene != Scene::PhaseFieldFracture {
        return;
    }

    let nx = 16usize;
    let ny = 6usize;
    let mesh = fem::FemMesh::tri_grid_2d(nx, ny, 2.4, 0.9);
    let mut history = vec![0.05; mesh.vert_count()];
    for (i, p) in mesh.vertices.iter().enumerate() {
        let x = p.x - 1.2;
        let y = p.y - 0.45;
        let notch = (-((x + 0.25) * (x + 0.25)) / 0.045 - (y * y) / 0.003).exp();
        let midline = (-(y * y) / 0.015).exp();
        history[i] = 0.08 + 7.0 * notch + 1.3 * midline;
    }
    let params = fem::PhaseFieldFractureParams {
        critical_energy_release_rate: 1.0,
        length_scale: 0.11,
        residual_stiffness: 1.0e-6,
    };
    let damage_old = vec![0.0; mesh.vert_count()];
    let Ok(result) =
        fem::solve_phase_field_fracture_jacobi(&mesh, &damage_old, &history, params, 80, 1.0e-5)
    else {
        return;
    };

    let to_world = |p: &MVec3, d: f64| -> Vec3 {
        Vec3::new(
            p.x as f32 - 1.2,
            (0.05 + 0.45 * d) as f32,
            p.y as f32 - 0.45,
        )
    };
    for cell in &mesh.cells {
        let verts = cell.verts();
        for e in 0..verts.len() {
            let a = verts[e] as usize;
            let b = verts[(e + 1) % verts.len()] as usize;
            let da = result.damage[a];
            let db = result.damage[b];
            let s = (0.5 * (da + db)).clamp(0.0, 1.0) as f32;
            let col = Color::srgb(0.10 + 0.90 * s, 0.20 * (1.0 - s), 0.95 * (1.0 - s));
            gizmos.line(
                to_world(&mesh.vertices[a], da),
                to_world(&mesh.vertices[b], db),
                col,
            );
        }
    }
    for (p, d) in mesh.vertices.iter().zip(result.damage.iter()) {
        let s = d.clamp(0.0, 1.0) as f32;
        gizmos.sphere(
            to_world(p, *d),
            0.012 + 0.020 * s,
            Color::srgb(0.2 + 0.8 * s, 0.25 * (1.0 - s), 1.0 - 0.8 * s),
        );
    }
}

