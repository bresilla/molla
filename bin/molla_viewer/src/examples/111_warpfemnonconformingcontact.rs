// ── Scene 111: WarpFemNonconformingContact ────────────────────────
struct WarpFemNonconformingContactState {
    contact_u: f64,
    contact_v: f64,
    load: f64,
    target_tri: [MVec3; 3],
    source_quad: [MVec3; 4],
    contact_point: MVec3,
    node_forces: [MVec3; 3],
    net_force_error: f64,
    moment_error: f64,
    user_reset: bool,
}

impl Default for WarpFemNonconformingContactState {
    fn default() -> Self {
        let mut me = Self {
            contact_u: 0.32,
            contact_v: 0.28,
            load: 1.0,
            target_tri: [
                MVec3::new(-0.55, 0.0, -0.35),
                MVec3::new(0.65, 0.0, -0.20),
                MVec3::new(-0.15, 0.0, 0.55),
            ],
            source_quad: [
                MVec3::new(-0.65, 0.12, -0.45),
                MVec3::new(0.55, 0.12, -0.40),
                MVec3::new(0.60, 0.12, 0.45),
                MVec3::new(-0.55, 0.12, 0.50),
            ],
            contact_point: MVec3::ZERO,
            node_forces: [MVec3::ZERO; 3],
            net_force_error: 0.0,
            moment_error: 0.0,
            user_reset: true,
        };
        me.recompute();
        me
    }
}

fn nonconforming_barycentric_point(tri: [MVec3; 3], u: f64, v: f64) -> MVec3 {
    tri[0] * (1.0 - u - v) + tri[1] * u + tri[2] * v
}

fn nonconforming_patch_transfer(_tri: [MVec3; 3], u: f64, v: f64, force: MVec3) -> [MVec3; 3] {
    let w0 = 1.0 - u - v;
    [force * w0, force * u, force * v]
}

fn nonconforming_force_moment_errors(
    tri: [MVec3; 3],
    point: MVec3,
    force: MVec3,
    node_forces: [MVec3; 3],
) -> (f64, f64) {
    let net: MVec3 = node_forces.iter().copied().sum();
    let moment: MVec3 = tri
        .iter()
        .zip(node_forces.iter())
        .map(|(p, f)| p.cross(*f))
        .sum();
    (
        (net - force).length(),
        (moment - point.cross(force)).length(),
    )
}

impl WarpFemNonconformingContactState {
    fn recompute(&mut self) {
        let max_v = (0.98 - self.contact_u).max(0.02);
        self.contact_v = self.contact_v.clamp(0.01, max_v);
        self.contact_u = self.contact_u.clamp(0.01, 0.98 - self.contact_v);
        let force = MVec3::new(0.15 * self.load, self.load, -0.10 * self.load);
        self.contact_point =
            nonconforming_barycentric_point(self.target_tri, self.contact_u, self.contact_v);
        self.node_forces =
            nonconforming_patch_transfer(self.target_tri, self.contact_u, self.contact_v, force);
        let (fe, me) = nonconforming_force_moment_errors(
            self.target_tri,
            self.contact_point,
            force,
            self.node_forces,
        );
        self.net_force_error = fe;
        self.moment_error = me;
        self.user_reset = false;
    }
}

fn update_warp_fem_nonconforming_contact(sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpFemNonconformingContact {
        return;
    }
    let st = &mut warp.fem_nonconforming_contact;
    if st.user_reset {
        st.recompute();
    }
}

fn draw_warp_fem_nonconforming_contact_gizmos(
    sim: Res<Sim>,
    warp: Res<WarpExampleStates>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::WarpFemNonconformingContact {
        return;
    }
    let st = &warp.fem_nonconforming_contact;
    let to_v = |p: MVec3| Vec3::new(p.x as f32, p.y as f32, p.z as f32);
    let source_col = Color::srgb(0.90, 0.55, 0.20);
    let target_col = Color::srgb(0.25, 0.70, 0.95);
    for e in 0..4 {
        gizmos.line(
            to_v(st.source_quad[e]),
            to_v(st.source_quad[(e + 1) % 4]),
            source_col,
        );
    }
    for &(a, b) in &[(0, 1), (1, 2), (2, 0)] {
        gizmos.line(to_v(st.target_tri[a]), to_v(st.target_tri[b]), target_col);
    }
    gizmos.sphere(to_v(st.contact_point), 0.025, Color::srgb(1.0, 0.95, 0.25));
    for (p, f) in st.target_tri.iter().zip(st.node_forces.iter()) {
        let a = to_v(*p);
        let b = to_v(*p + *f * 0.18);
        gizmos.line(a, b, Color::srgb(0.40, 1.0, 0.45));
    }
}

fn warp_fem_nonconforming_contact_pane(
    ui: &mut egui::Ui,
    st: &mut WarpFemNonconformingContactState,
) {
    ui.label("Warp FEM nonconforming contact — reduced load transfer");
    ui.separator();
    readout(ui, "Force err", &format!("{:.2e}", st.net_force_error));
    readout(ui, "Moment err", &format!("{:.2e}", st.moment_error));
    let mut changed = false;
    changed |= ui
        .add(egui::Slider::new(&mut st.contact_u, 0.01..=0.95).text("contact u"))
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut st.contact_v, 0.01..=0.95).text("contact v"))
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut st.load, 0.1..=2.0).text("load"))
        .changed();
    if changed || ui.button("Recompute").clicked() {
        st.user_reset = true;
    }
    ui.small("Orange = source patch; blue = nonmatching target triangle; green = transferred nodal forces.");
    ui.small("CPU oracle only until nonconforming contact kernels exist.");
}

