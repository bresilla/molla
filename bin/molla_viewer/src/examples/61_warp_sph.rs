// ── Scene 61: WarpSph (MPM-backed visual parity) ─────────────────
fn warp_sph_pane(ui: &mut egui::Ui, _state: &mut WarpExampleStates) {
    ui.label("Warp SPH — port of core/example_sph.py");
    ui.separator();
    ui.small("Same low-friction MPM as `WarpFluid` — molla has");
    ui.small("no real SPH solver, MPM stands in.");
}
