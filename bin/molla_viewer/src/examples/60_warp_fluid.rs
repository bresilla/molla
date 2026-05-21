// ── Scene 60: WarpFluid (MPM-backed) ─────────────────────────────
fn warp_fluid_pane(ui: &mut egui::Ui, _state: &mut WarpExampleStates) {
    ui.label("Warp fluid — port of core/example_fluid.py");
    ui.separator();
    ui.small("MPM Sand rheology with very low friction (~5°).");
    ui.small("Behaves fluid-like; share state with the MPM pane.");
}
