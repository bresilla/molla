// ── Scene 59: WarpDem (MPM-backed) ───────────────────────────────
fn warp_dem_pane(ui: &mut egui::Ui, _state: &mut WarpExampleStates) {
    ui.label("Warp DEM — port of core/example_dem.py");
    ui.separator();
    ui.small("MPM Sand rheology with high friction angle (~45°).");
    ui.small("Tweak via the MPM pane (the underlying state is shared).");
    ui.label("");
    ui.label("Slider state lives in the standard MPM pane — open it");
    ui.label("via the right-rail ribbon when it shows.");
}
