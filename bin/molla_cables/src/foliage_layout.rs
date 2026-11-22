//! Layout of the baked foliage textures shared by the bake tool, mesh and renderer.
#![allow(dead_code)]

/// ambientCG atlases whose leaves are extracted, in layer order.
pub const LEAF_SETS: [&str; 6] = [
    "Leaf001",
    "LeafSet001",
    "LeafSet003",
    "LeafSet014",
    "LeafSet022",
    "LeafSet024",
];
/// Number of individual leaf textures (layers) baked from `LEAF_SETS`.
pub const LEAF_TEXTURES: usize = 33;
/// Width (across the blade) and height (along the blade) of one leaf layer.
pub const LEAF_CELL: [u32; 2] = [352, 512];
/// Leaf layers per row in the packed atlas.
pub const LEAF_COLUMNS: u32 = 11;
/// Lateral half-extent of a leaf layer, in blade lengths.
pub const HALF_EXTENT: f64 = 0.33;

/// ambientCG bark materials: trunk, limb, young shoot.
pub const BARK_SETS: [&str; 3] = ["Bark004", "Bark003", "Bark005"];
/// Edge length of one square bark layer.
pub const BARK_CELL: u32 = 1024;
/// Physical width/height ratio of each bark scan.
pub const BARK_ASPECT: [f32; 3] = [1.0, 1.0, 2.0];
/// Circumferential world size of one bark tile, in metres.
pub const BARK_TILE_M: f64 = 0.2;
/// Wood radii mapped to the trunk and young-shoot bark.
pub const BARK_RADIUS_M: [f64; 2] = [0.03, 0.004];

/// Half-width of the leaf blade outline, in blade lengths, at fraction `t` from base to tip.
pub fn half_width(t: f64) -> f64 {
    (0.30 * (t.clamp(0.0, 1.0).sqrt() * (1.0 - t.clamp(0.0, 1.0))) / 0.3849).max(0.003)
}

/// Texture coordinate of blade point `t` (base 0, tip 1) at lateral offset `x` in blade lengths.
pub fn leaf_uv(t: f64, x: f64, mirror: bool) -> [f32; 2] {
    let x = if mirror { -x } else { x };
    [(0.5 + 0.5 * x / HALF_EXTENT) as f32, (1.0 - t) as f32]
}

/// Bark blend weight: 0 on trunk-sized wood, 1 on young shoots.
pub fn bark_thinness(radius: f64) -> f64 {
    let [thick, thin] = BARK_RADIUS_M.map(f64::ln);
    ((thick - radius.max(1e-6).ln()) / (thick - thin)).clamp(0.0, 1.0)
}

/// Whole number of bark tiles wrapped around a circumference of radius `radius`.
pub fn bark_wraps(radius: f64) -> f64 {
    (std::f64::consts::TAU * radius / BARK_TILE_M).round().max(1.0)
}
