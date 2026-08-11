//! GPU vertex-triangle self-contact vs the CPU oracle.
//!
//! Run with: `make test-gpu P=molla-geometry T=self_contact_gpu`.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_geometry::{
    detect_vertex_triangle_self_contacts, detect_vertex_triangle_self_contacts_gpu,
};
use molla_math::Vec3;

fn assert_matches(vertices: &[Vec3], triangles: &[[u32; 3]], thickness: f64) {
    let cpu = detect_vertex_triangle_self_contacts(vertices, triangles, thickness);
    let engine = Engine::new_gpu().expect("gpu");
    let gpu = detect_vertex_triangle_self_contacts_gpu(&engine, vertices, triangles, thickness)
        .expect("gpu detect");

    assert_eq!(
        gpu.len(),
        cpu.len(),
        "candidate count differs: gpu={} cpu={}",
        gpu.len(),
        cpu.len()
    );
    for (g, c) in gpu.iter().zip(cpu.iter()) {
        assert_eq!(g.vertex, c.vertex, "vertex mismatch");
        assert_eq!(g.triangle, c.triangle, "triangle mismatch");
        assert!((g.distance - c.distance).abs() < 1e-5, "distance mismatch");
        assert!((g.gap - c.gap).abs() < 1e-5, "gap mismatch");
        assert!((g.point - c.point).length() < 1e-5, "point mismatch");
        assert!((g.normal - c.normal).length() < 1e-5, "normal mismatch");
    }
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_self_contact_matches_cpu_small() {
    // Same fixture as the CPU unit test: vertex 3 sits just above the
    // remote triangle [0,1,2], and its own incident face [0,1,3] is skipped.
    let vertices = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.25, 0.25, 0.05),
    ];
    let triangles = vec![[0, 1, 2], [0, 1, 3]];
    assert_matches(&vertices, &triangles, 0.1);
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_self_contact_matches_cpu_two_layers() {
    // Two parallel triangle strips a small gap apart — several vertices
    // fall inside the shell against the opposite layer's faces.
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    // Lower layer at y = 0, upper layer at y = 0.06.
    for (layer, y) in [(0u32, 0.0f64), (1, 0.06)] {
        let base = layer * 4;
        vertices.push(Vec3::new(0.0, y, 0.0));
        vertices.push(Vec3::new(1.0, y, 0.0));
        vertices.push(Vec3::new(0.0, y, 1.0));
        vertices.push(Vec3::new(1.0, y, 1.0));
        triangles.push([base, base + 1, base + 2]);
        triangles.push([base + 1, base + 3, base + 2]);
    }
    assert_matches(&vertices, &triangles, 0.1);
}
