# Out of scope

This page is the current physics-only non-goal list. Items here are design
decisions, not gaps.

## External formats and asset pipelines

molla does not own robot/scene/mesh importers or exporters. Build procedural
models with `ModelBuilder` and procedural FEM meshes with the native grid
builders. Format adapters can live outside this repository if a downstream
application needs them.

## External solver wrappers and native shims

molla ships native Rust + WGSL physics. It does not wrap third-party physics
engines or add native ABI shims just to reach another solver.

## External language bindings and tensor-framework bridges

molla is Rust-first. No language binding layer, tensor bridge, or foreign
autograd framework is part of the engine surface. Learned controllers consume
plain row-major weight slabs and run through native actuator kernels.

## Viewer backends and recording/replay

The viewer is a single Bevy application under `bin/`. Alternate viewers,
record/replay systems, and backend-specific visual pipelines are outside the
engine plan.

## Non-portable GPU features

The GPU contract is portable `wgpu`/WGSL. Vendor-only matrix units,
multi-device peer access, and backend-specific command capture stay out until
they become portable across the backends molla targets.

## Generic constrained-dynamics mega-solvers

The native solver set is Featherstone, XPBD, VBD, MPM, and FEM. General-purpose
LCP/NLP solver work is not part of this plan unless it becomes a small native
physics slice with CPU oracle, GPU path where feasible, tests, benches, and a
viewer scene.
