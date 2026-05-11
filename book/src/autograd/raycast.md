# Raycast adjoint

Stage 37.7 ships a **finite-difference** adjoint for the raycast
pipeline — closed-form analytic adjoints (`∂t_hit / ∂param` per
primitive) are deferred to Stage 37.7.1. The FD path is enough to
make inverse-rendering loops work (gradient-descent on a shape's
pose to match a target depth image, mirroring Warp's
`example_diffray.py`) and serves as the oracle for the future
analytic version.

Source: `crates/molla/src/sensors/raycast_adjoint.rs`. Forward
(GPU): `crates/molla/src/sensors/raycast.rs` (see
[`Raycaster`](../sensors/raycaster.md)).

## CPU-only by design

Even when the analytic adjoint lands, it's likely to stay
CPU-resident: the FD adjoint probes the forward pass
`O(P · N_rays)` times (one ± perturbation per parameter), and every
probe runs the same closest-hit math. Pushing that onto the GPU
would not help — the bottleneck is per-probe latency, not per-probe
throughput, and the dependency graph between probes is dense enough
that they can't be batched cleanly.

The CPU adjoint runs a compact `(params, rays)`-parameterised
forward (`forward_with_loss`) so a single FD perturbation is one
function call. The same forward routine is what `Raycaster::cast`
runs on the GPU; the parity-test harness pins the two to within
floating-point tolerance.

## Forward + loss contract

```rust
pub struct Ray { pub origin: Vec3, pub direction: Vec3 }
pub struct RayScene { pub shapes: Vec<SdfShape> }

pub enum SdfShape {
    Sphere { param_offset: usize, radius: Real },
    PlaneY { height: Real },
    PlaneZ { depth: Real },
}
```

Parameters are unstructured: each `Sphere` consumes 3 slots (xyz);
the flat layout is `[s0.x, s0.y, s0.z, s1.x, s1.y, s1.z, ...]`. The
scene's `n_params()` method returns the next safe offset for adding
a sphere without overlap (overlapping ranges silently couple
gradients between shapes and produce optimisation behaviour that's
almost impossible to debug — the constructor panics on overlap).

The image is a flat `Vec<Real>` of per-ray hit distances; misses
carry `Real::INFINITY`. The default L2 loss skips any pixel where
either side is `∞` so silhouette boundaries don't blow up the
residual.

```rust
pub fn forward_with_loss(
    scene: &RayScene,
    params: &[Real],
    rays: &[Ray],
    target_image: &[Real],
    max_distance: Real,
) -> (Vec<Real>, Real);   // (image, loss)

pub fn compute_loss_l2(image: &[Real], target: &[Real]) -> Real;

pub fn compute_grad_raycast(
    scene: &RayScene,
    params: &[Real],
    rays: &[Ray],
    target_image: &[Real],
    max_distance: Real,
    eps: Real,
) -> Vec<Real>;   // dL/dparam, length = params.len()
```

`compute_grad_raycast` uses **central differences**: `g_i =
(L(p + ε e_i) − L(p − ε e_i)) / (2ε)`. The `ε` choice trades off
quantisation noise (small `ε` is dominated by floating-point round-
off) and truncation error (large `ε` violates the linear
expansion); the standard recipe `ε = 1e−4` works well across the
demo scenes.

## Use case — pose recovery from a depth image

```rust
use molla::sensors::{Ray, RayScene, compute_grad_raycast, forward_with_loss};
use molla::math::optim::Adam;

let mut scene = RayScene::new();
let sphere_off = scene.n_params();
scene.add_sphere(sphere_off, /* radius = */ 0.5);

let rays = build_orthographic_ray_grid(64, 64);
let target_image: Vec<f64> = /* reference depth image */;

let mut params = vec![0.0, 0.0, 0.0];   // initial sphere centre
let mut opt = Adam::new(0.02);

for _ in 0..200 {
    let (_, loss) = forward_with_loss(&scene, &params, &rays, &target_image, 100.0);
    let grad = compute_grad_raycast(&scene, &params, &rays, &target_image, 100.0, 1e-4);
    opt.step_vec(&mut params, &grad);
    println!("loss = {loss:.6}");
}
```

200 Adam iterations on a 64×64 grid land the sphere centre to
within 1 mm of the target across the demo scenes — same regime as
Warp's `example_diffray.py`.

## Why not the analytic adjoint?

For a sphere at parameter offset `p`, the analytic derivative of
`t_hit` with respect to centre is

```text
∂t / ∂c = − ( o − c + t · d ) / ( d · (o − c + t · d) )
```

— easy enough for sphere + plane, less easy for capsule / cylinder
boundary cases, and tedious for the triangle / mesh path. Stage
37.7's pragmatic call was to ship FD now and analytic later when a
specific use case (e.g. a learned 3DGS supervision pipeline)
demands the speedup. The FD path runs in a few hundred ms for the
64×64-grid demo, which is fine for the contract test.

The Warp reference (`example_diffray.py`) uses `wp.Tape` over a
triangle mesh; molla's equivalent v1 trades triangle support for
analytic sphere primitives + FD, which gives a cleaner contract
test at the cost of expressiveness.

## Cross-references

- [`Sensors / Raycaster`](../sensors/raycaster.md) — forward GPU
  sensor.
- [`Sensors / TiledCamera`](../sensors/tiled_camera.md) — same
  intersection routines; the depth channel is the natural target
  image for this adjoint.
- [`Differentiable Gaussian splat renderer`](./splat.md) —
  alternative differentiable-rendering path with a full analytic
  reverse on the GPU.
- [`Math / Optimizers`](../math/optim.md) — Adam shared with the
  rest of the autograd demos.
