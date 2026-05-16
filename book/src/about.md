# About this book

This is the long-form documentation for `molla`, the pure-Rust
wgpu-accelerated physics engine. Source lives at
[github.com/bresilla/molla](https://github.com/bresilla/molla).

## Authors

- **Trim Bresilla** — project lead, all engineering.

## Credits and lineage

molla is a clean-room reimplementation. The reference codebases under
`xtra/` are read for behaviour, data layout, and test oracles — they
are not copied. Specifically:

- [NVIDIA Newton](https://github.com/newton-physics/newton) is the
  reference physics engine. `Model` / `State` / `Control` / `Contacts`
  shape is borrowed; the solver inventory tracks Newton's; trajectory
  tests run the same scene in both engines and assert numerical
  agreement.
- [NVIDIA Warp](https://github.com/NVIDIA/warp) provides the compute
  primitives reference — raycast, FFT, marching cubes, sparse-grid
  insert/lookup, tile primitives (matmul, cholesky, convolution,
  stream-compaction, MCGP). ~30 viewer scenes are direct ports of
  `warp/examples/*.py`.

Both are Apache-2.0. molla itself is MIT-licensed (see `LICENSE` at
the repository root).

## Acknowledgements

- The Rust `wgpu` team for a Vulkan / Metal / DX12 / WebGPU runtime
  that just works.
- The Bevy maintainers for an ECS that the viewer rides on without
  fighting it.
- The `glam` team — host `f64` math + device `f32` math from the
  same type names is a small thing that saved a lot of churn.

## Contributing

molla is open source. Pull requests are welcome. Before you start a
substantial change, please:

1. Read `PLAN.md` at the repository root — it is the source of
   truth for scope, locked decisions, and the staged delivery
   roadmap.
2. Read [`architecture/design_rules.md`](./architecture/design_rules.md)
   for the conventions every change has to respect (CPU↔GPU
   parity, BSR `B=1`, persistent buffers, atomic-CAS pattern, no
   `cooperative_matrix`).
3. For new solvers / kernels, follow the existing scaffold:
   - Put reusable implementation in the owning functionality crate
     under `crates/` (`molla-solvers`, `molla-fem`,
     `molla-sensors`, etc.).
   - Put shared WGSL dispatch wrappers and shaders in
     `crates/molla-kernels` unless the kernel is intentionally owned
     by a functional crate such as `molla-fem`.
   - Keep CPU reference/oracle code beside the functionality it
     verifies.
   - Put parity tests and Criterion benches in the owning crate's
     `tests/` and `benches/` directories, gated by `#[ignore]` /
     `#[cfg(feature = "gpu")]` and `required-features = ["gpu"]`
     where appropriate.
   - Use `crates/molla` tests/benches only for facade compatibility
     or top-level product coverage.
   - If the feature has a user-visible behaviour: a viewer scene.

## License

```text
MIT License — copyright (c) 2026 Trim Bresilla
```

See `LICENSE` at the repository root for the full text.

## Citing molla

If molla helps your research:

```bibtex
@software{molla,
  author = {Bresilla, Trim},
  title  = {molla — a pure-Rust, wgpu-accelerated physics engine},
  url    = {https://github.com/bresilla/molla},
  year   = {2026}
}
```
