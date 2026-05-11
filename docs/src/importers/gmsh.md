# Gmsh importer

molla imports [Gmsh](https://gmsh.info/) `.msh` files into FEM
meshes via the third-party [`mshio`](https://crates.io/crates/mshio)
crate. Shipped in Stage 36.6. The importer lives at
`crates/molla/src/fem/mesh.rs` and is part of the FEM module.

Three entry points on `FemMesh`:

```rust
use molla::fem::FemMesh;

// From a UTF-8 string (e.g. include_str!):
let mesh = FemMesh::from_gmsh_str(include_str!("../assets/cube.msh"))?;

// From raw bytes (works for ASCII or binary MSH):
let mesh = FemMesh::from_gmsh_bytes(&fs::read("cube.msh")?)?;

// From a path on disk:
let mesh = FemMesh::from_gmsh_path(Path::new("cube.msh"))?;
```

## What is supported

### File format

- **MSH 4.1** — both ASCII and binary. `mshio` handles both with
  the same `parse_msh_bytes` entry point.
- Earlier MSH 2.x / 4.0 files are not supported; if you have one,
  re-export from Gmsh as 4.1 ASCII.

### Elements

The molla FEM v1 framework supports two element kinds, so the
importer only consumes those:

| MSH `ElementType` | molla `FemCell` |
|-------------------|-----------------|
| `Tri3`            | `FemCell::Tri([v0, v1, v2])` |
| `Tet4`            | `FemCell::Tet([v0, v1, v2, v3])` |

Every other element type (quad4, hex8, prism6, line2, point1,
higher-order tets, etc.) is **skipped silently with a warning**:

```text
WARN FemMesh::from_gmsh: skipped 12 Hex8 element(s) —
     only Tri3 and Tet4 are supported in v1
```

The skip is per-kind, not per-element, so one mixed mesh produces
one warning per kind rather than one per skipped element.

### Vertices

The `$Nodes` section is parsed into a flat `Vec<Vec3>`. Multi-block
node sections are flattened, and the global tag-to-flat-index map
is computed via either the per-block `node_tags` map (preferred,
present in most Gmsh outputs) or a fallback that assumes
sequential tags starting from `nodes_section.min_node_tag + base`.

The fallback is exercised when `mshio` reports
`block.node_tags = None`, which happens for some compact
single-block ASCII exports. Robustness: if an element references a
node tag missing from the map, the import returns a
`fem::Error` with a clear message.

## What is not supported

- **Quad / hex / higher-order elements.** The FEM v1 framework is
  P1-only for the scalar / vector spaces and P2-only for the
  Stage 36.3 P2 family — neither is built on quads or hexes.
  These element kinds are skipped (not errored).
- **Physical groups.** Gmsh's `$PhysicalNames` and per-element
  physical tag are parsed by `mshio` but not consumed by the
  importer. Boundary conditions in molla are applied via the
  `fem::BoundaryConditions` builder; you can't currently say "all
  elements with physical tag 7 are Dirichlet" via the MSH file.
- **Multi-domain meshes.** If the `.msh` contains multiple
  geometric regions, they get merged into a single `FemMesh`. To
  keep them separate, export each region as its own file.
- **Periodic boundaries** in `$Periodic`. Parsed by `mshio` but
  ignored.

## Worked example — heat equation on a Gmsh-authored cube

```rust
use molla::fem::{
    FemMesh, BoundaryConditions, HeatStepper, FemHeatConfig,
};

let mesh = FemMesh::from_gmsh_path(Path::new("cube.msh"))?;

let mut bcs = BoundaryConditions::new();
for (i, v) in mesh.vertices.iter().enumerate() {
    if v.z < 0.01 { bcs.add_dirichlet(i, 0.0); }
    if v.z > 0.99 { bcs.add_dirichlet(i, 1.0); }
}

let mut stepper = HeatStepper::new_cpu(&mesh, &bcs, FemHeatConfig::default())?;
for _ in 0..100 { stepper.step(0.01)?; }
let temperature: &[f64] = stepper.temperature();
```

## Authoring tips

To author a mesh in Gmsh that loads cleanly:

1. Use Gmsh 4.5+ — older versions sometimes default to MSH 2.x.
2. After the `Mesh > 3D` step, **File → Export... → MSH 4.1 ASCII**
   (the binary format also works; ASCII is easier to diff if you
   need to debug).
3. Make sure the resulting file has either `Tri3` (for 2D) or
   `Tet4` (for 3D) elements as the dominant kind. Mixed meshes
   work but you lose the non-supported elements.

A working Gmsh `.geo` script that produces a clean tet cube:

```text
SetFactory("OpenCASCADE");
Box(1) = {-0.5, -0.5, -0.5, 1.0, 1.0, 1.0};
Mesh.MeshSizeMin = 0.1;
Mesh.MeshSizeMax = 0.15;
Mesh 3;
Save "cube.msh";
```

## See also

- [`fem/mesh.md`](../fem/mesh.md) — `FemMesh` and the per-cell
  data model.
- [`fem/bc.md`](../fem/bc.md) — applying Dirichlet / Neumann
  boundary conditions to an imported mesh.
- [Gmsh documentation](https://gmsh.info/doc/texinfo/gmsh.html) —
  upstream reference.
- `mshio` [docs.rs page](https://docs.rs/mshio) — parser
  internals.
