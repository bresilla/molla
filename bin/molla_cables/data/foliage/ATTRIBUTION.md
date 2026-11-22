# Foliage textures

All source scans are from [ambientCG](https://ambientcg.com) and are released
under [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).
Attribution is not required; it is given here for provenance.

| Asset | Use | Download (2K-JPG) | SHA-256 of zip |
|---|---|---|---|
| [Leaf001](https://ambientcg.com/view?id=Leaf001) | 2 leaves | `Leaf001_2K-JPG.zip` | `5e1dcb8bfa04734cc3964c5f673b002cd03110ecb20a110db9cc20095f87d8bb` |
| [LeafSet001](https://ambientcg.com/view?id=LeafSet001) | 6 leaves | `LeafSet001_2K-JPG.zip` | `26b43318e64ae442241f396309e78ee772dac53577bd92edc450d294b46ae9ba` |
| [LeafSet003](https://ambientcg.com/view?id=LeafSet003) | 4 leaves | `LeafSet003_2K-JPG.zip` | `3b6b8a128c4809f9a924fb13f769a014c8258c25a19c77b2f658369363afc000` |
| [LeafSet014](https://ambientcg.com/view?id=LeafSet014) | 6 leaves | `LeafSet014_2K-JPG.zip` | `8065d616032cc3a2dfa7ff40ba716e483b7fc854ae8c8f03fa95ada2cb7fccc6` |
| [LeafSet022](https://ambientcg.com/view?id=LeafSet022) | 6 leaves | `LeafSet022_2K-JPG.zip` | `2af6b5d525c16b32dd8574ae07f37a78a42b198327500777a6512e041644f9c7` |
| [LeafSet024](https://ambientcg.com/view?id=LeafSet024) | 9 leaves | `LeafSet024_2K-JPG.zip` | `5e750c97b02717b94284128704039563acdd40e3b01a2b4a852c08eff61f5621` |
| [Bark004](https://ambientcg.com/view?id=Bark004) | trunk bark | `Bark004_2K-JPG.zip` | `1ef152366455b01423e1e8498db430fea691bda8cb388f235cbec49bd94eb7ea` |
| [Bark003](https://ambientcg.com/view?id=Bark003) | limb bark | `Bark003_2K-JPG.zip` | `8a46c556a103194f314bd3e6414746fba7b65a7964cb99e47f8cc65360a9318d` |
| [Bark005](https://ambientcg.com/view?id=Bark005) | young-shoot bark | `Bark005_2K-JPG.zip` | `8d48cc49ed3f1a655455ae2e746a8a1cfdd6447562d8127fd89f82acff527a96` |

Downloaded 2026-11-19. `make import-foliage-textures` re-downloads, checks the
hashes and rebakes this directory with `molla_foliage_bake`.

## Modifications

- Each of the 33 leaves is cut out of its atlas by its opacity mask, aligned
  from centroid to tip, stripped of its petiole and resampled (3×3 supersampled)
  so its smoothed outline fills Molla's simulated blade outline. Serrations,
  holes, spots and veins are kept. Normal maps are rotated with the leaf.
- Leaf colors and normals are bled into transparent texels for mip filtering.
- Bark color and OpenGL normal maps are resized to 1024² tiles; Bark005 is
  stored square and restored to its 2:1 aspect in the shader.
- Packed outputs: `leaves-color.jpg`, `leaves-alpha.png`, `leaves-normal.jpg`
  (11 × 3 grid of 352×512 layers), `bark-color.jpg`, `bark-normal.jpg`
  (three stacked 1024² layers).
