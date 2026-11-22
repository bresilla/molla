# Apple-tree skeleton provenance

**Tian Qiu, AppleQSM-Row13 (2024), version 1.**
Dataset: https://doi.org/10.6084/m9.figshare.26215094.v1
License: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).

Method: Qiu et al., *AppleQSM: Geometry-Based 3D Characterization of Apple
Tree Architecture in Orchards*, Plant Phenomics 6, 0179 (2024).
https://doi.org/10.34133/plantphenomics.0179

Reconstruction: [Trim Bresilla, Appleskel](https://github.com/bresilla/appleskel),
local checkout `9731e7ebd7ccf8d20c6fe761490fbcdd8e2fe83a`.

`apple_tree1.json` is the `tree_id` and `geometry` subset of:

```
../../../../OUSD/appleskel/examples/pipeline_output/apple_trees/tree1/
  Characterization/paper_dataset/tree1_characterization.json
```

SHA-256 of source JSON:
`f24f4619da147ead00e218747da3afd346351d76e8cfdd69c9c9cde2bc3beab4`.

SHA-256 of bundled geometry JSON:
`60a265cd903b88b3e65c76dfc71b231abaa0655263e9621ce6f9c88f3623ea70`.

Regenerate the subset using `make import-apple-tree`; `APPLE_TREE_INPUT`
overrides the source. Update these hashes and attribution when changing it.
The Appleskel source scan was ground-cropped, centered, skeletonized and
refined by its reconstruction pipeline. Molla then changes axes from Z-up
to Y-up, grounds the root, smooths and downsamples paths, retains a bounded
parent-connected branch subset, and creates tapered solid cells.

The reconstructed geometry and interactive material are not validated
scientific measurements. Radii, stiffness, and damping used in the demo are
approximations. No leaves or fruit are added, and no endorsement is implied.
The dataset attribution is retained separately from Molla's code license.
