const REGIONS: [(&str, &str, &str); 2] = [
    (
        "    if lane<params.width && !invalid_inputs {",
        "    workgroupBarrier();storageBarrier();\n    if lane==0u {publication_stop",
        include_str!("../shaders/soil_newton_publication_products.wgsl"),
    ),
    (
        "    if lane<params.width && valid {",
        "    if lane==0u && valid {",
        include_str!("../shaders/soil_newton_publication_reactions.wgsl"),
    ),
];

pub(crate) fn source(mut source: String) -> String {
    assert_eq!(
        source.matches("var<workgroup> h:array<f32,4160>;").count(),
        1
    );
    for (start, end, replacement) in REGIONS {
        assert_eq!(source.matches(start).count(), 1);
        let first = source.find(start).unwrap();
        let last = first + source[first..].find(end).unwrap();
        source.replace_range(first..last, replacement);
    }
    source
}

#[cfg(test)]
pub(crate) fn probe_source(input: String, candidate: bool) -> String {
    let input = super::active_publication::probe_source(input, true);
    if candidate { source(input) } else { input }
}

#[cfg(test)]
pub(crate) fn adaptive_source(mut source: String) -> String {
    for (_, _, region) in REGIONS {
        let axes = if region.contains("item<count*params.width*3u") {
            3
        } else {
            1
        };
        let adaptive = region
            .replace("tile+=16u", "tile+=publication_tile")
            .replace("min(16u,", "min(publication_tile,")
            .replace(
                "        for(var tile=0u;",
                &format!("        let publication_tile=2080u/({axes}u*params.width);\n        for(var tile=0u;"),
            );
        super::replace(&mut source, region, &adaptive);
    }
    source
}

#[test]
fn newton_publication_adaptive_order_and_bounds() {
    for width in 1usize..=32 {
        for axes in [1, 3] {
            let capacity = 2080 / (width * axes);
            assert!(capacity >= 16);
            assert!(2 * capacity * width * axes <= 4160);
            let mut counts = vec![0, 1, 16, 256, 601, 4096];
            counts.extend([capacity - 1, capacity, capacity + 1]);
            for count in counts {
                for dof in 0..width {
                    let mut order = Vec::new();
                    for tile in (0..count).step_by(capacity) {
                        let size = (count - tile).min(capacity);
                        for local in 0..size {
                            for axis in 0..axes {
                                let item = (local * width + dof) * axes + axis;
                                let owner = item % 256;
                                assert!(owner < 256);
                                assert_eq!(item / axes % width, dof);
                                assert!(2 * item + 1 < 4160);
                                order.push((tile + item / (axes * width), item % axes));
                            }
                        }
                    }
                    assert_eq!(
                        order,
                        (0..count)
                            .flat_map(|c| (0..axes).map(move |a| (c, a)))
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
    }
}

#[test]
fn newton_publication_products_preserve_order_and_scratch_bounds() {
    use crate::soil_newton_gpu::{
        ExperimentalSoilNewtonGpu as Solver, NewtonVariant, shader_source_for,
    };
    let control = super::active_publication::source(super::norm_tree::source(
        super::schur_inverse::source(super::core_source(super::source(
            Solver::cached_contact_source(Solver::cached_delta_source(
                Solver::stalled_release_source(shader_source_for(NewtonVariant::Fischer)),
            )),
        ))),
    ));
    let candidate = source(control.clone());
    let mut restored = candidate.clone();
    for (start, end, replacement) in REGIONS {
        let first = control.find(start).unwrap();
        let last = first + control[first..].find(end).unwrap();
        super::replace(&mut restored, replacement, &control[first..last]);
    }
    assert_eq!(restored, control);
    assert!(control.contains("for (var i=lane;i<4160u;i+=64u) {h[i]=0.0;}"));
    for width in [1usize, 3, 10, 31, 32] {
        for count in [0usize, 1, 15, 16, 17, 63, 64, 65, 255, 256, 257, 601, 4096] {
            for dof in 0..width {
                let mut order = Vec::new();
                for tile in (0..count).step_by(16) {
                    let size = (count - tile).min(16);
                    let mut owners = vec![0; size * width * 3];
                    for lane in 0..256 {
                        for item in (lane..owners.len()).step_by(256) {
                            owners[item] += 1;
                            assert!(2 * item + 1 < 4160);
                        }
                    }
                    assert!(owners.iter().all(|&n| n == 1));
                    for local in 0..size {
                        for axis in 0..3 {
                            let item = (local * width + dof) * 3 + axis;
                            assert_eq!(item / 3 % width, dof);
                            order.push((tile + item / (3 * width), item % 3));
                        }
                    }
                }
                let expected: Vec<_> = (0..count)
                    .flat_map(|c| (0..3).map(move |axis| (c, axis)))
                    .collect();
                assert_eq!(order, expected);
            }
        }
    }
    crate::soil_newton_protected::compile(
        &format!(
            "{}{}256\n{candidate}",
            crate::soil_newton_protected::VOLATILE_MARKER,
            crate::soil_newton_width::MARKER
        ),
        "newton",
    )
    .unwrap();
}
