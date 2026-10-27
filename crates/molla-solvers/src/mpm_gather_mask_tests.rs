use super::*;

#[test]
fn gather_masks_validate_both_shader_backends() {
    for base in [
        include_str!("../shaders/mpm.wgsl").to_owned(),
        native_atomics::shader_source(),
    ] {
        let source = format!(
            "{base}\n{}\n{}",
            include_str!("../shaders/mpm_deterministic_p2g.wgsl"),
            include_str!("../shaders/mpm_gather_masks.wgsl"),
        );
        let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}

#[test]
fn gather_masks_preserve_contributor_order_and_partial_chunks() {
    let mut checked = 0;
    for tile_base in [[0i32; 3], [4, 8, 12], [-8, 4, 0]] {
        for count in [0, 1, 31, 32, 33, 63, 64] {
            for phase in 0..64 {
                let points: Vec<[i32; 3]> = (0..count)
                    .map(|p| {
                        let p = p + phase;
                        [0, 1, 2]
                            .map(|axis| tile_base[axis] + ((p * (axis * 2 + 1)) % 9) as i32 - 3)
                    })
                    .collect();
                let mut masks = [[0u32; 2]; 64];
                for (pid, base) in points.iter().enumerate() {
                    let lo = [0, 1, 2].map(|i| base[i].max(tile_base[i]));
                    let hi = [0, 1, 2].map(|i| (base[i] + 2).min(tile_base[i] + 3));
                    for z in lo[2]..=hi[2] {
                        for y in lo[1]..=hi[1] {
                            for x in lo[0]..=hi[0] {
                                let node = (x - tile_base[0])
                                    + 4 * ((y - tile_base[1]) + 4 * (z - tile_base[2]));
                                masks[node as usize][pid / 32] |= 1 << (pid % 32);
                            }
                        }
                    }
                }
                for (node, mask) in masks.into_iter().enumerate() {
                    let cell = [node % 4, node / 4 % 4, node / 16];
                    let scalar: Vec<_> = points
                        .iter()
                        .enumerate()
                        .filter(|(_, base)| {
                            (0..3).all(|axis| {
                                (0..=2)
                                    .contains(&(tile_base[axis] + cell[axis] as i32 - base[axis]))
                            })
                        })
                        .map(|(pid, _)| pid)
                        .collect();
                    let mut selected = Vec::new();
                    for (word, mut bits) in mask.into_iter().enumerate() {
                        while bits != 0 {
                            selected.push(word * 32 + bits.trailing_zeros() as usize);
                            bits &= bits - 1;
                        }
                    }
                    assert_eq!(selected, scalar);
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 86_016);
}

#[test]
#[ignore = "requires GPU; exact workgroup-mask grid and full particle evolution"]
fn gather_masks_exact_grid_and_evolution() {
    pruning_tests::exact_grid_variant(false, true);
}

#[test]
#[ignore = "requires GPU and saved50k fixture; paired workgroup-mask transfer timing"]
fn gather_masks_saved_50k_transfer_pairs() {
    pruning_tests::saved_50k_variant(false, true);
}
