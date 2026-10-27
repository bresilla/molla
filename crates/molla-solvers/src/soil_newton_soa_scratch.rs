#[cfg(test)]
use super::distributed::mixed::complementarity;
use super::*;

fn fields(mut input: String, array: &str, member: &str, getter: &str, setter: &str) -> String {
    let marker = format!("{array}[");
    let mut at = 0;
    while let Some(next) = input[at..].find(&marker) {
        let start = at + next;
        let c_start = start + marker.len();
        let c_end = c_start + input[c_start..].find(']').unwrap();
        let suffix = format!(".{member}[");
        if !input[c_end + 1..].starts_with(&suffix) {
            at = c_end + 1;
            continue;
        }
        let index_start = c_end + 1 + suffix.len();
        let index_end = index_start + input[index_start..].find(']').unwrap();
        let contact = &input[c_start..c_end];
        let index = &input[index_start..index_end];
        let rest = &input[index_end + 1..];
        let (end, replacement) = if rest.starts_with('=') && !rest.starts_with("==") {
            let end = index_end + 1 + rest.find(';').unwrap();
            (
                end + 1,
                format!(
                    "{setter}({contact},{index},{});",
                    &input[index_end + 2..end]
                ),
            )
        } else {
            (index_end + 1, format!("{getter}({contact},{index})"))
        };
        input.replace_range(start..end, &replacement);
        at = start + replacement.len();
    }
    input
}

#[cfg(test)]
pub(crate) fn program(input: String, candidate: bool) -> String {
    let source = active_indices::source(complementarity::baseline(input), false);
    if candidate { transform(source) } else { source }
}

pub(crate) fn transform(mut source: String) -> String {
    replace(
        &mut source,
        "var<storage,read_write> blocks:array<Block>;",
        "var<storage,read_write> block_words:array<f32>;",
    );
    replace(
        &mut source,
        "var<storage,read_write> inverses:array<Inverse>;",
        "var<storage,read_write> inverse_words:array<f32>;",
    );
    assert_eq!(source.matches("blocks[c]=Block();").count(), 4);
    source = source.replace("blocks[c]=Block();", "soa_clear_block(c);");
    assert_eq!(source.matches("inverses[c]=Inverse();").count(), 2);
    source = source.replace("inverses[c]=Inverse();", "soa_clear_inverse(c);");
    replace(
        &mut source,
        "blocks[c].residual=fischer_residual(a);",
        "soa_store_residual(c,fischer_residual(a));",
    );
    source = fields(
        source,
        "blocks",
        "residual",
        "soa_residual",
        "soa_set_residual",
    );
    source = fields(source, "blocks", "direct", "soa_direct", "soa_set_direct");
    source = fields(
        source,
        "blocks",
        "machine",
        "soa_machine",
        "soa_set_machine",
    );
    source = fields(source, "inverses", "a", "soa_inverse", "soa_set_inverse");
    assert_eq!(source.matches("packed(blocks[c].residual)").count(), 3);
    source = source.replace("packed(blocks[c].residual)", "packed(soa_load_residual(c))");
    assert!(!source.contains("blocks["));
    assert!(!source.contains("inverses["));
    source.push_str(include_str!("../shaders/soil_newton_soa_scratch.wgsl"));
    source
}

#[cfg(test)]
pub(crate) fn logical_values(values: &[f32], inverse: bool) -> Vec<f32> {
    let width = if inverse { 36 } else { 114 };
    assert_eq!(values.len() % width, 0);
    let n = values.len() / width;
    let mut out = vec![0.0; values.len()];
    for c in 0..n {
        for i in 0..width {
            let src = if inverse || i < 6 {
                i * n + c
            } else {
                let base = if i < 78 { 6 } else { 78 };
                base * n + 2 * (((i - base) / 2) * n + c) + (i - base) % 2
            };
            out[c * width + i] = values[src];
        }
    }
    out
}

#[test]
fn newton_soa_scratch_diagnostic_bits() {
    for n in [1, 3, 256, 4096] {
        for inverse in [false, true] {
            let width = if inverse { 36 } else { 114 };
            let mut values = vec![0.0; width * n];
            let mut expected = values.clone();
            for c in 0..n {
                for i in 0..width {
                    let bits = 0x7f800000u32.wrapping_add((c * width + i) as u32);
                    expected[c * width + i] = f32::from_bits(bits);
                    let index = if inverse || i < 6 {
                        i * n + c
                    } else if i < 78 {
                        6 * n + (i - 6) / 2 * 2 * n + 2 * c + i % 2
                    } else {
                        78 * n + (i - 78) / 2 * 2 * n + 2 * c + i % 2
                    };
                    values[index] = f32::from_bits(bits);
                }
            }
            assert_eq!(
                bytemuck::cast_slice::<f32, u8>(&logical_values(&values, inverse)),
                bytemuck::cast_slice::<f32, u8>(&expected)
            );
        }
    }
}

#[test]
fn newton_soa_scratch_layout_and_source_contract() {
    for capacity in [0, 1, 3, 256, 4096] {
        let n = capacity.max(1);
        let mut occupied = vec![false; 114 * n];
        for c in 0..n {
            for row in 0..6 {
                let i = row * n + c;
                assert!(!occupied[i]);
                occupied[i] = true;
            }
            for (offset, count) in [(6, 36), (78, 18)] {
                for row in 0..count {
                    for component in 0..2 {
                        let i = offset * n + 2 * (row * n + c) + component;
                        assert!(!occupied[i]);
                        occupied[i] = true;
                    }
                }
            }
        }
        assert!(occupied.iter().all(|v| *v));
        let mut inverses = vec![false; 36 * n];
        for c in 0..n {
            for row in 0..36 {
                let i = row * n + c;
                assert!(!inverses[i]);
                inverses[i] = true;
            }
        }
        assert!(inverses.iter().all(|v| *v));
    }
    let original = shader_source_for(NewtonVariant::Fischer);
    let control = program(original.clone(), false);
    let candidate = program(original, true);
    #[cfg(feature = "soa-scratch-newton")]
    assert_eq!(
        candidate,
        ExperimentalSoilNewtonGpu::conditioned_runtime_program(
            shader_source_for(NewtonVariant::Fischer),
            true,
            256,
            SoilNewtonCapacity {
                dofs: 10,
                contacts: 4096,
                newton_iterations: 64,
                krylov_iterations: 64,
                max_owned_bytes: u64::MAX
            }
        )
    );
    for marker in ["@binding(", "var<workgroup>", "var<private>"] {
        assert_eq!(
            control.matches(marker).count(),
            candidate.matches(marker).count()
        );
    }
    for (name, source) in [("control", control), ("candidate", candidate)] {
        let words = crate::soil_newton_protected::compile(&source, "newton").unwrap();
        if let Some(root) = std::env::var_os("MOLLA_SOA_SCRATCH_SHADERS") {
            let root = std::path::PathBuf::from(root);
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(root.join(format!("{name}.wgsl")), source).unwrap();
            std::fs::write(
                root.join(format!("{name}.spv")),
                bytemuck::cast_slice(&words),
            )
            .unwrap();
        }
    }
}
