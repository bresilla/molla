use super::*;

pub(crate) fn program(input: String, candidate: bool) -> String {
    let source = soa_scratch::program(input, true);
    if candidate { transform(source) } else { source }
}

fn transform(mut source: String) -> String {
    replace(
        &mut source,
        "var<storage,read_write> basis:array<Basis>;",
        "var<storage,read_write> basis_words:array<f32>;",
    );
    let mut at = 0;
    let mut reads = 0;
    let mut writes = 0;
    while let Some(next) = source[at..].find("basis[") {
        let start = at + next;
        let index_start = start + "basis[".len();
        let index_end = index_start + source[index_start..].find(']').unwrap();
        assert_eq!(&source[index_end + 1..index_end + 2], ".");
        let field = &source[index_end + 2..index_end + 3];
        assert!(matches!(field, "v" | "z"));
        let offset = if field == "v" { 0 } else { 8 };
        let index = &source[index_start..index_end];
        let end = index_end + 3;
        let rest = &source[end..];
        let (end, replacement) = if rest.starts_with('=') && !rest.starts_with("==") {
            let end = end + rest.find(';').unwrap();
            writes += 1;
            (
                end + 1,
                format!(
                    "basis_store({index},{offset}u,{});",
                    &source[index_end + 4..end]
                ),
            )
        } else {
            reads += 1;
            (end, format!("basis_load({index},{offset}u)"))
        };
        source.replace_range(start..end, &replacement);
        at = start + replacement.len();
    }
    assert!(reads >= 10 && writes >= 10);
    assert!(!source.contains("basis["));
    source.push_str(include_str!("../shaders/soil_newton_basis_tiles.wgsl"));
    source
}

#[test]
fn newton_basis_tiles_address_contract() {
    for count in [1usize, 2, 31, 32, 33, 63, 64, 65, 65 * 4096] {
        let mut words = vec![None; 16 * count];
        for index in 0..count {
            let first = (index / 32) * 32;
            let width = 32.min(count - first);
            for component in 0..16 {
                let at = 16 * first + component * width + index % 32;
                assert!(words[at].replace((index, component)).is_none());
            }
        }
        assert!(words.iter().all(Option::is_some));
        for (tile_index, tile) in words.chunks(16 * 32).enumerate() {
            let width = tile.len() / 16;
            for component in 0..16 {
                for lane in 0..width {
                    assert_eq!(
                        tile[component * width + lane],
                        Some((32 * tile_index + lane, component))
                    );
                }
            }
        }
    }
}

#[test]
fn newton_basis_tiles_source_contract() {
    let control = program(shader_source_for(NewtonVariant::Fischer), false);
    let candidate = program(shader_source_for(NewtonVariant::Fischer), true);
    for marker in ["@binding(", "var<workgroup>", "var<private>"] {
        assert_eq!(
            control.matches(marker).count(),
            candidate.matches(marker).count()
        );
    }
    for (name, source) in [("control", control), ("candidate", candidate)] {
        let words = crate::soil_newton_protected::compile(&source, "newton").unwrap();
        if let Some(root) = std::env::var_os("MOLLA_BASIS_TILES_SHADERS") {
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
