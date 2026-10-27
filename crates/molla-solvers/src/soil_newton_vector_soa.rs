use super::*;

fn accesses(input: &str, name: &str) -> String {
    let marker = format!("{name}[");
    let mut out = String::new();
    let mut at = 0;
    while let Some(relative) = input[at..].find(&marker) {
        let start = at + relative;
        if start > 0
            && (input.as_bytes()[start - 1].is_ascii_alphanumeric()
                || input.as_bytes()[start - 1] == b'_')
        {
            out.push_str(&input[at..start + marker.len()]);
            at = start + marker.len();
            continue;
        }
        out.push_str(&input[at..start]);
        let index_start = start + marker.len();
        let mut end = index_start;
        let mut depth = 1;
        while depth > 0 {
            match input.as_bytes()[end] {
                b'[' => depth += 1,
                b']' => depth -= 1,
                _ => {}
            }
            end += 1;
        }
        let index = &input[index_start..end - 1];
        let rest = &input[end..];
        let trimmed = rest.trim_start();
        if trimmed.starts_with('=') && !trimmed.starts_with("==") {
            let rhs_start = end + rest.len() - trimmed.len() + 1;
            let rhs_end = rhs_start + input[rhs_start..].find(';').unwrap();
            out.push_str(&format!(
                "{name}_soa_store({index},{});",
                accesses(&input[rhs_start..rhs_end], name)
            ));
            at = rhs_end + 1;
        } else {
            out.push_str(&format!("{name}_soa_load({index})"));
            at = end;
        }
    }
    out.push_str(&input[at..]);
    out
}

fn helpers(name: &str) -> String {
    format!(
        "\nfn {name}_soa_load(index:u32)->Six {{\n    let n=arrayLength(&{name}_words)/8u;let c=min(index,n-1u);\n    return Six(vec4<f32>({name}_words[c],{name}_words[n+c],{name}_words[2u*n+c],{name}_words[3u*n+c]),vec4<f32>({name}_words[4u*n+c],{name}_words[5u*n+c],{name}_words[6u*n+c],{name}_words[7u*n+c]));\n}}\nfn {name}_soa_store(index:u32,v:Six) {{\n    let n=arrayLength(&{name}_words)/8u;let c=min(index,n-1u);\n    for(var i=0u;i<4u;i++) {{{name}_words[i*n+c]=v.k[i];{name}_words[(4u+i)*n+c]=v.w[i];}}\n}}\n"
    )
}

pub(crate) fn program(input: String, candidate: bool) -> String {
    let mut source = soa_scratch::program(input, true);
    if candidate {
        for name in ["direction", "vector", "scratch"] {
            replace(
                &mut source,
                &format!("var<storage,read_write> {name}:array<Six>;"),
                &format!("var<storage,read_write> {name}_words:array<f32>;"),
            );
            source = accesses(&source, name);
            source.push_str(&helpers(name));
        }
    }
    source
}

#[test]
fn newton_vector_soa_parser_and_address_contract() {
    assert_eq!(
        accesses(
            "scratch[active_indices[slot]]=scaled(scratch[c],1.);rate_scratch[rate_lane]=2;",
            "scratch"
        ),
        "scratch_soa_store(active_indices[slot],scaled(scratch_soa_load(c),1.));rate_scratch[rate_lane]=2;"
    );
    for n in [1usize, 3, 31, 32, 33, 256, 4096, 3 * 4096] {
        let mut words = vec![0u32; n * 8];
        let mut seen = vec![false; n * 8];
        for c in 0..n {
            for field in 0..8 {
                let offset = field * n + c;
                assert!(!seen[offset]);
                seen[offset] = true;
                words[offset] = (c * 8 + field) as u32 ^ 0x7fc0_0000;
            }
        }
        assert!(seen.into_iter().all(|s| s));
        for c in 0..n {
            for field in 0..8 {
                assert_eq!(words[field * n + c], (c * 8 + field) as u32 ^ 0x7fc0_0000);
            }
        }
        for c in [n, n + 1, u32::MAX as usize] {
            for field in 0..8 {
                assert_eq!(
                    words[field * n + c.min(n - 1)],
                    ((n - 1) * 8 + field) as u32 ^ 0x7fc0_0000
                );
            }
        }
    }
}

#[test]
fn newton_vector_soa_source_contract() {
    let control = program(shader_source_for(NewtonVariant::Fischer), false);
    let candidate = program(shader_source_for(NewtonVariant::Fischer), true);
    for marker in ["@binding(", "var<workgroup>", "var<private>"] {
        assert_eq!(
            candidate.matches(marker).count(),
            control.matches(marker).count()
        );
    }
    assert_eq!(
        control.matches("rate_scratch[").count(),
        candidate.matches("rate_scratch[").count()
    );
    for name in ["direction", "vector", "scratch"] {
        assert!(candidate.contains(&format!("fn {name}_soa_load")));
        assert_eq!(accesses(&candidate, name), candidate);
    }
    for (name, source) in [("control", control), ("candidate", candidate)] {
        let binary = crate::soil_newton_protected::compile(&source, "newton").unwrap();
        if let Some(root) = std::env::var_os("MOLLA_VECTOR_SOA_SHADERS") {
            let root = std::path::PathBuf::from(root);
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(root.join(format!("{name}.wgsl")), source).unwrap();
            std::fs::write(
                root.join(format!("{name}.spv")),
                bytemuck::cast_slice(&binary),
            )
            .unwrap();
        }
    }
}
