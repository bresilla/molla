use super::*;

const START: &str = "    var energy=vec2<f32>(0.0);var scale=0.0;";
const SPEED: &str = "            var speed=vec2<f32>(0.0);\n            for (var i=0u;i<params.width;i++) {speed=rate_add(speed,rate_mul(rows[c*params.width+i][axis],publication_q[i]));}";
const PREPARE: &str = r#"
    if params.pad2!=0u && !invalid_rates {
        for(var item=lane;item<3u*status.active_count;item+=256u) {
            let c=active_indices[item/3u];let axis=item%3u;
            var speed=vec2<f32>(0.0);
            for(var i=0u;i<params.width;i++) {speed=rate_add(speed,rate_mul(rows[c*params.width+i][axis],publication_q[i]));}
            basis[c].v.k[axis]=speed.x;basis[c].v.w[axis]=speed.y;
        }
    }
    storageBarrier();
"#;

pub(crate) fn source(mut source: String) -> String {
    assert!(source.starts_with(crate::soil_newton_protected::PREFIX));
    assert!(source.contains(&format!("{}256\n", crate::soil_newton_width::MARKER)));
    assert_eq!(source.matches("params.pad2").count(), 0);
    assert_eq!(source.matches(START).count(), 1);
    assert_eq!(source.matches(SPEED).count(), 1);
    assert!(source.contains("basis[c].v=scaled(scratch[c],1.0/beta);"));
    assert!(source.contains("conditioned_shared_ready=false;"));
    source = source.replace(START, &format!("{PREPARE}{START}"));
    source = source.replace(SPEED, &format!("            var speed=vec2<f32>(0.0);if params.pad2==0u {{\n{}\n            }} else {{speed=vec2<f32>(basis[c].v.k[axis],basis[c].v.w[axis]);}}", SPEED.split_once('\n').unwrap().1));
    source
}

#[test]
fn newton_publication_projection_source_and_ownership() {
    let control = soa_scratch::program(shader_source_for(NewtonVariant::Fischer), true);
    let candidate = source(control.clone());
    for marker in [
        "@binding(",
        "var<workgroup>",
        "var<storage",
        "publication_fault(",
    ] {
        assert_eq!(
            candidate.matches(marker).count(),
            control.matches(marker).count()
        );
    }
    for count in [0usize, 1, 63, 64, 65, 251, 256, 257, 4096, 16384] {
        let mut owners = vec![0; 3 * count];
        for lane in 0..256 {
            for item in (lane..owners.len()).step_by(256) {
                owners[item] += 1;
                assert!(item / 3 < count && item % 3 < 3);
            }
        }
        assert!(owners.iter().all(|&n| n == 1));
    }
    crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
}
