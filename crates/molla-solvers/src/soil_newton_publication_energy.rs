use super::*;

const SERIAL: &str = "                let m=mass[i*params.width+j];let dj=rate_add(publication_q[j],-free_rates[j]);\n                let e=rate_mul(0.5*m,rate_add(publication_product(di,publication_q[j]),publication_product(free_rates[i],dj)));";
const START: &str =
    "    if lane==0u && valid {\n        var total=vec2<f32>(0.0);var energy_scale=1.0;";

pub(crate) fn source(mut source: String, switched: bool) -> String {
    assert!(source.contains("var<workgroup> h:array<f32,4160>;"));
    let condition = if switched {
        "params.pad2!=0u && valid"
    } else {
        "valid"
    };
    let prepare = format!(
        r#"    if {condition} {{
        for(var cell=lane;cell<params.width*params.width;cell+=256u) {{
            let i=cell/params.width;let j=cell%params.width;
            let di=rate_add(publication_q[i],-free_rates[i]);
            let m=mass[i*params.width+j];let dj=rate_add(publication_q[j],-free_rates[j]);
            let e=rate_mul(0.5*m,rate_add(publication_product(di,publication_q[j]),publication_product(free_rates[i],dj)));
            h[2u*cell]=e.x;h[2u*cell+1u]=e.y;
        }}
    }}
    workgroupBarrier();
"#
    );
    replace(&mut source, START, &format!("{prepare}{START}"));
    let cached = "vec2<f32>(h[2u*(i*params.width+j)],h[2u*(i*params.width+j)+1u])";
    let replacement = if switched {
        assert_eq!(source.matches("params.pad2").count(), 1);
        format!(
            "                var e=vec2<f32>(0.0);if params.pad2==0u {{\n{}\n                    e=rate_mul(0.5*m,rate_add(publication_product(di,publication_q[j]),publication_product(free_rates[i],dj)));\n                }} else {{e={cached};}}",
            SERIAL.lines().next().unwrap()
        )
    } else {
        format!("                let e={cached};")
    };
    replace(&mut source, SERIAL, &replacement);
    source
}

#[test]
fn newton_publication_energy_source_and_order() {
    let control = soa_scratch::program(shader_source_for(NewtonVariant::Fischer), true);
    for switched in [false, true] {
        let candidate = source(control.clone(), switched);
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
        assert!(candidate.contains("total=rate_add(total,e);energy_scale+=abs(value(e));"));
        crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
    }
    for width in 1usize..=32 {
        let mut owners = vec![0; width * width];
        for lane in 0..256 {
            for cell in (lane..width * width).step_by(256) {
                owners[cell] += 1;
                assert!(2 * cell + 1 < 4160);
                assert_eq!((cell / width) * width + cell % width, cell);
            }
        }
        assert!(owners.iter().all(|&n| n == 1));
        let order: Vec<_> = (0..width)
            .flat_map(|i| (0..width).map(move |j| i * width + j))
            .collect();
        assert_eq!(order, (0..width * width).collect::<Vec<_>>());
    }
}
