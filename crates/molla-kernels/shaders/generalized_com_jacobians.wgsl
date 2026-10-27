@group(0) @binding(9) var<storage, read_write> com_jacobians: array<f32>;
var<private> ancestors: array<u32, B>;

fn publish_com_jacobians(root: u32) {
    for (var k=0u; k<B; k++) {
        let body=model[k];
        let child=body.metadata.y;
        let center=relative_position(child,root)
            +pose_vector(vec4<f64>(poses[child].rotation),vec3<f64>(body.com.xyz));
        for (var d=0u; d<D; d++) {
            var column=zero();
            if (ancestors[k] & (1u << d)) != 0u { column=columns[d]; }
            let linear=vec3<f32>(column.linear+cross(column.angular,center));
            let angular=vec3<f32>(column.angular);
            for (var row=0u; row<3u; row++) {
                com_jacobians[(child*6u+row)*D+d]=linear[row];
                com_jacobians[(child*6u+row+3u)*D+d]=angular[row];
            }
        }
    }
}
