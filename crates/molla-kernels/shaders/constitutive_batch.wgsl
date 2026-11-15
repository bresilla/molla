struct MaterialInput { values: array<f32,16> }
struct MaterialOutput { values: array<f32,100> }
@group(0) @binding(0) var<storage,read> material_inputs: array<MaterialInput>;
@group(0) @binding(1) var<storage,read_write> material_outputs: array<MaterialOutput>;

@compute @workgroup_size(64)
fn material_batch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&material_inputs)) { return; }
    let data = material_inputs[i].values;
    var output: MaterialOutput;
    output.values[1] = 1.0;
    if (!material_finite(data[11]) || !material_finite(data[12]) || !material_finite(data[13]) || data[11] < 0.0 || data[11] > 3.0 || data[12] < 2.0 || data[12] > 3.0 || (data[13] != 0.0 && data[13] != 1.0)) {
        material_outputs[i] = output;
        return;
    }
    let model = u32(data[11]); let n = u32(data[12]);
    if (f32(model) != data[11] || f32(n) != data[12]) { material_outputs[i] = output; return; }
    let f = mat3x3<f32>(data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8]);
    let response = material_evaluate(f,n,model,data[9],data[10],data[13] != 0.0);
    output.values[0] = response.energy;
    output.values[1] = f32(1u-response.valid);
    output.values[2] = f32(n);
    for (var k = 0u; k < 9u; k++) { output.values[4u+k] = response.gradient[k/3u][k%3u]; }
    for (var k = 0u; k < 81u; k++) { output.values[16u+k] = response.hessian[k]; }
    material_outputs[i] = output;
}
