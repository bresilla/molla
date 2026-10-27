@group(0) @binding(0) var<storage, read> previous: array<u32>;
@group(0) @binding(1) var<storage, read> upload: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_words: array<u32>;

const STRIDES: array<u32, 10> = array<u32, 10>(4u, 4u, 12u, 12u, 1u, 1u, 1u, 1u, 4u, 2u);

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let old_count = upload[0];
    let new_count = upload[1];
    let destination = gid.x;
    if destination >= new_count { return; }
    let mapping = upload[4u + destination];
    let imported = (mapping & 0x80000000u) != 0u;
    let source = mapping & 0x7fffffffu;
    let import_base = 4u + new_count + source * 42u;
    var prefix = 0u;
    for (var stream = 0u; stream < 10u; stream++) {
        let stride = STRIDES[stream];
        for (var word = 0u; word < stride; word++) {
            var value: u32;
            if imported {
                value = upload[import_base + prefix + word];
            } else {
                value = previous[prefix * old_count + source * stride + word];
            }
            output_words[prefix * new_count + destination * stride + word] = value;
        }
        prefix += stride;
    }
}
