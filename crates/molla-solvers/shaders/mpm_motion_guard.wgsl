@group(0) @binding(1) var<uniform> soil_motion_bound: vec4<f32>;

fn soil_motion_admitted(before: vec3<f32>, after: vec3<f32>) -> bool {
    for (var axis = 0u; axis < 3u; axis++) {
        let old_bits = bitcast<u32>(before[axis]);
        let new_bits = bitcast<u32>(after[axis]);
        let old_abs = old_bits & 0x7fffffffu;
        let new_abs = new_bits & 0x7fffffffu;
        if old_abs >= 0x7f800000u || new_abs >= 0x7f800000u { return false; }
        if old_bits == new_bits { continue; }
        if old_abs == 0u && new_abs == 0u { continue; }
        if (old_abs != 0u && old_abs < 0x00800000u) || (new_abs != 0u && new_abs < 0x00800000u) { return false; }
        if ((old_bits ^ new_bits) & 0x80000000u) != 0u && (old_abs > 0x7effffffu || new_abs > 0x7effffffu) { return false; }
        let difference = abs(after[axis] - before[axis]);
        let bits = bitcast<u32>(difference);
        if bits >= 0x7f800000u { return false; }
        let upper = bitcast<f32>(max(bits, 0x00800000u) + 1u);
        if upper > soil_motion_bound[axis] { return false; }
    }
    return true;
}
