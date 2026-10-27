fn reduce_parts()->vec2<f32> {
    workgroupBarrier();
    for (var stride=32u;stride>0u;stride/=2u) {
        if lane<stride {parts[lane]=rate_add(parts[lane],parts[lane+stride]);}
        workgroupBarrier();
    }
    return parts[0];
}
