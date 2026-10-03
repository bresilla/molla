struct ResponseColumn { linear: vec3<f32>, angular: vec3<f32> }

fn response_width(kind: u32) -> u32 {
    if kind <= 1u { return 1u; }
    if kind == 2u { return 3u; }
    if kind >= 4u && kind <= 6u { return 6u; }
    if kind == 7u { return 2u; }
    return 0u;
}

fn response_frame(j: u32, pose_offset: u32) -> Transform {
    var parent: Transform;
    parent.position = vec4<f32>(0.0);
    parent.rotation = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    if joint_parent[j] >= 0 { parent = body_q[pose_offset + u32(joint_parent[j])]; }
    let parent_anchor = transform_compose(parent, joint_x_p[j]);
    let child_anchor = transform_compose(body_q[pose_offset + u32(joint_child[j])], joint_x_c[j]);
    var frame = parent_anchor;
    if joint_reversed[j] != 0u { frame = child_anchor; }
    let kind = joint_type[j];
    if kind >= 4u && kind <= 6u {
        frame.position = child_anchor.position;
        if joint_reversed[j] != 0u { frame.position = parent_anchor.position; }
    }
    return frame;
}

fn response_column(j: u32, d: u32, frame: Transform, reference: vec3<f32>) -> ResponseColumn {
    var result: ResponseColumn;
    result.linear = vec3<f32>(0.0);
    result.angular = vec3<f32>(0.0);
    if joint_enabled[j] == 0u || (body_flags[u32(joint_child[j])] & 3u) != 0u { return result; }
    let kind = joint_type[j];
    if kind <= 2u || kind == 6u {
        let structural=(kind<=1u) || (kind==2u && ((joint_enabled[j]>>1u)&7u)==7u)
            || (kind==6u && (d<3u || ((joint_enabled[j]>>4u)&7u)==7u));
        if structural && (joint_enabled[j]&(1u<<(d+1u)))!=0u { return result; }
    }
    var local = vec3<f32>(0.0);
    var angular = false;
    if kind <= 1u { local = joint_axis[j].xyz; angular = kind == 1u; }
    else if kind == 2u { local[d] = 1.0; angular = true; }
    else if kind >= 4u && kind <= 6u { local[d % 3u] = 1.0; angular = d >= 3u; }
    else if kind == 7u {
        local = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), d == 1u);
        angular = d == 1u;
    }
    let axis = normalize(quat_rotate(frame.rotation, local));
    let direction = select(1.0, -1.0, joint_reversed[j] != 0u);
    if angular {
        result.angular = axis * direction;
        result.linear = cross(result.angular, reference - frame.position.xyz);
    } else { result.linear = axis * direction; }
    return result;
}
