struct PairedVelocity { linear: PositionPair, angular: PositionPair }

fn body_rate_finite(value: SpatialVecGpu) -> bool {
    let exponent=vec4<u32>(0x7f800000u);
    return all((bitcast<vec4<u32>>(value.linear)&exponent)!=exponent)
        && all((bitcast<vec4<u32>>(value.angular)&exponent)!=exponent);
}
fn body_rate_input_valid(body: u32) -> bool {
    return body_rate_finite(body_qd[body]) && body_rate_finite(body_velocity_low[body]);
}

fn rate_parent_rotation(j: u32) -> QuaternionPair {
    let parent=joint_parent[j];
    if parent<0 { return QuaternionPair(vec4<f32>(0.0,0.0,0.0,1.0),vec4<f32>(0.0)); }
    if (body_flags[u32(parent)]&3u)==0u {
        for (var back=1u; back<=min(j,MAX_JOINTS_PER_ART); back++) {
            let previous=j-back;
            if joint_child[previous]==parent { return fk_rotations[previous%MAX_JOINTS_PER_ART]; }
        }
    }
    return normalize_pair(u32(joint_child[j]),QuaternionPair(body_q[u32(parent)].rotation,vec4<f32>(0.0)));
}
fn rate_position(body: u32) -> PositionPair {
    return PositionPair(body_q[body].position.xyz,position_residue(body));
}
fn rate_difference(slot: u32, a: PositionPair, b: PositionPair) -> PositionPair {
    return position_add(slot,a,PositionPair(-b.high,-b.low));
}
fn rate_local_velocity(j: u32) -> PairedVelocity {
    let slot=u32(joint_child[j]);
    let kind=select(JOINT_FIXED,joint_type[j],joint_enabled[j]!=0u);
    let dof=joint_dof_offset[j];
    let zero=PositionPair(vec3<f32>(0.0),vec3<f32>(0.0));
    var result=PairedVelocity(zero,zero);
    if kind==JOINT_PRISMATIC || kind==JOINT_REVOLUTE {
        var axial=zero;
        for (var axis=0u; axis<3u; axis++) {
            let value=quaternion_scalar_product(slot,paired_rates[dof],
                vec2<f32>(joint_axis[j][axis],joint_frame_references[j].axis[axis]));
            axial.high[axis]=value.x; axial.low[axis]=value.y;
        }
        if kind==JOINT_PRISMATIC { result.linear=axial; } else { result.angular=axial; }
    }
    if kind==JOINT_BALL || kind==JOINT_FREE || kind==JOINT_DISTANCE || kind==JOINT_D6 {
        let angular_start=dof+select(3u,0u,kind==JOINT_BALL);
        for (var axis=0u; axis<3u; axis++) {
            let angular=paired_rates[angular_start+axis];
            result.angular.high[axis]=angular.x; result.angular.low[axis]=angular.y;
            if kind!=JOINT_BALL {
                let linear=paired_rates[dof+axis];
                result.linear.high[axis]=linear.x; result.linear.low[axis]=linear.y;
            }
        }
    }
    if kind==JOINT_CABLE {
        result.linear.high.z=paired_rates[dof].x; result.linear.low.z=paired_rates[dof].y;
        result.angular.high.x=paired_rates[dof+1u].x; result.angular.low.x=paired_rates[dof+1u].y;
    }
    for (var d=0u; d<joint_dof_offset[j+1u]-dof; d++) {
        if motion_axis_point(j,d) {
            if kind==JOINT_PRISMATIC { result.linear=zero; }
            if kind==JOINT_REVOLUTE { result.angular=zero; }
            if kind==JOINT_BALL && motion_axis_point(j,0u) && motion_axis_point(j,1u) && motion_axis_point(j,2u) {
                result.angular=zero;
            }
            if kind==JOINT_D6 {
                if d<3u { result.linear.high[d]=0.0; result.linear.low[d]=0.0; }
                else if d6_angular_lock_count(j)==3u { result.angular=zero; }
            }
        }
    }
    return result;
}
fn publish_paired_body_velocity(j: u32) {
    let child=u32(joint_child[j]);
    let parent=joint_parent[j];
    let kind=joint_type[j];
    let reference=joint_frame_references[j];
    let saved=fk_pair_rounding;
    fk_pair_rounding=true;
    let parent_rotation=rate_parent_rotation(j);
    let child_rotation=fk_rotations[j%MAX_JOINTS_PER_ART];
    let xp=joint_x_p[j]; let xc=joint_x_c[j];
    let parent_anchor_rotation=multiply(child,parent_rotation,QuaternionPair(xp.rotation,reference.parent_rotation));
    let child_anchor_rotation=multiply(child,child_rotation,QuaternionPair(xc.rotation,reference.child_rotation));
    var parent_position=PositionPair(vec3<f32>(0.0),vec3<f32>(0.0));
    if parent>=0 { parent_position=rate_position(u32(parent)); }
    let parent_anchor=position_add(child,parent_position,
        position_pair_rotate(child,parent_rotation,PositionPair(xp.position.xyz,reference.parent_position.xyz)));
    let child_center=position_add(child,rate_position(child),
        position_pair_rotate(child,child_rotation,PositionPair(body_com[child].xyz,vec3<f32>(0.0))));
    let child_anchor=position_add(child,rate_position(child),
        position_pair_rotate(child,child_rotation,PositionPair(xc.position.xyz,reference.child_position.xyz)));
    var frame_rotation=parent_anchor_rotation;
    var origin=parent_anchor;
    let reversed=joint_reversed[j]!=0u;
    if reversed { frame_rotation=child_anchor_rotation; origin=child_anchor; }
    if kind==JOINT_FREE || kind==JOINT_DISTANCE || kind==JOINT_D6 {
        origin=child_anchor;
        if reversed { origin=parent_anchor; }
    }
    if kind==JOINT_CABLE {
        let coord=joint_coord_offset[j];
        let extension=PositionPair(vec3<f32>(0.0,0.0,joint_q[coord]),
            vec3<f32>(0.0,0.0,bitcast<f32>(atomicLoad(&coordinate_error[coord]))));
        origin=position_add(child,origin,position_pair_rotate(child,frame_rotation,extension));
    }
    let zero=PositionPair(vec3<f32>(0.0),vec3<f32>(0.0));
    var velocity=PairedVelocity(zero,zero);
    if parent>=0 {
        let p=u32(parent);
        velocity.linear=PositionPair(body_qd[p].linear.xyz,body_velocity_low[p].linear.xyz);
        velocity.angular=PositionPair(body_qd[p].angular.xyz,body_velocity_low[p].angular.xyz);
        let parent_center=position_add(child,parent_position,
            position_pair_rotate(child,parent_rotation,PositionPair(body_com[p].xyz,vec3<f32>(0.0))));
        velocity.linear=position_add(child,velocity.linear,
            paired_cross(child,velocity.angular,rate_difference(child,child_center,parent_center)));
    }
    let local=rate_local_velocity(j);
    let direction=select(1.0,-1.0,reversed);
    let linear=position_pair_rotate(child,frame_rotation,PositionPair(direction*local.linear.high,direction*local.linear.low));
    let angular=position_pair_rotate(child,frame_rotation,PositionPair(direction*local.angular.high,direction*local.angular.low));
    velocity.linear=position_add(child,position_add(child,velocity.linear,linear),
        paired_cross(child,angular,rate_difference(child,child_center,origin)));
    velocity.angular=position_add(child,velocity.angular,angular);
    let high=SpatialVecGpu(vec4<f32>(velocity.linear.high,0.0),vec4<f32>(velocity.angular.high,0.0));
    let low=SpatialVecGpu(vec4<f32>(velocity.linear.low,0.0),vec4<f32>(velocity.angular.low,0.0));
    if !body_rate_finite(high) || !body_rate_finite(low) {
        atomicOr(&stop_status,32u); fk_pair_rounding=saved; return;
    }
    body_qd[child]=high;
    body_velocity_low[child]=low;
    fk_pair_rounding=saved;
}
