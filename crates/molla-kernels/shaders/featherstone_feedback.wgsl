alias VelocityBoxMatrix = array<array<f32, MAX_FEEDBACK_ROWS>, MAX_FEEDBACK_ROWS>;
alias VelocityBoxVector = array<f32, MAX_FEEDBACK_ROWS>;

struct RootDriveLocks {
    enabled: bool,
    inverse: array<f32,36>,
    projection: array<f32,36>,
    desired_rate: array<f32,6>,
}
var<private> root_drive_locks: RootDriveLocks;

const MAX_DRIVE_LOCK_ROWS: u32 = max(1u, 2u * MAX_JOINTS_PER_ART);
alias DriveLockRows = array<MixedStopRow, MAX_DRIVE_LOCK_ROWS>;
alias DriveLockVector = array<f32, MAX_DRIVE_LOCK_ROWS>;
alias DriveLockMatrix = array<DriveLockVector, MAX_DRIVE_LOCK_ROWS>;
struct ArticulationDriveLocks {
    count: u32,
    rank: u32,
    rows: DriveLockRows,
    bounds: DriveLockVector,
    scales: DriveLockVector,
    indices: array<u32, MAX_DRIVE_LOCK_ROWS>,
    factor: DriveLockMatrix,
}
var<private> articulation_drive_locks: ArticulationDriveLocks;

fn drive_feedback_requested(begin: u32, n: u32, counts: ptr<function, StopCounts>) -> bool {
    for (var k=0u;k<n;k++) {
        let first=joint_dof_offset[begin+k];
        for (var d=0u;d<(*counts)[k];d++) {
            let mode=joint_target_mode[first+d];
            if (mode>=TARGET_POS && mode<=TARGET_PV
                && (uniforms.stable_motor_feedback!=0u || joint_motor_model[first+d]==1u))
                || joint_brake[first+d]>0.0 { return true; }
        }
    }
    return false;
}

fn prepare_drive_locks(begin: u32, n: u32, counts: ptr<function, StopCounts>,
    sl: ptr<function, StopVectors>, sa: ptr<function, StopVectors>, ul: ptr<function, StopVectors>, ua: ptr<function, StopVectors>,
    inverse: ptr<function, StopBlocks>) -> bool {
    articulation_drive_locks.count=0u;
    articulation_drive_locks.rank=0u;
    if !prepare_root_drive_locks(begin,n,counts,inverse) { return false; }
    if root_drive_locks.enabled || !drive_feedback_requested(begin,n,counts) { return true; }
    var parents: StopParents;
    var candidates: MixedStopRows;
    var lo: MixedStopBounds;
    var hi: MixedStopBounds;
    var rows=0u;
    for (var k=0u;k<n;k++) {
        let j=begin+k;
        parents[k]=-1;
        for (var p=0u;p<k;p++) { if joint_child[begin+p]==joint_parent[j] { parents[k]=i32(p); } }
        if joint_enabled[j]==0u || (body_flags[u32(joint_child[j])] & 3u)!=0u
            || (joint_type[j]!=JOINT_BALL && joint_type[j]!=JOINT_D6) { continue; }
        let locked=mixed_angular_lock_count(j);
        if locked==0u || locked==3u { continue; }
        let q=mixed_joint_rotation(j);
        if !mixed_angular_rows(k,j,q,joint_dof_offset[j],uniforms.dt,&candidates,&lo,&hi,&rows) { return false; }
    }
    for (var r=0u;r<rows;r++) {
        if lo[r]!=hi[r] { continue; }
        let index=articulation_drive_locks.count;
        if index==MAX_DRIVE_LOCK_ROWS { return false; }
        articulation_drive_locks.rows[index]=candidates[r];
        articulation_drive_locks.bounds[index]=lo[r];
        articulation_drive_locks.count++;
    }
    let count=articulation_drive_locks.count;
    if count==0u { return true; }
    var gram: DriveLockMatrix;
    var unit: StopScalars;
    for (var c=0u;c<count;c++) {
        let row=articulation_drive_locks.rows[c];
        for (var d=0u;d<6u;d++) { unit[row.joint][d]=row.gradient[d]; }
        var response=stop_mass_response(n,&parents,counts,sl,sa,ul,ua,inverse,&unit);
        for (var r=0u;r<count;r++) { gram[r][c]=mixed_dot(articulation_drive_locks.rows[r],&response); }
        for (var d=0u;d<6u;d++) { unit[row.joint][d]=0.0; }
    }
    for (var r=0u;r<count;r++) {
        if !(gram[r][r]>=0.0 && xyz_chart_finite(gram[r][r])) { return false; }
        articulation_drive_locks.scales[r]=sqrt(gram[r][r]);
        for (var c=0u;c<r;c++) {
            let value=0.5*(gram[r][c]+gram[c][r]);
            if !xyz_chart_finite(value) { return false; }
            gram[r][c]=value; gram[c][r]=value;
        }
    }
    for (var row=0u;row<count;row++) {
        let scale=articulation_drive_locks.scales[row];
        if scale==0.0 { continue; }
        let rank=articulation_drive_locks.rank;
        var column: DriveLockVector;
        var diagonal=1.0;
        for (var c=0u;c<rank;c++) {
            let other=articulation_drive_locks.indices[c];
            var value=gram[row][other]/scale/articulation_drive_locks.scales[other];
            for (var d=0u;d<c;d++) { value=fma(-column[d],articulation_drive_locks.factor[c][d],value); }
            column[c]=value/articulation_drive_locks.factor[c][c];
            diagonal=fma(-column[c],column[c],diagonal);
        }
        if !xyz_chart_finite(diagonal) || diagonal < -1e-5 { return false; }
        if diagonal<=1e-6 { continue; }
        for (var c=0u;c<rank;c++) { articulation_drive_locks.factor[rank][c]=column[c]; }
        articulation_drive_locks.factor[rank][rank]=sqrt(diagonal);
        articulation_drive_locks.indices[rank]=row;
        articulation_drive_locks.rank++;
    }
    return true;
}

fn solve_drive_locks(rhs: DriveLockVector) -> DriveLockVector {
    let rank=articulation_drive_locks.rank;
    var solution: DriveLockVector;
    for (var r=0u;r<rank;r++) {
        let row=articulation_drive_locks.indices[r];
        var value=rhs[row]/articulation_drive_locks.scales[row];
        for (var c=0u;c<r;c++) { value=fma(-articulation_drive_locks.factor[r][c],solution[c],value); }
        solution[r]=value/articulation_drive_locks.factor[r][r];
    }
    for (var back=rank;back>0u;back--) {
        let r=back-1u;
        var value=solution[r];
        for (var c=r+1u;c<rank;c++) { value=fma(-articulation_drive_locks.factor[c][r],solution[c],value); }
        solution[r]=value/articulation_drive_locks.factor[r][r];
    }
    var result: DriveLockVector;
    for (var r=0u;r<rank;r++) {
        let row=articulation_drive_locks.indices[r];
        result[row]=solution[r]/articulation_drive_locks.scales[row];
    }
    return result;
}

fn project_drive_locks(n: u32, parents: ptr<function, StopParents>, counts: ptr<function, StopCounts>,
    sl: ptr<function, StopVectors>, sa: ptr<function, StopVectors>, ul: ptr<function, StopVectors>, ua: ptr<function, StopVectors>,
    inverse: ptr<function, StopBlocks>, affine: bool, values: ptr<function, StopScalars>) -> bool {
    let count=articulation_drive_locks.count;
    if count==0u { return true; }
    for (var refinement=0u;refinement<4u;refinement++) {
        var rhs: DriveLockVector;
        var satisfied=true;
        for (var r=0u;r<count;r++) {
            let expected=select(0.0,articulation_drive_locks.bounds[r],affine);
            let value=mixed_dot(articulation_drive_locks.rows[r],values);
            rhs[r]=value-expected;
            satisfied=satisfied && xyz_chart_finite(value) && abs(rhs[r])<=1e-6*(1.0+abs(value)+abs(expected));
        }
        if refinement>0u && satisfied { return true; }
        let multipliers=solve_drive_locks(rhs);
        var force: StopScalars;
        for (var r=0u;r<count;r++) { mixed_accumulate(articulation_drive_locks.rows[r],multipliers[r],&force); }
        let response=stop_mass_response(n,parents,counts,sl,sa,ul,ua,inverse,&force);
        for (var k=0u;k<n;k++) {
            for (var d=0u;d<(*counts)[k];d++) {
                (*values)[k][d]-=response[k][d];
                if !xyz_chart_finite((*values)[k][d]) { return false; }
            }
        }
    }
    for (var r=0u;r<count;r++) {
        let expected=select(0.0,articulation_drive_locks.bounds[r],affine);
        let value=mixed_dot(articulation_drive_locks.rows[r],values);
        if !xyz_chart_finite(value) || abs(value-expected)>1e-6*(1.0+abs(value)+abs(expected)) { return false; }
    }
    return true;
}

fn prepare_root_drive_locks(begin: u32, n: u32, counts: ptr<function, StopCounts>, inverse: ptr<function, StopBlocks>) -> bool {
    root_drive_locks.enabled=false;
    if n!=1u || joint_parent[begin]>=0 || joint_enabled[begin]==0u
        || (joint_type[begin]!=JOINT_BALL && joint_type[begin]!=JOINT_D6) { return true; }
    let lock_count=mixed_angular_lock_count(begin);
    if lock_count==0u || lock_count==3u { return true; }
    let width=(*counts)[0];
    let dof=joint_dof_offset[begin];
    var requested=false;
    for (var d=0u;d<width;d++) {
        let mode=joint_target_mode[dof+d];
        requested=requested || (mode>=TARGET_POS && mode<=TARGET_PV
            && (uniforms.stable_motor_feedback!=0u || joint_motor_model[dof+d]==1u))
            || joint_brake[dof+d]>0.0;
    }
    if !requested { return true; }
    let q=mixed_joint_rotation(begin);
    var stops: MixedStopRows;
    var lo: MixedStopBounds;
    var hi: MixedStopBounds;
    var rows=0u;
    if !mixed_angular_rows(0u,begin,q,dof,uniforms.dt,&stops,&lo,&hi,&rows) { return false; }
    var gradients: array<array<f32,6>,6>;
    var targets: array<f32,6>;
    var locks=0u;
    for (var row=0u;row<rows;row++) {
        if lo[row]!=hi[row] { continue; }
        if locks==6u { return false; }
        gradients[locks]=stops[row].gradient;
        targets[locks]=lo[row];
        locks++;
    }
    if locks==0u { return true; }
    var mobility=(*inverse)[0];
    if root_response_paired {
        for (var i=0u;i<36u;i++) { mobility[i]+=root_inverse_low[i]; }
    }
    var responses: array<array<f32,6>,6>;
    var gram: array<f32,36>;
    for (var a=0u;a<locks;a++) {
        for (var r=0u;r<width;r++) {
            for (var c=0u;c<width;c++) { responses[a][r]=fma(mobility[r*6u+c],gradients[a][c],responses[a][r]); }
        }
    }
    for (var a=0u;a<locks;a++) {
        for (var b=0u;b<locks;b++) {
            for (var d=0u;d<width;d++) { gram[a*6u+b]=fma(gradients[a][d],responses[b][d],gram[a*6u+b]); }
        }
    }
    let factor=joint_psd_inverse(gram,locks);
    if !factor.valid { return false; }
    for (var r=0u;r<width;r++) {
        root_drive_locks.desired_rate[r]=0.0;
        for (var c=0u;c<width;c++) {
            var normal=0.0;
            var correction=0.0;
            for (var a=0u;a<locks;a++) {
                for (var b=0u;b<locks;b++) {
                    let coefficient=responses[a][r]*factor.matrix[a*6u+b];
                    normal=fma(coefficient,gradients[b][c],normal);
                    correction=fma(coefficient,responses[b][c],correction);
                }
            }
            root_drive_locks.projection[r*6u+c]=select(0.0,1.0,r==c)-normal;
            root_drive_locks.inverse[r*6u+c]=mobility[r*6u+c]-correction;
        }
        for (var a=0u;a<locks;a++) {
            for (var b=0u;b<locks;b++) { root_drive_locks.desired_rate[r]=fma(responses[a][r]*factor.matrix[a*6u+b],targets[b],root_drive_locks.desired_rate[r]); }
        }
    }
    root_drive_locks.enabled=true;
    return true;
}

fn drive_mass_response(n: u32, parents: ptr<function, StopParents>, counts: ptr<function, StopCounts>,
    sl: ptr<function, StopVectors>, sa: ptr<function, StopVectors>, ul: ptr<function, StopVectors>, ua: ptr<function, StopVectors>,
    inverse: ptr<function, StopBlocks>, force: ptr<function, StopScalars>) -> StopScalars {
    if !root_drive_locks.enabled {
        var result=stop_mass_response(n,parents,counts,sl,sa,ul,ua,inverse,force);
        if !project_drive_locks(n,parents,counts,sl,sa,ul,ua,inverse,false,&result) {
            for (var k=0u;k<n;k++) { for (var d=0u;d<(*counts)[k];d++) { result[k][d]=bitcast<f32>(0x7fc00000u); } }
        }
        return result;
    }
    var result: StopScalars;
    for (var r=0u;r<(*counts)[0];r++) {
        for (var c=0u;c<(*counts)[0];c++) { result[0][r]=fma(root_drive_locks.inverse[r*6u+c],(*force)[0][c],result[0][r]); }
    }
    return result;
}

fn drive_predicted_rate(begin: u32, k: u32, d: u32, counts: ptr<function, StopCounts>, acceleration: ptr<function, StopScalars>) -> f32 {
    let first=joint_dof_offset[begin+k];
    if !root_drive_locks.enabled { return joint_qd[first+d]+uniforms.dt*(*acceleration)[k][d]; }
    var value=root_drive_locks.desired_rate[d];
    for (var c=0u;c<(*counts)[0];c++) { value=fma(root_drive_locks.projection[d*6u+c],joint_qd[first+c]+uniforms.dt*(*acceleration)[0][c],value); }
    return value;
}

fn solve_velocity_box(
    n: u32, matrix: ptr<function, VelocityBoxMatrix>,
    rhs: ptr<function, VelocityBoxVector>, limits: ptr<function, VelocityBoxVector>,
    torque: ptr<function, VelocityBoxVector>,
) -> bool {
    var fixed: array<bool, MAX_FEEDBACK_ROWS>;
    var scales: VelocityBoxVector;
    for (var i=0u;i<n;i++) {
        let diagonal=(*matrix)[i][i];
        if !(diagonal>0.0 && diagonal<=3.402823e38) { return false; }
        scales[i]=sqrt(diagonal);
        (*torque)[i]=0.0;
        fixed[i]=(*limits)[i]==0.0;
    }
    var indices: array<u32, MAX_FEEDBACK_ROWS>;
    var lower: VelocityBoxMatrix;
    var candidate: VelocityBoxVector;
    for (var iteration=0u;iteration<4u*n+32u;iteration++) {
        var count=0u;
        for (var i=0u;i<n;i++) {
            if !fixed[i] { indices[count]=i; count++; }
        }
        for (var r=0u;r<count;r++) {
            let i=indices[r];
            var value=(*rhs)[i];
            for (var j=0u;j<n;j++) {
                if fixed[j] { value-=(*matrix)[i][j]*(*torque)[j]; }
            }
            value/=scales[i];
            for (var c=0u;c<=r;c++) {
                let j=indices[c];
                var entry=(*matrix)[i][j]/scales[i]/scales[j];
                for (var k=0u;k<c;k++) { entry-=lower[r][k]*lower[c][k]; }
                if c==r {
                    if !(entry>0.0 && entry<=3.402823e38) { return false; }
                    lower[r][c]=sqrt(entry);
                } else { lower[r][c]=entry/lower[c][c]; }
            }
            for (var c=0u;c<r;c++) { value-=lower[r][c]*candidate[c]; }
            candidate[r]=value/lower[r][r];
        }
        for (var reverse=count;reverse>0u;reverse--) {
            let r=reverse-1u;
            var value=candidate[r];
            for (var c=r+1u;c<count;c++) { value-=lower[c][r]*candidate[c]; }
            candidate[r]=value/lower[r][r];
        }
        var alpha=1.0;
        var hit=n;
        for (var r=0u;r<count;r++) {
            let i=indices[r];
            candidate[r]/=scales[i];
            if !(abs(candidate[r])<=3.402823e38) { return false; }
            let limit=(*limits)[i];
            var fraction=1.0;
            if candidate[r]<-limit { fraction=(-limit-(*torque)[i])/(candidate[r]-(*torque)[i]); }
            else if candidate[r]>limit { fraction=(limit-(*torque)[i])/(candidate[r]-(*torque)[i]); }
            else { continue; }
            if fraction<=alpha { alpha=max(fraction,0.0);hit=i; }
        }
        for (var r=0u;r<count;r++) {
            let i=indices[r];
            (*torque)[i]=clamp((*torque)[i]+alpha*(candidate[r]-(*torque)[i]),-(*limits)[i],(*limits)[i]);
        }
        if hit<n { fixed[hit]=true;continue; }
        var release=n;
        for (var i=0u;i<n;i++) {
            if !fixed[i] || (*limits)[i]==0.0 { continue; }
            var gradient=-(*rhs)[i];
            for (var j=0u;j<n;j++) { gradient+=(*matrix)[i][j]*(*torque)[j]; }
            let tolerance=1e-6*(1.0+abs((*rhs)[i]));
            if ((*torque)[i]==-(*limits)[i] && gradient < -tolerance)
                || ((*torque)[i]==(*limits)[i] && gradient > tolerance) { release=i;break; }
        }
        if release<n { fixed[release]=false; } else { return true; }
    }
    return false;
}

fn apply_joint_compliance(
    begin: u32, n: u32,
    counts: ptr<function, StopCounts>,
    sl: ptr<function, StopVectors>, sa: ptr<function, StopVectors>,
    ul: ptr<function, StopVectors>, ua: ptr<function, StopVectors>,
    inverse: ptr<function, StopBlocks>,
    acceleration: ptr<function, StopScalars>,
) -> bool {
    var parents: StopParents;
    for (var k = 0u; k < n; k++) {
        parents[k] = -1;
        for (var p = 0u; p < k; p++) {
            if joint_child[begin+p] == joint_parent[begin+k] { parents[k] = i32(p); }
        }
    }
    var efforts: StopScalars;
    var unit: StopScalars;
    var changed = false;
    for (var k = 0u; k < n; k++) {
        let j = begin+k;
        for (var d = 0u; d < (*counts)[k]; d++) {
            let dof = joint_dof_offset[j]+d;
            let c = joint_compliance[dof];
            if c.enabled == 0u { continue; }
            if c.enabled != 1u || joint_type[j] != JOINT_D6 || (*counts)[k] != 6u
                || !(c.lower <= c.upper) || !(c.lower <= 3.402823e38) || !(c.upper >= -3.402823e38)
                || !(c.natural_frequency >= 0.0 && c.natural_frequency <= 3.402823e38)
                || !(c.damping_ratio >= 0.0 && c.damping_ratio <= 3.402823e38) { return false; }
            if c.natural_frequency == 0.0 { continue; }
            let position = motor_coordinate(j,d);
            let rate = joint_qd[dof];
            if c.lower != c.upper && position >= c.lower && position <= c.upper { continue; }
            unit[k][d] = 1.0;
            let response = stop_mass_response(n, &parents, counts, sl, sa, ul, ua, inverse, &unit);
            unit[k][d] = 0.0;
            let mobility = response[k][d];
            if !(mobility >= 0.0 && mobility <= 3.402823e38) { return false; }
            if mobility == 0.0 { continue; }
            let omega = 6.283185307179586*c.natural_frequency;
            let stiffness = omega*omega;
            let damping = 2.0*c.damping_ratio*omega;
            let denominator = 1.0+damping*uniforms.dt+stiffness*uniforms.dt*uniforms.dt;
            let equilibrium = clamp(position,c.lower,c.upper);
            var effort = ((-stiffness*(position-equilibrium)-(damping+stiffness*uniforms.dt)*rate)
                /denominator)/mobility;
            if c.lower != c.upper {
                effort = select(min(effort,0.0),max(effort,0.0),position<c.lower);
            }
            if !(abs(position) <= 3.402823e38 && abs(rate) <= 3.402823e38
                && stiffness <= 3.402823e38 && damping <= 3.402823e38
                && denominator <= 3.402823e38 && abs(effort) <= 3.402823e38) { return false; }
            efforts[k][d] = effort;
            changed = changed || effort != 0.0;
        }
    }
    if !changed { return true; }
    let delta = stop_mass_response(n, &parents, counts, sl, sa, ul, ua, inverse, &efforts);
    for (var k = 0u; k < n; k++) {
        for (var d = 0u; d < (*counts)[k]; d++) {
            if !(abs((*acceleration)[k][d]+delta[k][d]) <= 3.402823e38) { return false; }
        }
    }
    for (var k = 0u; k < n; k++) {
        for (var d = 0u; d < (*counts)[k]; d++) { (*acceleration)[k][d] += delta[k][d]; }
    }
    return true;
}

fn react_velocity_feedback(
    begin: u32, n: u32,
    counts: ptr<function, StopCounts>,
    sl: ptr<function, StopVectors>, sa: ptr<function, StopVectors>,
    ul: ptr<function, StopVectors>, ua: ptr<function, StopVectors>,
    inverse: ptr<function, StopBlocks>,
    acceleration: ptr<function, StopScalars>,
) -> bool {
    var parents: StopParents;
    var joints: array<u32, MAX_FEEDBACK_ROWS>;
    var axes: array<u32, MAX_FEEDBACK_ROWS>;
    var rows = 0u;
    for (var k = 0u; k < n; k += 1u) {
        parents[k] = -1;
        let j = begin + k;
        for (var p = 0u; p < k; p += 1u) {
            if joint_child[begin+p] == joint_parent[j] { parents[k] = i32(p); }
        }
        for (var d = 0u; d < (*counts)[k]; d += 1u) {
            let dof = joint_dof_offset[j] + d;
            if joint_target_mode[dof] != TARGET_VEL || joint_target_kd[dof] <= 0.0
                || joint_motor_model[dof] == 1u
                || joint_brake[dof] > 0.0
                || abs(joint_limit_upper[dof]-joint_limit_lower[dof]) < 1e-12 { continue; }
            if !(joint_target_kd[dof] > 0.0 && joint_target_kd[dof] <= 3.402823e38) { return false; }
            if rows == MAX_FEEDBACK_ROWS { return false; }
            joints[rows] = k;
            axes[rows] = d;
            rows += 1u;
        }
    }
    if rows == 0u { return true; }
    var predicted: StopScalars;
    for (var k=0u;k<n;k++) {
        let first=joint_dof_offset[begin+k];
        for (var d=0u;d<(*counts)[k];d++) { predicted[k][d]=joint_qd[first+d]+uniforms.dt*(*acceleration)[k][d]; }
    }
    if !project_drive_locks(n,&parents,counts,sl,sa,ul,ua,inverse,true,&predicted) { return false; }
    var gram: array<array<f32, MAX_FEEDBACK_ROWS>, MAX_FEEDBACK_ROWS>;
    var unit: StopScalars;
    for (var c = 0u; c < rows; c += 1u) {
        unit[joints[c]][axes[c]] = 1.0;
        let response = drive_mass_response(n, &parents, counts, sl, sa, ul, ua, inverse, &unit);
        unit[joints[c]][axes[c]] = 0.0;
        for (var r = 0u; r < rows; r += 1u) { gram[r][c] = response[joints[r]][axes[r]]; }
    }
    for (var r = 0u; r < rows; r += 1u) {
        for (var c = 0u; c < r; c += 1u) {
            let value = 0.5 * (gram[r][c] + gram[c][r]);
            gram[r][c] = value;
            gram[c][r] = value;
        }
    }
    var torque: VelocityBoxVector;
    var rhs: VelocityBoxVector;
    var limits: VelocityBoxVector;
    var matrix: VelocityBoxMatrix;
    for (var r = 0u; r < rows; r += 1u) {
        let dof = joint_dof_offset[begin+joints[r]] + axes[r];
        let kd = joint_target_kd[dof];
        limits[r] = joint_motor_max_force[dof];
        var rate=predicted[joints[r]][axes[r]];
        if root_drive_locks.enabled { rate=drive_predicted_rate(begin,joints[r],axes[r],counts,acceleration); }
        rhs[r] = joint_target_vel[dof] - rate;
        for (var c = 0u; c < rows; c += 1u) { matrix[r][c] = uniforms.dt * gram[r][c]; }
        matrix[r][r] += 1.0 / kd;
    }
    if !solve_velocity_box(rows,&matrix,&rhs,&limits,&torque) { return false; }
    var correction: StopScalars;
    for (var r = 0u; r < rows; r += 1u) {
        if !(abs(torque[r]) <= 3.402823e38) { return false; }
        correction[joints[r]][axes[r]] = torque[r];
    }
    let delta = stop_mass_response(n, &parents, counts, sl, sa, ul, ua, inverse, &correction);
    for (var k = 0u; k < n; k += 1u) {
        for (var d = 0u; d < (*counts)[k]; d += 1u) {
            let value = (*acceleration)[k][d] + delta[k][d];
            if !(abs(value) <= 3.402823e38) { return false; }
            (*acceleration)[k][d] = value;
        }
    }
    return true;
}

fn apply_motor_feedback(
    begin: u32, n: u32,
    counts: ptr<function, StopCounts>,
    sl: ptr<function, StopVectors>, sa: ptr<function, StopVectors>,
    ul: ptr<function, StopVectors>, ua: ptr<function, StopVectors>,
    inverse: ptr<function, StopBlocks>,
    acceleration: ptr<function, StopScalars>,
) -> bool {
    var parents: StopParents;
    for (var k = 0u; k < n; k += 1u) {
        parents[k] = -1;
        for (var p = 0u; p < k; p += 1u) {
            if joint_child[begin+p] == joint_parent[begin+k] { parents[k] = i32(p); }
        }
    }
    var correction: StopScalars;
    var unit: StopScalars;
    var changed = false;
    for (var k = 0u; k < n; k += 1u) {
        let j = begin + k;
        for (var d = 0u; d < (*counts)[k]; d += 1u) {
            let dof = joint_dof_offset[j] + d;
            let mode = joint_target_mode[dof];
            let pd = mode >= TARGET_POS && mode <= TARGET_PV;
            let acceleration_model = joint_motor_model[dof] == 1u;
            let deferred = pd && (uniforms.stable_motor_feedback != 0u || acceleration_model);
            let damping = joint_brake[dof];
            if mode==TARGET_VEL && !acceleration_model && uniforms.stable_motor_feedback!=0u && damping==0.0 {
                continue;
            }
            if !deferred && damping == 0.0 { continue; }
            if !(damping >= 0.0 && damping <= 3.402823e38) { return false; }
            let kind = joint_type[j];
            if abs(joint_limit_upper[dof] - joint_limit_lower[dof]) < 1e-12 { continue; }
            let kp = joint_target_kp[dof];
            let kd = joint_target_kd[dof];
            if !(kp >= 0.0 && kp <= 3.402823e38 && kd >= 0.0 && kd <= 3.402823e38) { return false; }
            unit[k][d] = 1.0;
            let response = drive_mass_response(n, &parents, counts, sl, sa, ul, ua, inverse, &unit);
            unit[k][d] = 0.0;
            let inverse_inertia = response[k][d];
            if !(inverse_inertia >= 0.0 && inverse_inertia <= 3.402823e38) { return false; }
            if inverse_inertia == 0.0 { continue; }
            var q = 0.0;
            if mode == TARGET_POS || mode == TARGET_PV || kind == JOINT_CABLE { q = motor_coordinate(j, d); }
            let qd = joint_qd[dof];
            let drive = drive_torque(mode, q, qd, kp, kd, joint_target_pos[dof], joint_target_vel[dof]);
            var scale = 1.0;
            if acceleration_model && pd { scale = 1.0/inverse_inertia; }
            let stiffness = select(kp, 0.0, mode == TARGET_VEL);
            var stabilized = drive*scale;
            let brake_load = damping * uniforms.dt * inverse_inertia;
            if !(brake_load >= 0.0 && brake_load <= 3.402823e38) { return false; }
            if uniforms.stable_motor_feedback != 0u && pd {
                let denominator = 1.0 + (kd + stiffness * uniforms.dt) * scale * uniforms.dt * inverse_inertia + brake_load;
                if !(denominator >= 1.0 && denominator <= 3.402823e38) { return false; }
                let total = ((drive - stiffness * uniforms.dt * qd)*scale - damping * qd)/denominator;
                stabilized = total + damping * (qd + uniforms.dt * inverse_inertia * total);
            }
            let cap = joint_motor_max_force[dof];
            if !(cap >= 0.0) { return false; }
            let motor = clamp(stabilized, -cap, cap);
            let brake = -damping * (qd + uniforms.dt * inverse_inertia * motor) / (1.0 + brake_load);
            var cable = 0.0;
            if kind == JOINT_CABLE {
                let spring = select(joint_cable_k_lin[j], joint_cable_k_bend[j], d != 0u);
                let damper = select(joint_cable_c_lin[j], joint_cable_c_bend[j], d != 0u);
                cable = -spring * q - damper * qd;
            }
            let before = joint_force[dof] + select(clamp(drive, -cap, cap), 0.0, deferred) - joint_friction[dof] * qd
                + cable + joint_muscle_torque[dof];
            let after = joint_force[dof] + motor + brake - joint_friction[dof] * qd
                + cable + joint_muscle_torque[dof];
            let delta = clamp_or_passthrough(after, joint_effort_limit[dof])
                - clamp_or_passthrough(before, joint_effort_limit[dof]);
            if !(abs(drive) <= 3.402823e38 && abs(stabilized) <= 3.402823e38
                && abs(before) <= 3.402823e38 && abs(after) <= 3.402823e38 && abs(delta) <= 3.402823e38) { return false; }
            correction[k][d] = delta;
            changed = changed || delta != 0.0;
        }
    }
    if !changed { return true; }
    let delta = stop_mass_response(n, &parents, counts, sl, sa, ul, ua, inverse, &correction);
    for (var k = 0u; k < n; k += 1u) {
        for (var d = 0u; d < (*counts)[k]; d += 1u) {
            if !(abs((*acceleration)[k][d] + delta[k][d]) <= 3.402823e38) { return false; }
        }
    }
    for (var k = 0u; k < n; k += 1u) {
        for (var d = 0u; d < (*counts)[k]; d += 1u) {
            (*acceleration)[k][d] += delta[k][d];
        }
    }
    return true;
}
