struct FrozenRow { la: vec4<f32>, lb: vec4<f32>, aa: vec4<f32>, ab: vec4<f32>, info: vec4<f32> }
struct Six { linear: vec4<f32>, angular: vec4<f32> }
struct Product { value: Six, energy: f32 }
// Per-fiber PCG state, block-Jacobi factor, tree factor D_c, tree coupling W_c, forward sweep
// value, warm starts for the first and later Newton iterations, and the island's step residual.
struct Krylov { x: Six, x_low: Six, r: Six, z: Six, p: Six, ap: Six, rhs: Six, chol: array<f32,36>, factor: array<f32,36>, coupling: array<f32,36>, y: Six, warm: array<Six,2>, reference: vec4<f32> }
@group(0) @binding(13) var<storage, read_write> frozen: array<FrozenRow>;
@group(0) @binding(14) var<storage, read_write> krylov: array<Krylov>;
var<workgroup> reduction: array<f32,128>;
var<workgroup> group_fault: u32;
var<workgroup> small_p: array<Six,128>;
var<workgroup> small_x: array<Six,128>;
var<workgroup> small_x_low: array<Six,128>;
var<workgroup> island_range: vec2<u32>;

fn six_zero() -> Six { return Six(vec4<f32>(0.0),vec4<f32>(0.0)); }
fn grabbed(i: u32) -> bool { return params.grab_target.w >= 0.0 && f32(i) == params.grab_target.w; }
fn six_dot(a: Six,b: Six) -> f32 { return dot(a.linear.xyz,b.linear.xyz)+dot(a.angular.xyz,b.angular.xyz); }
fn six_add(a: Six,b: Six,scale: f32) -> Six { return Six(a.linear+scale*b.linear,a.angular+scale*b.angular); }
fn six_component(v: Six,k: u32) -> f32 { if k < 3u { return v.linear[k]; } return v.angular[k-3u]; }
fn wide_add(a: vec2<f32>,b: vec2<f32>) -> vec2<f32> {
    let sum = a.x+b.x; let part = sum-a.x;
    let error = (a.x-(sum-part))+(b.x-part)+(a.y+b.y);
    let high = sum+error; return vec2<f32>(high,error-(high-sum));
}
fn wide_product(a: f32,b: f32) -> vec2<f32> {
    let high = a*b; return vec2<f32>(high,fma(a,b,-high));
}
fn wide_scale(a: vec2<f32>,b: f32) -> vec2<f32> {
    return wide_add(wide_product(a.x,b),vec2<f32>(a.y*b,0.0));
}
fn wide_row(r: FrozenRow,a: Six,b: Six,al: Six,bl: Six) -> vec2<f32> {
    var sum = vec2<f32>(0.0);
    for (var k = 0u; k < 6u; k++) {
        sum = wide_add(sum,wide_scale(vec2<f32>(six_component(a,k),six_component(al,k)),six_component(gradient(r,0u),k)));
        sum = wide_add(sum,wide_scale(vec2<f32>(six_component(b,k),six_component(bl,k)),six_component(gradient(r,1u),k)));
    }
    return sum;
}
fn gradient(r: FrozenRow,side: u32) -> Six { if side == 0u { return Six(r.la,r.aa); } return Six(r.lb,r.ab); }
fn row_product(r: FrozenRow,a: Six,b: Six) -> f32 {
    return dot(r.la.xyz,a.linear.xyz-b.linear.xyz)+dot(r.la.xyz+r.lb.xyz,b.linear.xyz)
        +dot(r.aa.xyz,a.angular.xyz-b.angular.xyz)+dot(r.aa.xyz+r.ab.xyz,b.angular.xyz);
}
fn mass_apply(i: u32,v: Six) -> Six {
    let f = fibers[i]; if f.com_mass.w == 0.0 { return v; }
    let q = motions[i].rotation;
    let angular = rot(q,mat3x3<f32>(f.inertia0.xyz,f.inertia1.xyz,f.inertia2.xyz)*rot(conj(q),v.angular.xyz));
    return Six(v.linear/f.com_mass.w,vec4<f32>(angular,0.0));
}
fn cg_vector(i: u32,solution: bool) -> Six {
    if params.numerics.w <= 128.0 { let j = colors[params.islands.x+i]; if solution { return small_x[j]; } return small_p[j]; }
    if solution { return krylov[i].x; } return krylov[i].p;
}
fn solution_low(i: u32) -> Six {
    if params.numerics.w <= 128.0 { return small_x_low[colors[params.islands.x+i]]; } return krylov[i].x_low;
}
fn cg_sync() {
    if params.numerics.w <= 128.0 { workgroupBarrier(); } else { storageBarrier(); }
}
fn system_apply(i: u32,solution: bool) -> Product {
    let f = fibers[i]; let v = cg_vector(i,solution);
    var values: array<vec2<f32>,6>;
    for (var k = 0u; k < 6u; k++) {
        var basis = six_zero(); if k < 3u { basis.linear[k] = 1.0; } else { basis.angular[k-3u] = 1.0; }
        let column = mass_apply(i,basis);
        for (var a = 0u; a < 6u; a++) {
            values[a] = wide_add(values[a],wide_product(six_component(column,a),six_component(v,k)));
        }
    }
    var energy = six_dot(v,mass_apply(i,v));
    for (var j = 0u; j < f.vertices.w; j++) {
        let edge = colors[f.vertices.z+j]; let b = bonds[edge/2u]; let side = edge%2u;
        let a = cg_vector(b.fibers.x,solution); let c = cg_vector(b.fibers.y,solution);
        for (var k = 0u; k < 6u; k++) {
            let r = frozen[(edge/2u)*6u+k];
            let wide = wide_row(r,a,c,six_zero(),six_zero());
            let projection = wide.x+wide.y; let scalar = wide_scale(wide,r.info.y);
            let g = gradient(r,side);
            for (var a = 0u; a < 6u; a++) { values[a] = wide_add(values[a],wide_scale(scalar,six_component(g,a))); }
            energy += 0.5*projection*(scalar.x+scalar.y);
        }
    }
    if grabbed(i) {
        for (var k = 0u; k < 3u; k++) {
            let r = frozen[params.counts.y*6u+k]; let g = gradient(r,0u);
            let projection = wide_row(r,v,six_zero(),six_zero(),six_zero());
            let scalar = wide_scale(projection,r.info.y);
            for (var a = 0u; a < 6u; a++) { values[a] = wide_add(values[a],wide_scale(scalar,six_component(g,a))); }
            energy += (projection.x+projection.y)*(scalar.x+scalar.y);
        }
    }
    return Product(Six(vec4<f32>(values[0].x+values[0].y,values[1].x+values[1].y,values[2].x+values[2].y,0.0),
        vec4<f32>(values[3].x+values[3].y,values[4].x+values[4].y,values[5].x+values[5].y,0.0)),energy);
}
fn system_residual(i: u32) -> Six {
    let f = fibers[i]; let x = cg_vector(i,true); let xl = solution_low(i);
    var values: array<vec2<f32>,6>;
    for (var k = 0u; k < 6u; k++) {
        var basis = six_zero(); if k < 3u { basis.linear[k] = 1.0; } else { basis.angular[k-3u] = 1.0; }
        let column = mass_apply(i,basis);
        for (var a = 0u; a < 6u; a++) {
            values[a] = wide_add(values[a],-wide_scale(vec2<f32>(six_component(x,k),six_component(xl,k)),six_component(column,a)));
        }
    }
    for (var j = 0u; j < f.vertices.w; j++) {
        let edge = colors[f.vertices.z+j]; let b = bonds[edge/2u]; let side = edge%2u;
        let a = cg_vector(b.fibers.x,true); let c = cg_vector(b.fibers.y,true);
        let al = solution_low(b.fibers.x); let cl = solution_low(b.fibers.y);
        for (var k = 0u; k < 6u; k++) {
            let r = frozen[(edge/2u)*6u+k];
            let constraint = wide_add(vec2<f32>(r.info.x,0.0),wide_row(r,a,c,al,cl));
            let scalar = -wide_add(wide_scale(constraint,r.info.y),vec2<f32>(r.info.z,0.0));
            let g = gradient(r,side);
            for (var axis = 0u; axis < 6u; axis++) {
                values[axis] = wide_add(values[axis],wide_scale(scalar,six_component(g,axis)));
            }
        }
    }
    if grabbed(i) {
        for (var k = 0u; k < 3u; k++) {
            let r = frozen[params.counts.y*6u+k]; let g = gradient(r,0u);
            let constraint = wide_add(vec2<f32>(r.info.x,0.0),wide_row(r,x,six_zero(),xl,six_zero()));
            let scalar = -wide_add(wide_scale(constraint,r.info.y),vec2<f32>(r.info.z,0.0));
            for (var a = 0u; a < 6u; a++) { values[a] = wide_add(values[a],wide_scale(scalar,six_component(g,a))); }
        }
    }
    return Six(vec4<f32>(values[0].x+values[0].y,values[1].x+values[1].y,values[2].x+values[2].y,0.0),
        vec4<f32>(values[3].x+values[3].y,values[4].x+values[4].y,values[5].x+values[5].y,0.0));
}
fn precondition(i: u32,v: Six) -> Six {
    var values: array<f32,6>;
    for (var a = 0u; a < 6u; a++) {
        var value = six_component(v,a);
        for (var b = 0u; b < a; b++) { value -= krylov[i].chol[a*6u+b]*values[b]; }
        values[a] = value/krylov[i].chol[a*6u+a];
    }
    for (var reverse = 0u; reverse < 6u; reverse++) {
        let a = 5u-reverse; var value = values[a];
        for (var b = a+1u; b < 6u; b++) { value -= krylov[i].chol[b*6u+a]*values[b]; }
        values[a] = value/krylov[i].chol[a*6u+a];
    }
    return Six(vec4<f32>(values[0],values[1],values[2],0.0),vec4<f32>(values[3],values[4],values[5],0.0));
}
fn initialize_system(i: u32) {
    var matrix: array<f32,36>; var rhs = six_zero(); let f = fibers[i];
    var wide_rhs: array<vec2<f32>,6>;
    for (var b = 0u; b < 6u; b++) {
        var basis = six_zero(); if b < 3u { basis.linear[b] = 1.0; } else { basis.angular[b-3u] = 1.0; }
        let column = mass_apply(i,basis);
        for (var a = 0u; a < 6u; a++) { matrix[a*6u+b] = six_component(column,a); }
    }
    for (var j = 0u; j < f.vertices.w; j++) {
        let edge = colors[f.vertices.z+j];
        for (var k = 0u; k < 6u; k++) {
            let r = frozen[(edge/2u)*6u+k]; let g = gradient(r,edge%2u);
            let load = -wide_add(wide_product(r.info.x,r.info.y),vec2<f32>(r.info.z,0.0));
            for (var a = 0u; a < 6u; a++) { wide_rhs[a] = wide_add(wide_rhs[a],wide_scale(load,six_component(g,a))); }
            for (var a = 0u; a < 6u; a++) {
                for (var b = 0u; b <= a; b++) { matrix[a*6u+b] += six_component(g,a)*six_component(g,b)*r.info.y; }
            }
        }
    }
    if grabbed(i) {
        for (var k = 0u; k < 3u; k++) {
            let r = frozen[params.counts.y*6u+k]; let g = gradient(r,0u);
            let load = -wide_add(wide_product(r.info.x,r.info.y),vec2<f32>(r.info.z,0.0));
            for (var a = 0u; a < 6u; a++) {
                wide_rhs[a] = wide_add(wide_rhs[a],wide_scale(load,six_component(g,a)));
                for (var b = 0u; b <= a; b++) { matrix[a*6u+b] += six_component(g,a)*six_component(g,b)*r.info.y; }
            }
        }
    }
    rhs = Six(vec4<f32>(wide_rhs[0].x+wide_rhs[0].y,wide_rhs[1].x+wide_rhs[1].y,wide_rhs[2].x+wide_rhs[2].y,0.0),
        vec4<f32>(wide_rhs[3].x+wide_rhs[3].y,wide_rhs[4].x+wide_rhs[4].y,wide_rhs[5].x+wide_rhs[5].y,0.0));
    var scales: array<f32,6>;
    for (var a = 0u; a < 6u; a++) {
        scales[a] = sqrt(matrix[a*6u+a]);
        if scales[a] <= 0.0 || !finite4(vec4<f32>(scales[a])) { atomicOr(&status[0],4u); scales[a] = 1.0; }
    }
    for (var a = 0u; a < 6u; a++) {
        for (var b = 0u; b <= a; b++) { matrix[a*6u+b] /= scales[a]*scales[b]; }
        matrix[a*6u+a] += 1e-5;
    }
    for (var a = 0u; a < 6u; a++) {
        for (var b = 0u; b <= a; b++) {
            var value = matrix[a*6u+b];
            for (var k = 0u; k < b; k++) { value -= matrix[a*6u+k]*matrix[b*6u+k]; }
            if a == b {
                if value <= 0.0 || !finite4(vec4<f32>(value)) { atomicOr(&status[0],4u); value = 1.0; }
                matrix[a*6u+b] = sqrt(value);
            } else { matrix[a*6u+b] = value/matrix[b*6u+b]; }
        }
    }
    for (var a = 0u; a < 6u; a++) {
        for (var b = 0u; b <= a; b++) { matrix[a*6u+b] *= scales[a]; }
    }
    krylov[i].chol = matrix; krylov[i].rhs = rhs; krylov[i].r = rhs; krylov[i].x = six_zero(); krylov[i].x_low = six_zero();
    let z = precondition(i,rhs); krylov[i].z = z; krylov[i].p = z;
    if params.numerics.w <= 128.0 { let j = colors[params.islands.x+i]; small_p[j] = z; small_x[j] = six_zero(); small_x_low[j] = six_zero(); }
}
fn reduce_sum(lane: u32,value: f32) -> f32 {
    reduction[lane] = value; workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride /= 2u) {
        if lane < stride { reduction[lane] += reduction[lane+stride]; }
        workgroupBarrier();
    }
    return workgroupUniformLoad(&reduction[0]);
}
fn uniform_fault(lane: u32) -> u32 {
    if lane == 0u { group_fault = atomicLoad(&status[0]); }
    return workgroupUniformLoad(&group_fault);
}
@compute @workgroup_size(64)
fn freeze_rows(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x; if atomicLoad(&status[0]) != 0u { return; }
    if i == 0u { atomicStore(&status[1],0u); atomicStore(&status[2],0u); atomicStore(&status[3],0u); }
    if i == 0u && params.grab_target.w >= 0.0 {
        let body = u32(params.grab_target.w); let m = motions[body]; let grab = grab_parameters(body);
        let r = rot(m.rotation,params.grab_anchor.xyz-fibers[body].com_mass.xyz);
        let error = m.center.xyz+r-grab.xyz;
        let lambda = vec3<f32>(predicted[body].center.w,predicted[body].velocity.w,predicted[body].omega.w);
        for (var k = 0u; k < 3u; k++) {
            var f: FrozenRow; var axis = vec3<f32>(0.0); axis[k] = 1.0;
            f.la = vec4<f32>(axis,0.0); f.aa = vec4<f32>(cross(r,axis),0.0);
            f.info = vec4<f32>(error[k],params.step.x*params.step.x/grab.w,lambda[k],1.0);
            frozen[params.counts.y*6u+k] = f;
            if !finite4(f.info) || f.info.y <= 0.0 { atomicOr(&status[0],4u); }
        }
    }
    if i >= params.counts.y { return; }
    let b = bonds[i]; let s = bond_states[i];
    for (var k = 0u; k < 6u; k++) {
        var f: FrozenRow;
        if s.damage.x < 1.0 {
            let r = row(b,motions[b.fibers.x],motions[b.fibers.y],s.damage.x,k);
            let inv_alpha = (params.step.x*params.step.x)/r.compliance;
            var lambda = 0.0; if k < 3u { lambda = s.lambda0[k]; } else { lambda = s.lambda1[k-3u]; }
            f.info = vec4<f32>(r.value,inv_alpha,lambda,1.0);
            if fibers[b.fibers.x].com_mass.w > 0.0 { f.la = vec4<f32>(r.la,0.0); f.aa = vec4<f32>(r.aa,0.0); }
            if fibers[b.fibers.y].com_mass.w > 0.0 { f.lb = vec4<f32>(r.lb,0.0); f.ab = vec4<f32>(r.ab,0.0); }
            if !finite4(f.info) || inv_alpha <= 0.0 { atomicOr(&status[0],4u); }
        }
        frozen[i*6u+k] = f;
    }
}
const NO_PARENT: u32 = 0xffffffffu;
struct Factor { l: array<f32,36>, ok: bool }
var<workgroup> tree_span: vec2<u32>;
var<workgroup> tree_fault: atomic<u32>;
var<workgroup> tree_ready: u32;
var<workgroup> warm_slot: u32;
var<workgroup> warm_better: u32;
var<workgroup> step_reference: f32;

fn tree_parent(i: u32) -> u32 { return colors[params.tree.x+i]; }
fn other_fiber(edge: u32) -> u32 { let b = bonds[edge/2u]; return select(b.fibers.x,b.fibers.y,edge%2u == 0u); }
fn six_set(v: ptr<function,Six>,k: u32,value: f32) { if k < 3u { (*v).linear[k] = value; } else { (*v).angular[k-3u] = value; } }
// True when edge `j` is the first of fiber `i`'s edges reaching `neighbor`.
fn first_edge(i: u32,j: u32,neighbor: u32) -> bool {
    let f = fibers[i];
    for (var k = 0u; k < j; k++) { if other_fiber(colors[f.vertices.z+k]) == neighbor { return false; } }
    return true;
}
// Full symmetric diagonal block A_ii: mass, bond rows and grab rows touching fiber `i`.
fn diagonal_block(i: u32) -> array<f32,36> {
    var m: array<f32,36>; let f = fibers[i];
    for (var b = 0u; b < 6u; b++) {
        var basis = six_zero(); six_set(&basis,b,1.0);
        let column = mass_apply(i,basis);
        for (var a = 0u; a < 6u; a++) { m[a*6u+b] = six_component(column,a); }
    }
    for (var j = 0u; j < f.vertices.w; j++) {
        let edge = colors[f.vertices.z+j];
        for (var k = 0u; k < 6u; k++) {
            let r = frozen[(edge/2u)*6u+k]; let g = gradient(r,edge%2u);
            for (var a = 0u; a < 6u; a++) {
                for (var b = 0u; b < 6u; b++) { m[a*6u+b] += six_component(g,a)*six_component(g,b)*r.info.y; }
            }
        }
    }
    if grabbed(i) {
        for (var k = 0u; k < 3u; k++) {
            let r = frozen[params.counts.y*6u+k]; let g = gradient(r,0u);
            for (var a = 0u; a < 6u; a++) {
                for (var b = 0u; b < 6u; b++) { m[a*6u+b] += six_component(g,a)*six_component(g,b)*r.info.y; }
            }
        }
    }
    return m;
}
// Off-diagonal block A_ab (rows: dofs of `a`, columns: dofs of `b`) from every bond joining them.
fn coupling_block(a: u32,b: u32) -> array<f32,36> {
    var m: array<f32,36>; let f = fibers[a];
    for (var j = 0u; j < f.vertices.w; j++) {
        let edge = colors[f.vertices.z+j]; if other_fiber(edge) != b { continue; }
        let side = edge%2u;
        for (var k = 0u; k < 6u; k++) {
            let r = frozen[(edge/2u)*6u+k]; let ga = gradient(r,side); let gb = gradient(r,1u-side);
            for (var x = 0u; x < 6u; x++) {
                for (var y = 0u; y < 6u; y++) { m[x*6u+y] += six_component(ga,x)*six_component(gb,y)*r.info.y; }
            }
        }
    }
    return m;
}
fn product6(a: array<f32,36>,b: array<f32,36>) -> array<f32,36> {
    var m: array<f32,36>;
    for (var x = 0u; x < 6u; x++) {
        for (var y = 0u; y < 6u; y++) {
            var value = 0.0; for (var k = 0u; k < 6u; k++) { value += a[x*6u+k]*b[k*6u+y]; }
            m[x*6u+y] = value;
        }
    }
    return m;
}
fn apply6(m: array<f32,36>,v: Six,transpose: bool) -> Six {
    var out = six_zero();
    for (var x = 0u; x < 6u; x++) {
        var value = 0.0;
        for (var k = 0u; k < 6u; k++) { value += select(m[x*6u+k],m[k*6u+x],transpose)*six_component(v,k); }
        six_set(&out,x,value);
    }
    return out;
}
// Diagonally scaled, regularized Cholesky factor, as for the block-Jacobi preconditioner.
fn cholesky6(input: array<f32,36>) -> Factor {
    var m = input; var ok = true; var scales: array<f32,6>;
    for (var a = 0u; a < 6u; a++) {
        scales[a] = sqrt(m[a*6u+a]);
        if scales[a] <= 0.0 || !finite4(vec4<f32>(scales[a])) { ok = false; scales[a] = 1.0; }
    }
    for (var a = 0u; a < 6u; a++) {
        for (var b = 0u; b <= a; b++) { m[a*6u+b] /= scales[a]*scales[b]; }
        m[a*6u+a] += 1e-5;
    }
    for (var a = 0u; a < 6u; a++) {
        for (var b = 0u; b <= a; b++) {
            var value = m[a*6u+b];
            for (var k = 0u; k < b; k++) { value -= m[a*6u+k]*m[b*6u+k]; }
            if a == b {
                if value <= 0.0 || !finite4(vec4<f32>(value)) { ok = false; value = 1.0; }
                m[a*6u+b] = sqrt(value);
            } else { m[a*6u+b] = value/m[b*6u+b]; }
        }
    }
    for (var a = 0u; a < 6u; a++) {
        for (var b = 0u; b <= a; b++) { m[a*6u+b] *= scales[a]; }
    }
    return Factor(m,ok);
}
fn chol_solve(l: array<f32,36>,v: Six) -> Six {
    var values: array<f32,6>;
    for (var a = 0u; a < 6u; a++) {
        var value = six_component(v,a);
        for (var b = 0u; b < a; b++) { value -= l[a*6u+b]*values[b]; }
        values[a] = value/l[a*6u+a];
    }
    for (var reverse = 0u; reverse < 6u; reverse++) {
        let a = 5u-reverse; var value = values[a];
        for (var b = a+1u; b < 6u; b++) { value -= l[b*6u+a]*values[b]; }
        values[a] = value/l[a*6u+a];
    }
    return Six(vec4<f32>(values[0],values[1],values[2],0.0),vec4<f32>(values[3],values[4],values[5],0.0));
}
fn tree_sync() { workgroupBarrier(); storageBarrier(); }
fn level_bounds(span: vec2<u32>,level: u32) -> vec2<u32> {
    return vec2<u32>(colors[params.tree.z+span.x+level],colors[params.tree.z+span.x+level+1u]);
}
// Block LDLᵀ along the spanning forest, deepest level first: D_c = A_cc − Σ A_ck W_k, W_c = D_c⁻¹ A_cp.
fn tree_factor(lane: u32,span: vec2<u32>) {
    for (var level = 0u; level < span.y; level++) {
        let bounds = level_bounds(span,level);
        for (var slot = bounds.x+lane; slot < bounds.y; slot += 128u) {
            let c = colors[params.tree.w+slot]; let f = fibers[c];
            var d = diagonal_block(c);
            for (var j = 0u; j < f.vertices.w; j++) {
                let k = other_fiber(colors[f.vertices.z+j]);
                if k == c || tree_parent(k) != c || !first_edge(c,j,k) { continue; }
                let schur = product6(coupling_block(c,k),krylov[k].coupling);
                for (var e = 0u; e < 36u; e++) { d[e] -= schur[e]; }
            }
            let factor = cholesky6(d);
            if !factor.ok { atomicStore(&tree_fault,1u); }
            krylov[c].factor = factor.l;
            let p = tree_parent(c);
            if p != NO_PARENT {
                let a = coupling_block(c,p); var w: array<f32,36>;
                for (var column = 0u; column < 6u; column++) {
                    var v = six_zero();
                    for (var row = 0u; row < 6u; row++) { six_set(&v,row,a[row*6u+column]); }
                    let s = chol_solve(factor.l,v);
                    for (var row = 0u; row < 6u; row++) { w[row*6u+column] = six_component(s,row); }
                }
                krylov[c].coupling = w;
            }
        }
        tree_sync();
    }
}
// z = A⁻¹ r on the spanning forest: forward elimination towards the roots, then back-substitution.
fn tree_solve(lane: u32,span: vec2<u32>) {
    for (var level = 0u; level < span.y; level++) {
        let bounds = level_bounds(span,level);
        for (var slot = bounds.x+lane; slot < bounds.y; slot += 128u) {
            let c = colors[params.tree.w+slot]; let f = fibers[c];
            var y = krylov[c].r;
            for (var j = 0u; j < f.vertices.w; j++) {
                let k = other_fiber(colors[f.vertices.z+j]);
                if k == c || tree_parent(k) != c || !first_edge(c,j,k) { continue; }
                y = six_add(y,apply6(krylov[k].coupling,krylov[k].y,true),-1.0);
            }
            krylov[c].y = y;
        }
        tree_sync();
    }
    for (var back = 0u; back < span.y; back++) {
        let bounds = level_bounds(span,span.y-1u-back);
        for (var slot = bounds.x+lane; slot < bounds.y; slot += 128u) {
            let c = colors[params.tree.w+slot];
            var z = chol_solve(krylov[c].factor,krylov[c].y);
            let p = tree_parent(c);
            if p != NO_PARENT { z = six_add(z,apply6(krylov[c].coupling,krylov[p].z,false),-1.0); }
            krylov[c].z = z;
        }
        tree_sync();
    }
}
// Preconditioned residual for every fiber of the island: the tree solve or block Jacobi.
fn precondition_island(lane: u32,count: u32,range: vec2<u32>,span: vec2<u32>,tree: bool) {
    if tree { tree_sync(); tree_solve(lane,span); return; }
    for (var slot = lane; slot < count; slot += 128u) {
        let i = colors[params.islands.z+range.x+slot];
        krylov[i].z = precondition(i,krylov[i].r);
    }
    tree_sync();
}
fn island_dot(lane: u32,count: u32,range: vec2<u32>) -> f32 {
    var local = 0.0;
    for (var slot = lane; slot < count; slot += 128u) {
        let i = colors[params.islands.z+range.x+slot]; local += six_dot(krylov[i].r,krylov[i].z);
    }
    return reduce_sum(lane,local);
}
fn set_solution(i: u32,slot: u32,x: Six) {
    krylov[i].x = x; krylov[i].x_low = six_zero();
    if params.numerics.w <= 128.0 { small_x[slot] = x; small_x_low[slot] = six_zero(); }
}
fn set_direction(i: u32,slot: u32) {
    krylov[i].p = krylov[i].z;
    if params.numerics.w <= 128.0 { small_p[slot] = krylov[i].z; }
}
@compute @workgroup_size(128)
fn solve_coupled(@builtin(local_invocation_id) id: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
    let lane = id.x; if uniform_fault(lane) != 0u { return; }
    if lane == 0u {
        island_range = vec2<u32>(colors[params.islands.y+group.x],colors[params.islands.y+group.x+1u]);
        tree_span = vec2<u32>(colors[params.tree.y+2u*group.x],colors[params.tree.y+2u*group.x+1u]);
        warm_slot = min(atomicLoad(&status[4]),1u);
        atomicStore(&tree_fault,0u);
    }
    let range = workgroupUniformLoad(&island_range); let count = range.y-range.x;
    let span = workgroupUniformLoad(&tree_span); let newton = workgroupUniformLoad(&warm_slot);
    for (var slot = lane; slot < count; slot += 128u) { initialize_system(colors[params.islands.z+range.x+slot]); }
    tree_sync();
    tree_factor(lane,span);
    if lane == 0u { tree_ready = select(0u,1u,atomicLoad(&tree_fault) == 0u); }
    let tree = workgroupUniformLoad(&tree_ready) == 1u;
    if tree { precondition_island(lane,count,range,span,true); }
    let initial = island_dot(lane,count,range); var rho = initial;
    if lane == 0u {
        let first = colors[params.islands.z+range.x];
        if newton == 0u { krylov[first].reference.x = initial; }
        step_reference = max(initial,krylov[first].reference.x);
    }
    let reference = workgroupUniformLoad(&step_reference);
    let tolerance_scale = select(0.5,1.0,params.islands.w == 1u);
    let absolute = tolerance_scale*params.numerics.x*params.numerics.x/f32(params.islands.w);
    let tolerance = max(tolerance_scale*reference*params.coupled.y*params.coupled.y,absolute);
    if lane == 0u { atomicMax(&status[3],bitcast<u32>(initial)); }
    if !finite4(vec4<f32>(initial)) || initial < 0.0 { if lane == 0u { atomicOr(&status[0],4u); } return; }
    if uniform_fault(lane) != 0u || initial <= absolute { return; }
    for (var slot = lane; slot < count; slot += 128u) {
        let i = colors[params.islands.z+range.x+slot]; set_solution(i,slot,krylov[i].warm[newton]);
    }
    tree_sync();
    for (var slot = lane; slot < count; slot += 128u) {
        let i = colors[params.islands.z+range.x+slot]; krylov[i].r = system_residual(i);
    }
    precondition_island(lane,count,range,span,tree);
    let warm = island_dot(lane,count,range);
    if lane == 0u { warm_better = select(0u,1u,finite4(vec4<f32>(warm)) && warm >= 0.0 && warm < initial); }
    if workgroupUniformLoad(&warm_better) == 1u {
        rho = warm;
    } else {
        for (var slot = lane; slot < count; slot += 128u) {
            let i = colors[params.islands.z+range.x+slot];
            set_solution(i,slot,six_zero()); krylov[i].r = krylov[i].rhs;
        }
        precondition_island(lane,count,range,span,tree);
    }
    for (var slot = lane; slot < count; slot += 128u) { set_direction(colors[params.islands.z+range.x+slot],slot); }
    cg_sync(); tree_sync();
    var converged = rho <= tolerance; var used = 0u; var relative = select(1.0,sqrt(rho/reference),converged);
    for (var iteration = 0u; !converged && iteration < u32(params.coupled.x); iteration++) {
        used = iteration+1u; var local = 0.0;
        for (var slot = lane; slot < count; slot += 128u) {
            let i = colors[params.islands.z+range.x+slot];
            let ap = system_apply(i,false); krylov[i].ap = ap.value; local += ap.energy;
        }
        let denominator = reduce_sum(lane,local);
        if denominator <= 0.0 || !finite4(vec4<f32>(denominator,rho,0.0,0.0)) { break; }
        let alpha = rho/denominator;
        for (var slot = lane; slot < count; slot += 128u) {
            let i = colors[params.islands.z+range.x+slot];
            var high = krylov[i].x; var low = krylov[i].x_low;
            for (var k = 0u; k < 6u; k++) {
                let value = wide_add(vec2<f32>(six_component(high,k),six_component(low,k)),wide_product(six_component(krylov[i].p,k),alpha));
                if k < 3u { high.linear[k] = value.x; low.linear[k] = value.y; }
                else { high.angular[k-3u] = value.x; low.angular[k-3u] = value.y; }
            }
            krylov[i].x = high; krylov[i].x_low = low;
            if params.numerics.w <= 128.0 { small_x[slot] = high; small_x_low[slot] = low; }
            krylov[i].r = six_add(krylov[i].r,krylov[i].ap,-alpha);
        }
        precondition_island(lane,count,range,span,tree);
        var next = island_dot(lane,count,range); var restarted = false;
        if next <= tolerance || iteration+1u == u32(params.coupled.x) {
            cg_sync();
            for (var slot = lane; slot < count; slot += 128u) {
                let i = colors[params.islands.z+range.x+slot]; krylov[i].r = system_residual(i);
            }
            precondition_island(lane,count,range,span,tree);
            next = island_dot(lane,count,range); restarted = true; relative = sqrt(next/reference);
            if next <= tolerance && next >= 0.0 { converged = true; break; }
        }
        if !finite4(vec4<f32>(next)) || next < 0.0 { break; }
        let beta = select(next/rho,0.0,restarted);
        for (var slot = lane; slot < count; slot += 128u) {
            let i = colors[params.islands.z+range.x+slot];
            krylov[i].p = six_add(krylov[i].z,krylov[i].p,beta);
            if params.numerics.w <= 128.0 { small_p[slot] = krylov[i].p; }
        }
        cg_sync(); tree_sync(); rho = next;
    }
    for (var slot = lane; slot < count; slot += 128u) {
        let i = colors[params.islands.z+range.x+slot]; krylov[i].warm[newton] = krylov[i].x;
    }
    if lane == 0u {
        atomicMax(&status[1],used); atomicMax(&status[2],bitcast<u32>(relative));
        if !converged { atomicOr(&status[0],4u); }
    }
}
@compute @workgroup_size(64)
fn apply_coupled(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x == 0u { atomicAdd(&status[4],1u); }
    let i = id.x; if atomicLoad(&status[0]) != 0u { return; }
    if i < params.counts.x && fibers[i].com_mass.w > 0.0 {
        var m = motions[i]; let x = krylov[i].x;
        m.center += x.linear; m.rotation = normalize(mul(exponential(x.angular.xyz),m.rotation)); motions[i] = m;
        if grabbed(i) {
            var lambda = vec3<f32>(0.0);
            for (var k = 0u; k < 3u; k++) {
                let r = frozen[params.counts.y*6u+k];
                let constraint = wide_add(vec2<f32>(r.info.x,0.0),wide_row(r,x,six_zero(),krylov[i].x_low,six_zero()));
                let wide_lambda = -wide_scale(constraint,r.info.y); lambda[k] = wide_lambda.x+wide_lambda.y;
            }
            predicted[i].center.w = lambda.x; predicted[i].velocity.w = lambda.y; predicted[i].omega.w = lambda.z;
        }
    }
    if i < params.counts.y && bond_states[i].damage.x < 1.0 {
        let b = bonds[i]; var s = bond_states[i]; let a = krylov[b.fibers.x].x; let c = krylov[b.fibers.y].x;
        for (var k = 0u; k < 6u; k++) {
            let r = frozen[i*6u+k];
            let constraint = wide_add(vec2<f32>(r.info.x,0.0),wide_row(r,a,c,krylov[b.fibers.x].x_low,krylov[b.fibers.y].x_low));
            let wide_lambda = -wide_scale(constraint,r.info.y); let lambda = wide_lambda.x+wide_lambda.y;
            if k < 3u { s.lambda0[k] = lambda; } else { s.lambda1[k-3u] = lambda; }
        }
        bond_states[i] = s;
    }
}
