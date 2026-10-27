struct Params { islands: u32, blocks: u32, iterations: u32, tolerance: f32, scratch_stride: u32, pad0: u32, pad1: u32, pad2: u32 }
struct Block { contact: vec4<f32>, metadata: vec4<u32> }
struct Solution { impulse: vec4<f32>, velocity: vec4<f32>, diagnostics: vec4<u32> }
@group(0) @binding(0) var<storage, read> gram: array<f32>;
@group(0) @binding(1) var<storage, read> free: array<f32>;
@group(0) @binding(2) var<storage, read> blocks: array<Block>;
@group(0) @binding(3) var<storage, read> counts: array<u32>;
@group(0) @binding(4) var<storage, read> island_count: array<u32>;
@group(0) @binding(5) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(6) var<storage, read_write> scratch: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> solutions: array<Solution>;
@group(0) @binding(8) var<storage, read_write> diagnostics: array<vec4<u32>>;
@group(0) @binding(9) var<uniform> u: Params;
const INPUT: u32 = 33554432u;
const NONFINITE: u32 = 67108864u;
const PSD: u32 = 134217728u;
const CONVERGENCE: u32 = 268435456u;
const MAX: f32 = 3.4028234e38;

fn finite(v: f32) -> bool { return v == v && abs(v) <= MAX; }
fn finite3(v: vec3<f32>) -> bool { return finite(v.x) && finite(v.y) && finite(v.z); }
fn norm2(v: vec2<f32>) -> f32 {
    let scale = max(abs(v.x), abs(v.y));
    if scale == 0.0 { return 0.0; }
    return scale * length(v / scale);
}
fn sread(i: u32) -> f32 { return bitcast<f32>(atomicLoad(&scratch[i])); }
fn swrite(i: u32, v: f32) { atomicStore(&scratch[i], bitcast<u32>(v)); }
fn rounded(i: u32, v: f32) -> f32 { swrite(i, v); return sread(i); }
fn fault(island: u32, code: u32) {
    diagnostics[island].x = code;
    atomicOr(&status, code);
}
fn normal_roundoff(impulse: f32, velocity: f32, diagonal: f32) -> f32 {
    if impulse == 0.0 { return 0.0; }
    let bits = bitcast<u32>(impulse);
    let increase = velocity < 0.0;
    let next = bitcast<f32>(select(bits - 1u, bits + 1u, (impulse > 0.0) == increase));
    if !finite(next) { return 0.0; }
    return 0.5*abs(next-impulse)*diagonal;
}
fn physical_product(island: u32, n: u32, row: u32, x: u32, initial: f32) -> f32 {
    let stride = 3u * u.blocks;
    let temp = (island + 1u) * u.scratch_stride - 1u;
    var sum = initial; var low = 0.0;
    for (var j = 0u; j < n; j += 1u) {
        let a = gram[island*stride*stride + row*stride + j]; let b = sread(x+j);
        let v = rounded(temp, a*b); let total = rounded(temp, sum+v);
        let virtual_v = rounded(temp, total-sum); let virtual_sum = rounded(temp, total-virtual_v);
        low += (sum-virtual_sum) + (v-virtual_v) + fma(a, b, -v);
        sum = total;
    }
    return sum + low;
}
fn product(island: u32, n: u32, row: u32, x: u32, initial: f32) -> f32 {
    let physical = physical_product(island,n,row,x,initial);
    let block = blocks[island*u.blocks+row/3u];
    if row%3u == 0u && block.metadata.x == 3u {
        return fma(block.contact.z,sread(x+row),physical);
    }
    return physical;
}
fn disk_linear(a: vec3<f32>, v: vec2<f32>, shift: f32) -> vec2<f32> {
    let scale = max(a.x, a.z);
    if scale == 0.0 {
        if shift > 0.0 { return -v/shift; }
        if all(v == vec2<f32>(0.0)) { return vec2<f32>(0.0); }
        return vec2<f32>(MAX);
    }
    let normalized = a/scale;
    let high = 0.5*(normalized.x+normalized.z) + norm2(vec2<f32>(0.5*(normalized.x-normalized.z),normalized.y));
    let low = max(fma(normalized.x,normalized.z,-normalized.y*normalized.y)/high,0.0);
    var first = vec2<f32>(high-normalized.z,normalized.y);
    if normalized.x < normalized.z { first = vec2<f32>(normalized.y,high-normalized.x); }
    if all(first == vec2<f32>(0.0)) { first = vec2<f32>(1.0,0.0); }
    first /= norm2(first);
    let second = vec2<f32>(-first.y,first.x);
    let projected = vec2<f32>(dot(first,v),dot(second,v));
    let denominator = vec2<f32>(high,low) + vec2<f32>(shift/scale);
    var value = vec2<f32>(0.0);
    for (var i = 0u; i < 2u; i += 1u) {
        if denominator[i] == 0.0 {
            if projected[i] != 0.0 { return vec2<f32>(MAX); }
        } else { value[i] = -projected[i]/denominator[i]/scale; }
    }
    return first*value.x + second*value.y;
}
fn disk(a: vec3<f32>, v: vec2<f32>, radius: f32) -> vec2<f32> {
    if radius == 0.0 { return vec2<f32>(0.0); }
    let trial = disk_linear(a, v, 0.0);
    if all(abs(trial) < vec2<f32>(MAX)) && norm2(trial) <= radius { return trial; }
    var low = 0.0; var high = norm2(v)/radius;
    if high == 0.0 { return vec2<f32>(0.0); }
    for (var i = 0u; i < 32u; i += 1u) {
        let middle = 0.5*low + 0.5*high;
        if middle == low || middle == high { break; }
        if norm2(disk_linear(a, v, middle)) > radius { low = middle; } else { high = middle; }
    }
    return disk_linear(a, v, high);
}
fn evaluate(normal: f32, a: vec3<f32>, tangent: vec3<f32>, velocity: vec3<f32>, mu: f32) -> vec4<f32> {
    let t = disk(tangent, velocity.yz + a.yz*normal, mu*normal);
    return vec4<f32>(normal, t, velocity.x + dot(a, vec3<f32>(normal, t)));
}
fn contact(island: u32, a: vec3<f32>, tangent: vec3<f32>, velocity: vec3<f32>, mu: f32) -> vec3<f32> {
    if velocity.x >= 0.0 { return vec3<f32>(0.0); }
    let scale = max(a.x, max(tangent.x, tangent.z));
    if scale == 0.0 || a.x == 0.0 { fault(island, CONVERGENCE); return vec3<f32>(0.0); }
    if mu == 0.0 || all(a.yz == vec2<f32>(0.0)) {
        let normal = -velocity.x/a.x;
        return vec3<f32>(normal, disk(tangent, velocity.yz, mu*normal));
    }
    let speed = max(1.0, max(abs(velocity.x), max(abs(velocity.y), abs(velocity.z))));
    let aa = a/scale; let tt = tangent/scale; let vv = velocity/speed;
    var low = 0.0; var high = -vv.x/aa.x;
    var best = evaluate(high, aa, tt, vv, mu);
    var bracketed = best.w >= 0.0;
    for (var i = 0u; i < 64u && !bracketed; i += 1u) {
        if abs(best.w)*speed <= u.tolerance*0.1 { return best.xyz*(speed/scale); }
        low = high; high *= 2.0;
        best = evaluate(high, aa, tt, vv, mu);
        if !finite3(best.xyz) || !finite(best.w) { fault(island, NONFINITE); return vec3<f32>(0.0); }
        bracketed = best.w >= 0.0;
    }
    if !bracketed { fault(island, CONVERGENCE); return vec3<f32>(0.0); }
    for (var i = 0u; i < 64u; i += 1u) {
        if abs(best.w)*speed <= u.tolerance*0.1 { break; }
        let middle = 0.5*low + 0.5*high;
        if middle == low || middle == high { break; }
        let value = evaluate(middle, aa, tt, vv, mu);
        if abs(value.w) < abs(best.w) { best = value; }
        if value.w > 0.0 { high = middle; } else { low = middle; }
    }
    return best.xyz*(speed/scale);
}

@compute @workgroup_size(64)
fn solve_impacts(@builtin(global_invocation_id) id: vec3<u32>) {
    let island = id.x;
    if island >= u.islands { return; }
    var empty: Solution;
    for (var b = 0u; b < u.blocks; b += 1u) { solutions[island*u.blocks+b] = empty; }
    if atomicLoad(&status) != 0u { return; }
    diagnostics[island] = vec4<u32>(0u);
    if island_count[0] > u.islands { fault(island, INPUT); return; }
    if island >= island_count[0] { return; }
    let count = counts[island];
    if count > u.blocks { fault(island, INPUT); return; }
    let stride = 3u*u.blocks; let n = 3u*count;
    let matrix = island*stride*stride;
    let factor = island*u.scratch_stride;
    let x = factor + stride*stride; let shifted = x+stride; let gradient = shifted+stride; let modes = gradient+stride;
    for (var b = 0u; b < count; b += 1u) {
        let block = blocks[island*u.blocks+b]; let kind = block.metadata.x; let c = block.contact;
        let r = 3u*b;
        if kind > 3u || block.metadata.y > 1u || any(block.metadata.zw != vec2<u32>(0u)) { fault(island, INPUT); return; }
        let v = vec3<f32>(free[island*stride+r], free[island*stride+r+1u], free[island*stride+r+2u]);
        if !finite3(v) { fault(island, NONFINITE); return; }
        if kind == 3u {
            if (bitcast<u32>(c.x)&0x7fffffffu) > 0x7f800000u || (bitcast<u32>(c.y)&0x7fffffffu) > 0x7f800000u
                || c.x > c.y || c.x > MAX || c.y < -MAX
                || !finite(c.z) || c.z < 0.0 || c.w != 0.0 { fault(island, INPUT); return; }
        } else if !finite3(c.xyz) || !finite(c.w) { fault(island, NONFINITE); return; }
        if kind == 0u {
            if c.y < 0.0 || c.x < c.y || c.z < 0.0 || c.z > 1.0 || c.w != v.x || v.x > 0.0 { fault(island, INPUT); return; }
            swrite(shifted+r, v.x+c.z*v.x); swrite(shifted+r+1u, 2.0*v.y); swrite(shifted+r+2u, 2.0*v.z);
        } else {
            if (kind != 3u && any(c != vec4<f32>(0.0))) || block.metadata.y != 0u || any(v.yz != vec2<f32>(0.0)) { fault(island, INPUT); return; }
            swrite(shifted+r, v.x); swrite(shifted+r+1u, 0.0); swrite(shifted+r+2u, 0.0);
        }
        swrite(modes+b, f32(block.metadata.y));
    }
    for (var i = 0u; i < n; i += 1u) {
        if !finite(sread(shifted+i)) { fault(island, NONFINITE); return; }
        let di = gram[matrix+i*stride+i];
        if !finite(di) { fault(island, NONFINITE); return; }
        if di < 0.0 { fault(island, PSD); return; }
        for (var j = 0u; j < n; j += 1u) {
            let value = gram[matrix+i*stride+j]; let transpose = gram[matrix+j*stride+i];
            let dj = gram[matrix+j*stride+j];
            if !finite(value) || !finite(transpose) || !finite(dj) { fault(island, NONFINITE); return; }
            let scale = sqrt(di)*sqrt(max(dj, 0.0));
            if !finite(scale) || abs(value-transpose) > 2e-6*scale || abs(value) > scale*(1.0+2e-6) { fault(island, PSD); return; }
            if (i%3u != 0u && blocks[island*u.blocks+i/3u].metadata.x != 0u)
                || (j%3u != 0u && blocks[island*u.blocks+j/3u].metadata.x != 0u) {
                if value != 0.0 { fault(island, INPUT); return; }
            }
        }
        for (var j = 0u; j <= i; j += 1u) {
            let scale = sqrt(di)*sqrt(gram[matrix+j*stride+j]);
            var value = 0.0;
            if scale > 0.0 { value = gram[matrix+i*stride+j]/scale; }
            for (var k = 0u; k < j; k += 1u) { value -= sread(factor+i*stride+k)*sread(factor+j*stride+k); }
            if !finite(value) || (i == j && value < -2e-5) { fault(island, PSD); return; }
            if i == j { swrite(factor+i*stride+j, sqrt(max(value, 1e-7))); }
            else { swrite(factor+i*stride+j, value/sread(factor+j*stride+j)); }
        }
    }
    var iterations = 0u;
    for (var transition = 0u; transition <= count; transition += 1u) {
        for (var i = 0u; i < n; i += 1u) { swrite(x+i, 0.0); }
        var converged = false; var residual = MAX;
        for (var sweep = 0u; sweep < u.iterations; sweep += 1u) {
            iterations += 1u;
            for (var b = 0u; b < count; b += 1u) {
                let block = blocks[island*u.blocks+b]; let r = 3u*b;
                let a = vec3<f32>(gram[matrix+r*stride+r], gram[matrix+r*stride+r+1u], gram[matrix+r*stride+r+2u]);
                if block.metadata.x == 0u {
                    let t = vec3<f32>(gram[matrix+(r+1u)*stride+r+1u], gram[matrix+(r+1u)*stride+r+2u], gram[matrix+(r+2u)*stride+r+2u]);
                    swrite(x+r, 0.0); swrite(x+r+1u, 0.0); swrite(x+r+2u, 0.0);
                    let v = vec3<f32>(product(island,n,r,x,sread(shifted+r)), product(island,n,r+1u,x,sread(shifted+r+1u)), product(island,n,r+2u,x,sread(shifted+r+2u)));
                    let mu = select(block.contact.x, block.contact.y, sread(modes+b) != 0.0);
                    let value = contact(island,a,t,v,mu);
                    if !finite3(value) { fault(island, NONFINITE); return; }
                    swrite(x+r,value.x); swrite(x+r+1u,value.y); swrite(x+r+2u,value.z);
                } else {
                    let velocity = product(island,n,r,x,sread(shifted+r));
                    var value = sread(x+r);
                    var diagonal = a.x;
                    if block.metadata.x == 3u { diagonal += block.contact.z; }
                    if diagonal > 0.0 { value -= velocity/diagonal; }
                    else if block.metadata.x == 3u {
                        value = 0.0;
                        if velocity > 0.0 { value = block.contact.x; }
                        else if velocity < 0.0 { value = block.contact.y; }
                    }
                    if block.metadata.x == 2u { value = max(value,0.0); }
                    if block.metadata.x == 3u { value = clamp(value,block.contact.x,block.contact.y); }
                    if !finite(value) { fault(island, NONFINITE); return; }
                    swrite(x+r,value);
                }
            }
            if atomicLoad(&status) != 0u { return; }
            residual = 0.0;
            var excess = 0.0;
            for (var b = 0u; b < count; b += 1u) {
                let block = blocks[island*u.blocks+b]; let r = 3u*b;
                let impulse = vec3<f32>(sread(x+r),sread(x+r+1u),sread(x+r+2u));
                let v = vec3<f32>(product(island,n,r,x,sread(shifted+r)), product(island,n,r+1u,x,sread(shifted+r+1u)), product(island,n,r+2u,x,sread(shifted+r+2u)));
                if !finite3(v) { fault(island, NONFINITE); return; }
                swrite(gradient+r,v.x); swrite(gradient+r+1u,v.y); swrite(gradient+r+2u,v.z);
                var error = abs(v.x);
                if block.metadata.x == 3u {
                    if block.contact.x == block.contact.y { error = 0.0; }
                    else if impulse.x == block.contact.x { error = max(-v.x,0.0); }
                    else if impulse.x == block.contact.y { error = max(v.x,0.0); }
                } else if block.metadata.x != 1u && impulse.x == 0.0 { error = max(-v.x,0.0); }
                residual = max(residual,error);
                excess = max(excess,max(error-normal_roundoff(impulse.x,v.x,gram[matrix+r*stride+r]),0.0));
                if block.metadata.x != 0u { continue; }
                let mu = select(block.contact.x,block.contact.y,sread(modes+b) != 0.0);
                let radius = mu*impulse.x; let length_t = norm2(impulse.yz);
                if !finite(radius) || !finite(length_t) { fault(island, NONFINITE); return; }
                if impulse.x < 0.0 || length_t > radius*(1.0+2e-6) { residual = MAX; excess = MAX; }
                if radius == 0.0 { continue; }
                if length_t == 0.0 { let error_t = 0.5*norm2(v.yz); residual = max(residual,error_t); excess = max(excess,error_t); continue; }
                let direction = impulse.yz/length_t;
                let multiplier = max(-dot(v.yz,direction),0.0);
                let gap = max(radius-length_t,0.0)/radius;
                var stationarity = norm2(v.yz);
                if multiplier > 0.0 {
                    let temp = (island+1u)*u.scratch_stride-1u;
                    let product_t = rounded(temp,v.z*impulse.y);
                    let determinant = fma(v.y,impulse.z,-product_t)-fma(v.z,impulse.y,-product_t);
                    stationarity = abs(determinant)/length_t;
                }
                if !finite(stationarity) { fault(island, NONFINITE); return; }
                let error_t = 0.5*max(stationarity,multiplier*gap);
                residual = max(residual,error_t); excess = max(excess,error_t);
            }
            if excess <= u.tolerance { converged = true; break; }
        }
        diagnostics[island] = vec4<u32>(0u,iterations,bitcast<u32>(residual),0u);
        if !converged { fault(island, CONVERGENCE); return; }
        var promoted = false;
        for (var b = 0u; b < count; b += 1u) {
            if blocks[island*u.blocks+b].metadata.x == 0u && sread(modes+b) == 0.0
                && 0.5*norm2(vec2<f32>(sread(gradient+3u*b+1u),sread(gradient+3u*b+2u))) > 1e-4 {
                swrite(modes+b,1.0); promoted = true;
            }
        }
        if promoted { continue; }
        for (var b = 0u; b < count; b += 1u) {
            let r = 3u*b; let slot = island*u.blocks+b;
            let velocity = vec3<f32>(physical_product(island,n,r,x,free[island*stride+r]),physical_product(island,n,r+1u,x,free[island*stride+r+1u]),physical_product(island,n,r+2u,x,free[island*stride+r+2u]));
            if !finite3(velocity) { fault(island, NONFINITE); return; }
            let sticking = blocks[slot].metadata.x == 0u && 0.5*norm2(vec2<f32>(sread(gradient+r+1u),sread(gradient+r+2u))) <= u.tolerance;
            solutions[slot] = Solution(vec4<f32>(sread(x+r),sread(x+r+1u),sread(x+r+2u),0.0),vec4<f32>(velocity,0.0),vec4<u32>(u32(sread(modes+b)),u32(sticking),iterations,bitcast<u32>(residual)));
        }
        return;
    }
    fault(island,CONVERGENCE);
}
