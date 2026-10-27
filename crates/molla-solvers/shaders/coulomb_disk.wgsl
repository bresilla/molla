struct Params { islands: u32, contacts: u32, iterations: u32, tolerance: f32, scratch_stride: u32, kinds: u32, pad1: u32, pad2: u32 }
struct Solution { impulse: vec2<f32>, slip: vec2<f32>, cap: f32, regime: u32, iterations: u32, residual: f32 }
@group(0) @binding(0) var<storage, read> grams: array<f32>;
@group(0) @binding(1) var<storage, read> free_velocity: array<f32>;
@group(0) @binding(2) var<storage, read> contact_data: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> contact_counts: array<u32>;
@group(0) @binding(4) var<storage, read> island_count: array<u32>;
@group(0) @binding(5) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(6) var<storage, read_write> scratch: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> solution_words: array<u32>;
@group(0) @binding(8) var<uniform> u: Params;
@group(0) @binding(9) var<storage, read_write> diagnostics: array<vec4<u32>>;
@group(0) @binding(10) var<storage, read> block_kinds: array<u32>;

const INPUT: u32 = 65536u;
const NONFINITE: u32 = 131072u;
const PSD: u32 = 262144u;
const CONVERGENCE: u32 = 524288u;
const CAPTURE: f32 = 1e-4;
const ALLOCATION_TOLERANCE: f32 = 8.0 * 1.1920929e-7;

fn block_kind(slot: u32) -> u32 {
    if (u.kinds == 0u) { return 0u; }
    return block_kinds[slot];
}

fn sread(index: u32) -> f32 { return bitcast<f32>(atomicLoad(&scratch[index])); }
fn swrite(index: u32, value: f32) { atomicStore(&scratch[index], bitcast<u32>(value)); }
fn rounded(index: u32, value: f32) -> f32 { swrite(index, value); return sread(index); }

fn residue_base(island: u32) -> u32 {
    let stride=2u*u.contacts;
    return island*u.scratch_stride+2u*stride*stride+5u*stride;
}
fn store_solution(slot: u32, value: Solution, low: vec2<f32>) {
    let offset=8u*slot;
    solution_words[offset]=bitcast<u32>(value.impulse.x);
    solution_words[offset+1u]=bitcast<u32>(value.impulse.y);
    solution_words[offset+2u]=bitcast<u32>(value.slip.x);
    solution_words[offset+3u]=bitcast<u32>(value.slip.y);
    solution_words[offset+4u]=bitcast<u32>(value.cap);
    solution_words[offset+5u]=value.regime;
    solution_words[offset+6u]=value.iterations;
    solution_words[offset+7u]=bitcast<u32>(value.residual);
    let residue=8u*u.islands*u.contacts+2u*slot;
    solution_words[residue]=bitcast<u32>(low.x);
    solution_words[residue+1u]=bitcast<u32>(low.y);
}
fn pair_sum(slot: u32, a: f32, b: f32) -> vec2<f32> {
    let high=rounded(slot,a+b);
    let virtual_b=rounded(slot,high-a);
    let virtual_a=rounded(slot,high-virtual_b);
    let error_a=rounded(slot,a-virtual_a);
    let error_b=rounded(slot,b-virtual_b);
    return vec2<f32>(high,rounded(slot,error_a+error_b));
}
fn pair_add(slot: u32, a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let sum=pair_sum(slot,a.x,b.x);
    let low=rounded(slot,rounded(slot,sum.y+a.y)+b.y);
    return pair_sum(slot,sum.x,low);
}
fn pair_product(slot: u32, a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let high=rounded(slot,a.x*b.x);
    let ah=bitcast<f32>(bitcast<u32>(a.x)&0xfffff000u);
    let bh=bitcast<f32>(bitcast<u32>(b.x)&0xfffff000u);
    let al=rounded(slot,a.x-ah);let bl=rounded(slot,b.x-bh);
    var low=rounded(slot,rounded(slot,ah*bh)-high);
    low=rounded(slot,low+rounded(slot,ah*bl));
    low=rounded(slot,low+rounded(slot,al*bh));
    low=rounded(slot,low+rounded(slot,al*bl));
    low=rounded(slot,low+rounded(slot,a.x*b.y));
    low=rounded(slot,low+rounded(slot,a.y*b.x));
    low=rounded(slot,low+rounded(slot,a.y*b.y));
    return pair_sum(slot,high,low);
}
fn pair_divide(slot: u32, a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let high=rounded(slot,a.x/b.x);
    let error=pair_add(slot,a,-pair_product(slot,b,vec2<f32>(high,0.0)));
    return pair_sum(slot,high,(error.x+error.y)/b.x);
}
fn pair_root(slot: u32, value: vec2<f32>) -> vec2<f32> {
    let high=rounded(slot,sqrt(max(value.x,0.0)));
    if high==0.0 { return vec2<f32>(0.0); }
    let error=pair_add(slot,value,-pair_product(slot,vec2<f32>(high,0.0),vec2<f32>(high,0.0)));
    return pair_sum(slot,high,(error.x+error.y)/(2.0*high));
}
fn paired_direction(slot: u32, x: vec2<f32>, y: vec2<f32>, radius: f32) -> mat2x2<f32> {
    let scale=max(abs(x.x),abs(y.x));
    if scale==0.0 || radius==0.0 { return mat2x2<f32>(vec2<f32>(0.0),vec2<f32>(0.0)); }
    let exponent=frexp(scale).exp;
    let sx=ldexp(x,vec2<i32>(-exponent)); let sy=ldexp(y,vec2<i32>(-exponent));
    let norm=pair_root(slot,pair_add(slot,pair_product(slot,sx,sx),pair_product(slot,sy,sy)));
    let gain=pair_divide(slot,vec2<f32>(radius,0.0),norm);
    return mat2x2<f32>(pair_product(slot,sx,gain),pair_product(slot,sy,gain));
}
fn paired_disk_gap(slot: u32, value: vec2<f32>, low: vec2<f32>, radius: f32) -> f32 {
    let exponent=frexp(radius).exp;
    let r=ldexp(radius,-exponent);
    let x=ldexp(vec2<f32>(value.x,low.x),vec2<i32>(-exponent));
    let y=ldexp(vec2<f32>(value.y,low.y),vec2<i32>(-exponent));
    let rr=pair_product(slot,vec2<f32>(r,0.0),vec2<f32>(r,0.0));
    let squared=pair_add(slot,pair_product(slot,x,x),pair_product(slot,y,y));
    let gap=pair_divide(slot,pair_add(slot,rr,-squared),rr);
    return gap.x+gap.y;
}

fn fault(island: u32, code: u32) {
    diagnostics[island] = vec4<u32>(code, 0u, 0u, 0u);
    atomicOr(&status, code);
}

fn finite(x: f32) -> bool { return (bitcast<u32>(x) & 0x7fffffffu) < 0x7f800000u; }

fn solve_entry(island: u32, stride: u32, row: u32, col: u32) -> f32 {
    var value = grams[island*stride*stride+row*stride+col];
    if row == col && row%2u == 0u && block_kind(island*u.contacts+row/2u) == 3u {
        value += contact_data[island*u.contacts+row/2u].z;
    }
    return value;
}

fn scalar_project(slot: u32, value: f32, scale: f32) -> f32 {
    let kind = block_kind(slot);
    if kind == 2u { return max(value, 0.0); }
    if kind == 3u { return clamp(value, contact_data[slot].x/scale, contact_data[slot].y/scale); }
    return value;
}

fn reference_impulse(slot: u32) -> f32 {
    if block_kind(slot) == 3u { return scalar_project(slot, 0.0, 1.0); }
    return 0.0;
}

fn scalar_pinned(slot: u32, gradient: f32) -> bool {
    let kind = block_kind(slot);
    if kind == 2u { return gradient > u.tolerance; }
    if kind == 3u { return contact_data[slot].x == contact_data[slot].y || abs(gradient) > u.tolerance; }
    return false;
}

fn positive_semidefinite(island: u32, stride: u32, n: u32, scale: f32, factor: u32) -> bool {
    let base = island*stride*stride;
    for (var i = 0u; i < n; i += 1u) {
        for (var j = 0u; j <= i; j += 1u) {
            let a = grams[base+i*stride+j]/scale;
            let b = grams[base+j*stride+i]/scale;
            if abs(a-b) > 1e-6 {
                diagnostics[island] = vec4<u32>(PSD,0u,bitcast<u32>(a-b),0x80000000u+i*stride+j);
                atomicOr(&status,PSD); return false;
            }
            let value = 0.5*a+0.5*b;
            swrite(factor+i*stride+j,value);
            swrite(factor+j*stride+i,value);
        }
    }
    for (var k = 0u; k < n; k += 1u) {
        var pivot = k;
        for (var i = k+1u; i < n; i += 1u) {
            if sread(factor+i*stride+i) > sread(factor+pivot*stride+pivot) { pivot = i; }
        }
        let diagonal = sread(factor+pivot*stride+pivot);
        if diagonal <= 1e-7 {
            for (var i = k; i < n; i += 1u) {
                var bound = 0.0;
                for (var j = k; j < n; j += 1u) { bound += abs(sread(factor+i*stride+j)); }
                if !finite(bound) || bound > 1e-4 {
                    diagnostics[island] = vec4<u32>(PSD,0u,bitcast<u32>(bound),0x40000000u+i);
                    atomicOr(&status,PSD); return false;
                }
            }
            return true;
        }
        if pivot != k {
            for (var j = k; j < n; j += 1u) {
                let value = sread(factor+k*stride+j);
                swrite(factor+k*stride+j,sread(factor+pivot*stride+j));
                swrite(factor+pivot*stride+j,value);
            }
            for (var i = k; i < n; i += 1u) {
                let value = sread(factor+i*stride+k);
                swrite(factor+i*stride+k,sread(factor+i*stride+pivot));
                swrite(factor+i*stride+pivot,value);
            }
        }
        for (var i = k+1u; i < n; i += 1u) {
            let left = sread(factor+i*stride+k)/diagonal;
            for (var j = i; j < n; j += 1u) {
                let value = fma(-left,sread(factor+j*stride+k),sread(factor+i*stride+j));
                if !finite(value) {
                    diagnostics[island] = vec4<u32>(PSD,0u,bitcast<u32>(value),1u+i*stride+j);
                    atomicOr(&status,PSD); return false;
                }
                swrite(factor+i*stride+j,value);
                swrite(factor+j*stride+i,value);
            }
        }
    }
    return true;
}

fn unit2(v: vec2<f32>) -> vec2<f32> {
    let scale = max(abs(v.x), abs(v.y));
    if (scale == 0.0) { return vec2<f32>(0.0); }
    let scaled = v / scale;
    let squared = fma(scaled.x, scaled.x, scaled.y * scaled.y);
    var inverse = inverseSqrt(squared);
    inverse *= fma(-0.5 * squared, inverse * inverse, 1.5);
    return scaled * inverse;
}
fn relative_disk_gap(value: vec2<f32>, radius: f32, round_slot: u32) -> f32 {
    let split = frexp(radius);
    let scaled = ldexp(value, vec2<i32>(-split.exp));
    let rr = rounded(round_slot, split.fract * split.fract);
    let xx = rounded(round_slot, scaled.x * scaled.x);
    let yy = rounded(round_slot, scaled.y * scaled.y);
    let first = rounded(round_slot, rr - xx);
    let virtual_x = rounded(round_slot, rr - first);
    let virtual_r = rounded(round_slot, first + virtual_x);
    let second = rounded(round_slot, first - yy);
    let virtual_y = rounded(round_slot, first - second);
    let virtual_first = rounded(round_slot, second + virtual_y);
    let low = (rr - virtual_r) + (virtual_x - xx)
        + (first - virtual_first) + (virtual_y - yy)
        + fma(split.fract, split.fract, -rr) - fma(scaled.x, scaled.x, -xx) - fma(scaled.y, scaled.y, -yy);
    return (second + low) / rr;
}
fn norm2(v: vec2<f32>) -> f32 {
    let scale = max(abs(v.x), abs(v.y));
    if (scale == 0.0) { return 0.0; }
    return scale * length(v / scale);
}
fn project(v: vec2<f32>, radius: f32) -> vec2<f32> {
    if (radius == 0.0) { return vec2<f32>(0.0); }
    let scale = max(abs(v.x), abs(v.y));
    if (scale == 0.0) { return v; }
    let unit = v / scale;
    let length_unit = length(unit);
    if (scale <= radius / length_unit) { return v; }
    return unit2(v) * radius;
}

fn product(matrix: u32, stride: u32, n: u32, row: u32, vector: u32) -> f32 {
    var sum = 0.0;
    var error = 0.0;
    for (var j = 0u; j < n; j += 1u) {
        let value = sread(matrix + row * stride + j) * sread(vector + j) - error;
        let next = sum + value;
        error = (next - sum) - value;
        sum = next;
    }
    return sum;
}

fn slip_at(island: u32, n: u32, row: u32, physical: u32) -> vec2<f32> {
    let stride = 2u * u.contacts;
    let round_slot = island * u.scratch_stride + stride * stride;
    var sum = free_velocity[island * stride + row];
    var error = 0.0;
    for (var j = 0u; j < n; j += 1u) {
        let g = grams[island * stride * stride + row * stride + j];
        let impulse = sread(physical + j);
        let value = rounded(round_slot, g * impulse);
        let total = rounded(round_slot, sum + value);
        let virtual_value = rounded(round_slot, total - sum);
        let virtual_sum = rounded(round_slot, total - virtual_value);
        error += (sum - virtual_sum) + (value - virtual_value) + fma(g, impulse, -value);
        sum = total;
        if j%2u == 0u && block_kind(island*u.contacts+j/2u) != 0u {
            let low_value = rounded(round_slot,g*sread(physical+j+1u));
            let low_total = rounded(round_slot,sum+low_value);
            let virtual_low = rounded(round_slot,low_total-sum);
            let virtual_high = rounded(round_slot,low_total-virtual_low);
            error += (sum-virtual_high)+(low_value-virtual_low)
                + fma(g,sread(physical+j+1u),-low_value);
            sum = low_total;
        }
        if block_kind(island*u.contacts+j/2u)==0u {
            let low=sread(residue_base(island)+j);
            if low!=0.0 {
                let low_value=rounded(round_slot,g*low);
                let low_total=rounded(round_slot,sum+low_value);
                let virtual_low=rounded(round_slot,low_total-sum);
                let virtual_high=rounded(round_slot,low_total-virtual_low);
                error+=(sum-virtual_high)+(low_value-virtual_low)+fma(g,low,-low_value);
                sum=low_total;
            }
        }
    }
    let high = rounded(round_slot, sum + error);
    let virtual_error = rounded(round_slot, high - sum);
    let virtual_sum = rounded(round_slot, high - virtual_error);
    let low = (sum - virtual_sum) + (error - virtual_error);
    return vec2<f32>(high, low);
}

fn gradient_at(island: u32, n: u32, row: u32, physical: u32) -> vec2<f32> {
    let velocity = slip_at(island, n, row, physical);
    if row%2u != 0u || block_kind(island*u.contacts+row/2u) != 3u { return velocity; }
    let slot = island*u.contacts+row/2u;
    let round_slot = island*u.scratch_stride + 4u*u.contacts*u.contacts;
    let diagonal = contact_data[slot].z;
    let impulse = sread(physical+row);
    let feedback = rounded(round_slot, diagonal*impulse);
    let high = rounded(round_slot, velocity.x+feedback);
    let virtual_feedback = rounded(round_slot, high-velocity.x);
    let virtual_velocity = rounded(round_slot, high-virtual_feedback);
    let low = velocity.y + (velocity.x-virtual_velocity) + (feedback-virtual_feedback)
        + fma(diagonal,impulse,-feedback)+diagonal*sread(physical+row+1u);
    return vec2<f32>(high,low);
}

fn kkt(island: u32, count: u32, x: u32, caps: u32, gradient: u32, impulse_scale: f32) -> f32 {
    let physical = x + 4u * u.contacts;
    let boundaries = x - 4u * u.contacts * u.contacts + 1u;
    let round_slot = boundaries - 1u;
    for (var c = 0u; c < count; c += 1u) {
        let normalized = vec2<f32>(sread(x + 2u*c), sread(x + 2u*c + 1u));
        var value = normalized * impulse_scale;
        if (sread(boundaries + c) != 0.0) { value = unit2(normalized) * sread(caps + c); }
        if block_kind(island*u.contacts+c) == 3u && impulse_scale > 0.0 {
            let bounds = contact_data[island*u.contacts+c].xy;
            if normalized.x <= bounds.x/impulse_scale { value.x = bounds.x; }
            else if normalized.x >= bounds.y/impulse_scale { value.x = bounds.y; }
        }
        if block_kind(island*u.contacts+c) != 0u { value.x = scalar_project(island*u.contacts+c, value.x, 1.0); }
        swrite(physical + 2u*c, value.x); swrite(physical + 2u*c + 1u, value.y);
        swrite(residue_base(island)+2u*c,0.0); swrite(residue_base(island)+2u*c+1u,0.0);
    }
    return physical_kkt(island, count, physical, caps, gradient, round_slot);
}

fn physical_kkt(island: u32, count: u32, physical: u32, caps: u32, gradient: u32, round_slot: u32) -> f32 {
    var residual = 0.0;
    var merit = 0.0;
    for (var c = 0u; c < count; c += 1u) {
        let vx = gradient_at(island, count * 2u, c * 2u, physical);
        let vy = gradient_at(island, count * 2u, c * 2u + 1u, physical);
        let v = vec2<f32>(vx.x, vy.x);
        swrite(gradient + c * 2u, v.x);
        swrite(gradient + c * 2u + 1u, v.y);
        if (!finite(v.x) || !finite(v.y)) { fault(island, NONFINITE); return 3.4028234e38; }
        let radius = sread(caps + c);
        let kind = block_kind(island * u.contacts + c);
        if (kind != 0u) {
            let impulse = sread(physical + c * 2u);
            let low = sread(physical + c * 2u + 1u);
            if (kind == 2u && (impulse < 0.0 || (impulse == 0.0 && low < 0.0))) { return 3.4028234e38; }
            var error = abs(vx.x + vx.y);
            if (kind == 2u && impulse == 0.0 && low == 0.0) { error = max(-(vx.x + vx.y), 0.0); }
            if kind == 3u {
                let bounds = contact_data[island*u.contacts+c].xy;
                if impulse < bounds.x || impulse > bounds.y
                    || (impulse == bounds.x && low < 0.0) || (impulse == bounds.y && low > 0.0) { return 3.4028234e38; }
                if bounds.x == bounds.y { error = 0.0; }
                else if impulse == bounds.x && low == 0.0 { error = max(-(vx.x+vx.y),0.0); }
                else if impulse == bounds.y && low == 0.0 { error = max(vx.x+vx.y,0.0); }
            }
            residual = max(residual, error);
            merit += min(error, 1e10) * min(error, 1e10);
            continue;
        }
        if (radius == 0.0) { continue; }
        let value = vec2<f32>(sread(physical + c * 2u), sread(physical + c * 2u + 1u));
        let low=vec2<f32>(sread(residue_base(island)+2u*c),sread(residue_base(island)+2u*c+1u));
        let magnitude = norm2(value);
        let speed = norm2(v);
        if (magnitude == 0.0) { residual = max(residual, speed); merit += min(speed, 1e10) * min(speed, 1e10); continue; }
        let value_exponent = frexp(max(abs(value.x), abs(value.y))).exp;
        let scaled_value = ldexp(value, vec2<i32>(-value_exponent));
        let first = rounded(round_slot, v.x * scaled_value.y);
        let second = rounded(round_slot, v.y * scaled_value.x);
        let cross_low = fma(v.x, scaled_value.y, -first) - fma(v.y, scaled_value.x, -second)
            + vx.y * scaled_value.y - vy.y * scaled_value.x;
        var angular_error = abs((first - second) + cross_low) / norm2(scaled_value);
        if any(low!=vec2<f32>(0.0)) {
            let scaled_low=ldexp(low,vec2<i32>(-value_exponent));
            let cross=pair_add(round_slot,pair_product(round_slot,vx,vec2<f32>(scaled_value.y,scaled_low.y)),
                -pair_product(round_slot,vy,vec2<f32>(scaled_value.x,scaled_low.x)));
            angular_error=abs(cross.x+cross.y)/norm2(scaled_value);
        }
        if (dot(v, scaled_value) > 0.0) { angular_error = speed; }
        let ratio = value / radius;
        var gap = abs(relative_disk_gap(value, radius, round_slot)) / (1.0 + norm2(ratio));
        if any(low!=vec2<f32>(0.0)) { gap=abs(paired_disk_gap(round_slot,value,low,radius))/(1.0+norm2(ratio)); }
        let stride = 2u * u.contacts;
        let row = island * stride * stride + 2u * c * stride + 2u * c;
        let a = abs(grams[row]); let b = abs(grams[row + 1u]); let d = abs(grams[row + stride + 1u]);
        let scale = max(a, max(b, d));
        var radial = 0.0;
        if (scale > 0.0) {
            let curvature = max(a / scale + b / scale, d / scale + b / scale);
            radial = min(speed, ((scale * radius) * gap) * curvature);
        }
        let error = max(angular_error, max(speed * gap, radial));
        residual = max(residual, error); merit += min(error, 1e10) * min(error, 1e10);
        if (magnitude > radius * (1.0 + 2e-6)) { return 3.4028234e38; }
    }
    swrite(round_slot + u.contacts + 1u, merit);
    return residual;
}

fn neighbor(value: f32, direction: i32) -> f32 {
    if (value == 0.0 || direction == 0) { return value; }
    let bits = bitcast<u32>(value);
    if ((value > 0.0) == (direction > 0)) { return bitcast<f32>(bits + 1u); }
    return bitcast<f32>(bits - 1u);
}

fn refine_physical_impulses(island: u32, count: u32, physical: u32, caps: u32, gradient: u32, round_slot: u32) -> f32 {
    let stride=2u*u.contacts;
    var residual=physical_kkt(island,count,physical,caps,gradient,round_slot);
    for (var sweep=0u;sweep<32u && residual>0.999999*u.tolerance;sweep++) {
        for (var c=0u;c<count;c++) {
            let kind=block_kind(island*u.contacts+c);
            let row=2u*c;
            if kind==0u {
                let radius=sread(caps+c);
                if radius==0.0 { continue; }
                let vx=gradient_at(island,2u*count,row,physical);
                let vy=gradient_at(island,2u*count,row+1u,physical);
                let old=vec2<f32>(sread(physical+row),sread(physical+row+1u));
                var x=vec2<f32>(old.x,sread(residue_base(island)+row));
                var y=vec2<f32>(old.y,sread(residue_base(island)+row+1u));
                if norm2(old)>=radius*(1.0-2e-6) && dot(vec2<f32>(vx.x,vy.x),old)<0.0 {
                    let projected=paired_direction(round_slot,-vx,-vy,radius);
                    x=projected[0];y=projected[1];
                } else {
                    let a=grams[island*stride*stride+row*stride+row];
                    let b=grams[island*stride*stride+row*stride+row+1u];
                    let d=grams[island*stride*stride+(row+1u)*stride+row+1u];
                    let determinant=pair_add(round_slot,pair_product(round_slot,vec2<f32>(a,0.0),vec2<f32>(d,0.0)),
                        -pair_product(round_slot,vec2<f32>(b,0.0),vec2<f32>(b,0.0)));
                    if determinant.x>0.0 {
                        let dx=pair_divide(round_slot,pair_add(round_slot,pair_product(round_slot,vec2<f32>(b,0.0),vy),
                            -pair_product(round_slot,vec2<f32>(d,0.0),vx)),determinant);
                        let dy=pair_divide(round_slot,pair_add(round_slot,pair_product(round_slot,vec2<f32>(b,0.0),vx),
                            -pair_product(round_slot,vec2<f32>(a,0.0),vy)),determinant);
                        x=pair_add(round_slot,x,dx);y=pair_add(round_slot,y,dy);
                        if norm2(vec2<f32>(x.x,y.x))>radius {
                            let projected=paired_direction(round_slot,x,y,radius);x=projected[0];y=projected[1];
                        }
                    }
                }
                swrite(physical+row,x.x);swrite(physical+row+1u,y.x);
                swrite(residue_base(island)+row,x.y);swrite(residue_base(island)+row+1u,y.y);
                continue;
            }
            let diagonal=solve_entry(island,stride,row,row);
            if diagonal<=0.0 { continue; }
            let velocity=gradient_at(island,2u*count,row,physical);
            let correction=-(velocity.x+velocity.y)/diagonal;
            let high=sread(physical+row);
            let increment=rounded(round_slot,sread(physical+row+1u)+correction);
            var value=rounded(round_slot,high+increment);
            let virtual_increment=rounded(round_slot,value-high);
            let virtual_high=rounded(round_slot,value-virtual_increment);
            var low=(high-virtual_high)+(increment-virtual_increment);
            let projected=scalar_project(island*u.contacts+c,value,1.0);
            if value!=projected { value=projected; low=0.0; }
            if kind==2u && value==0.0 && low<0.0 { low=0.0; }
            if kind==3u {
                let bounds=contact_data[island*u.contacts+c].xy;
                if (value==bounds.x && low<0.0) || (value==bounds.y && low>0.0) { low=0.0; }
            }
            if !finite(value) || !finite(low) { fault(island,NONFINITE); return 3.4028234e38; }
            swrite(physical+row,value); swrite(physical+row+1u,low);
        }
        residual=physical_kkt(island,count,physical,caps,gradient,round_slot);
    }
    return residual;
}

fn polish(island: u32, count: u32, physical: u32, caps: u32, gradient: u32, factor: u32) -> f32 {
    var best_error = refine_physical_impulses(island, count, physical, caps, gradient, factor);
    if best_error<=0.999999*u.tolerance { return best_error; }
    var best_merit = sread(factor + u.contacts + 1u);
    for (var sweep = 0u; sweep < 4u; sweep += 1u) {
        for (var c = 0u; c < count; c += 1u) {
            let kind = block_kind(island * u.contacts + c);
            var best = vec2<f32>(sread(physical + 2u*c), sread(physical + 2u*c + 1u));
            var best_low=vec2<f32>(sread(residue_base(island)+2u*c),sread(residue_base(island)+2u*c+1u));
            let slip = vec2<f32>(sread(gradient + 2u*c), sread(gradient + 2u*c + 1u));
            let center = best;
            for (var choice = 0u; choice < 10u; choice += 1u) {
                var candidate: vec2<f32>;
                if (choice == 0u) {
                    if (kind != 0u) {
                        let stride = 2u * u.contacts;
                        let diagonal = solve_entry(island,stride,2u*c,2u*c);
                        if (diagonal <= 0.0) { continue; }
                        candidate = vec2<f32>(center.x - slip.x / diagonal, 0.0);
                        candidate.x = scalar_project(island*u.contacts+c,candidate.x,1.0);
                    } else {
                        if (sread(factor + 1u + c) == 0.0 || norm2(slip) <= u.tolerance) { continue; }
                        candidate = -unit2(slip) * sread(caps + c);
                    }
                } else {
                    let offset = choice - 1u;
                    candidate = vec2<f32>(neighbor(center.x, i32(offset % 3u) - 1), neighbor(center.y, i32(offset / 3u) - 1));
                }
                if kind != 0u { candidate = vec2<f32>(scalar_project(island*u.contacts+c,candidate.x,1.0),0.0); }
                if (!finite(candidate.x) || !finite(candidate.y)) { continue; }
                swrite(physical + 2u*c, candidate.x); swrite(physical + 2u*c + 1u, candidate.y);
                swrite(residue_base(island)+2u*c,0.0);swrite(residue_base(island)+2u*c+1u,0.0);
                let error = physical_kkt(island, count, physical, caps, gradient, factor);
                let merit = sread(factor + u.contacts + 1u);
                if (error < best_error || (error == best_error && merit < best_merit)) {
                    best = candidate; best_low=vec2<f32>(0.0); best_error = error; best_merit = merit;
                }
            }
            swrite(physical + 2u*c, best.x); swrite(physical + 2u*c + 1u, best.y);
            swrite(residue_base(island)+2u*c,best_low.x);swrite(residue_base(island)+2u*c+1u,best_low.y);
            best_error = physical_kkt(island, count, physical, caps, gradient, factor);
            best_merit = sread(factor + u.contacts + 1u);
        }
        if (best_error <= 0.999999 * u.tolerance) { break; }
    }
    return best_error;
}

fn basis_residual(basis: u32, stride: u32, n: u32, rank: u32, vector: u32) -> f32 {
    for (var sweep = 0u; sweep < 2u; sweep += 1u) {
        for (var b = 0u; b < rank; b += 1u) {
            let dot = product(basis, stride, n, b, vector) / product(basis, stride, n, b, basis+b*stride);
            for (var i = 0u; i < n; i += 1u) {
                swrite(vector+i, sread(vector+i)-dot*sread(basis+b*stride+i));
            }
        }
    }
    var squared = 0.0;
    for (var i = 0u; i < n; i += 1u) { squared += sread(vector+i)*sread(vector+i); }
    return squared;
}

fn canonicalize(island: u32, count: u32, matrix: u32, basis: u32, vector: u32,
    correction: u32, value: u32, normal: u32, caps: u32) -> bool {
    let stride = 2u * u.contacts;
    let n = 2u * count;
    var has_scalar = false;
    var dimension = count;
    var scale = 0.0;
    for (var c = 0u; c < count; c += 1u) {
        has_scalar = has_scalar || block_kind(island*u.contacts+c) != 0u;
        if (block_kind(island*u.contacts+c) == 0u) { dimension += 1u; }
        scale = max(scale, max(abs(sread(value+2u*c)), abs(sread(value+2u*c+1u))));
    }
    if (!has_scalar || scale == 0.0) { return true; }
    var rank = 0u;
    for (var row = 0u; row <= 2u*n; row += 1u) {
        let coordinate = row-n-1u;
        if (row > n) {
            let c = coordinate/2u;
            let kind = block_kind(island*u.contacts+c);
            let slip = vec2<f32>(sread(normal+2u*c), sread(normal+2u*c+1u));
            var pinned = kind == 0u && (sread(caps+c) == 0.0 || norm2(slip) > u.tolerance);
            pinned = pinned || (coordinate%2u == 0u && scalar_pinned(island*u.contacts+c,slip.x));
            if (!pinned) { continue; }
        }
        var maximum = 0.0;
        for (var i = 0u; i < n; i += 1u) {
            var entry = free_velocity[island*stride+i];
            if (row < n) { entry = solve_entry(island,stride,row,i); }
            if (row > n) { entry = select(0.0, 1.0, i == coordinate); }
            swrite(vector+i, entry);
            maximum = max(maximum, abs(entry));
        }
        if (maximum == 0.0) { continue; }
        var squared = 0.0;
        for (var i = 0u; i < n; i += 1u) {
            let entry = sread(vector+i) / maximum;
            swrite(vector+i, entry);
            squared += entry*entry;
        }
        for (var i = 0u; i < n; i += 1u) { swrite(vector+i, sread(vector+i)/sqrt(squared)); }
        squared = basis_residual(basis, stride, n, rank, vector);
        if (squared > 1e-12) {
            for (var i = 0u; i < n; i += 1u) { swrite(basis+rank*stride+i, sread(vector+i)/sqrt(squared)); }
            rank += 1u;
        }
        if (rank == dimension) { return true; }
    }
    for (var i = 0u; i < n; i += 1u) {
        let c = i/2u;
        let kind = block_kind(island*u.contacts+c);
        let slip = vec2<f32>(sread(normal+2u*c), sread(normal+2u*c+1u));
        let pinned = (kind == 0u && (sread(caps+c) == 0.0 || norm2(slip) > u.tolerance))
            || (i%2u == 0u && scalar_pinned(island*u.contacts+c,slip.x));
        swrite(matrix+n+i, select(0.0, 1.0, pinned));
        swrite(matrix+2u*n+i, sread(value+i));
    }
    for (var reduction = 0u; reduction < count; reduction += 1u) {
        var changed = false;
        for (var c = 0u; c < count; c += 1u) {
            if (block_kind(island*u.contacts+c) != 0u || sread(matrix+n+2u*c) != 0.0) { continue; }
            let impulse = vec2<f32>(sread(value+2u*c), sread(value+2u*c+1u));
            let magnitude = norm2(impulse);
            let cap = sread(caps+c);
            if (magnitude == 0.0 || abs(magnitude-cap) > ALLOCATION_TOLERANCE*cap) { continue; }
            for (var i = 0u; i < n; i += 1u) { swrite(vector+i, 0.0); }
            swrite(vector+2u*c, impulse.x/magnitude); swrite(vector+2u*c+1u, impulse.y/magnitude);
            if (basis_residual(basis, stride, n, rank, vector) > 1e-12) { continue; }
            for (var axis = 0u; axis < 2u; axis += 1u) {
                let coordinate = 2u*c+axis;
                swrite(matrix+n+coordinate, 1.0);
                for (var i = 0u; i < n; i += 1u) { swrite(vector+i, select(0.0, 1.0, i == coordinate)); }
                let squared = basis_residual(basis, stride, n, rank, vector);
                if (squared > 1e-12) {
                    for (var i = 0u; i < n; i += 1u) { swrite(basis+rank*stride+i, sread(vector+i)/sqrt(squared)); }
                    rank += 1u;
                }
            }
            changed = true;
            if (rank == dimension) { return true; }
        }
        if (!changed) { break; }
    }
    for (var i = 0u; i < n; i += 1u) { swrite(vector+i, sread(value+i)/scale); }
    for (var b = 0u; b < rank; b += 1u) { swrite(matrix+b, product(basis, stride, n, b, vector)); }
    for (var i = 0u; i < n; i += 1u) { swrite(value+i, 0.0); swrite(correction+i, 0.0); }
    for (var iteration = 0u; iteration < u.iterations; iteration += 1u) {
        for (var b = 0u; b < rank; b += 1u) {
            let offset = (sread(matrix+b)-product(basis, stride, n, b, value)) / product(basis, stride, n, b, basis+b*stride);
            for (var i = 0u; i < n; i += 1u) {
                swrite(value+i, sread(value+i)+offset*sread(basis+b*stride+i));
            }
        }
        for (var c = 0u; c < count; c += 1u) {
            let r = 2u*c;
            let candidate = vec2<f32>(sread(value+r)+sread(correction+r), sread(value+r+1u)+sread(correction+r+1u));
            var projected = candidate;
            let kind = block_kind(island*u.contacts+c);
            if (kind == 0u) { projected = project(candidate, sread(caps+c)/scale); }
            else {
                projected.y = 0.0;
                projected.x = scalar_project(island*u.contacts+c,projected.x,scale);
            }
            if (sread(matrix+n+r) != 0.0) { projected.x = sread(matrix+2u*n+r)/scale; }
            if (sread(matrix+n+r+1u) != 0.0) { projected.y = sread(matrix+2u*n+r+1u)/scale; }
            swrite(value+r, projected.x); swrite(value+r+1u, projected.y);
            swrite(correction+r, candidate.x-projected.x); swrite(correction+r+1u, candidate.y-projected.y);
        }
        for (var i = 0u; i < n; i += 1u) { swrite(normal+i, sread(value+i)+sread(correction+i)); }
        var residual = 0.0;
        for (var b = 0u; b < rank; b += 1u) {
            residual = max(residual, abs(product(basis, stride, n, b, value)-sread(matrix+b)));
            let dot = product(basis, stride, n, b, normal) / product(basis, stride, n, b, basis+b*stride);
            for (var i = 0u; i < n; i += 1u) { swrite(normal+i, sread(normal+i)-dot*sread(basis+b*stride+i)); }
        }
        for (var i = 0u; i < n; i += 1u) {
            if (!finite(sread(value+i)) || !finite(sread(normal+i))) { return false; }
            residual = max(residual, abs(sread(normal+i)));
        }
        swrite(vector, residual);
        if (residual <= ALLOCATION_TOLERANCE) {
            for (var i = 0u; i < n; i += 1u) {
                var impulse = sread(value+i)*scale;
                if i%2u == 0u && block_kind(island*u.contacts+i/2u) == 3u {
                    let bounds = contact_data[island*u.contacts+i/2u].xy;
                    if sread(value+i) <= bounds.x/scale { impulse = bounds.x; }
                    else if sread(value+i) >= bounds.y/scale { impulse = bounds.y; }
                }
                if (sread(matrix+n+i) != 0.0) { impulse = sread(matrix+2u*n+i); }
                if i%2u == 0u && block_kind(island*u.contacts+i/2u) != 0u {
                    impulse = scalar_project(island*u.contacts+i/2u,impulse,1.0);
                }
                swrite(value+i, impulse);
            }
            return true;
        }
    }
    return false;
}

@compute @workgroup_size(64)
fn solve_disks(@builtin(global_invocation_id) gid: vec3<u32>) {
    let island = gid.x;
    if (island >= u.islands) { return; }
    var empty: Solution;
    for (var c = 0u; c < u.contacts; c += 1u) {
        store_solution(island*u.contacts+c,empty,vec2<f32>(0.0));
        swrite(residue_base(island)+2u*c,0.0);swrite(residue_base(island)+2u*c+1u,0.0);
    }
    if (atomicLoad(&status) != 0u) { return; }
    diagnostics[island] = vec4<u32>(0u);
    if (island_count[0] > u.islands) { fault(island, INPUT); return; }
    if (island >= island_count[0]) { return; }
    let count = contact_counts[island];
    if (count > u.contacts) { fault(island, INPUT); return; }
    if (count == 0u) { return; }
    let stride = 2u * u.contacts;
    let n = 2u * count;
    let matrix = island * u.scratch_stride;
    let factor = matrix + stride * stride;
    let x = factor + stride * stride;
    let y = x + stride;
    let next = y + stride;
    let gradient = next + stride;
    let caps = gradient + stride;
    let modes = caps + u.contacts;
    var gram_scale = 0.0;
    var has_scalar = false;
    for (var c=0u;c<count;c++) { has_scalar=has_scalar || block_kind(island*u.contacts+c)!=0u; }
    for (var i = 0u; i < n; i += 1u) {
        if (!finite(free_velocity[island * stride + i])) { fault(island, NONFINITE); return; }
        for (var j = 0u; j < n; j += 1u) {
                let value = grams[island * stride * stride + i * stride + j];
            if (!finite(value)) { fault(island, NONFINITE); return; }
            gram_scale = max(gram_scale, abs(value));
            if ((i % 2u == 1u && block_kind(island*u.contacts+i/2u) != 0u)
                || (j % 2u == 1u && block_kind(island*u.contacts+j/2u) != 0u)) {
                if (value != 0.0) { fault(island, INPUT); return; }
            }
        }
    }
    for (var c = 0u; c < count; c += 1u) {
        let data = contact_data[island * u.contacts + c];
        let kind = block_kind(island * u.contacts + c);
        if (kind > 3u) { fault(island, INPUT); return; }
        if kind == 3u {
            if (bitcast<u32>(data.x)&0x7fffffffu) > 0x7f800000u
                || (bitcast<u32>(data.y)&0x7fffffffu) > 0x7f800000u
                || data.x > data.y || data.x > 3.4028234e38 || data.y < -3.4028234e38
                || !finite(data.z) || data.z < 0.0 || data.w != 0.0 {
                fault(island,INPUT); return;
            }
        } else {
            if (!finite(data.x) || !finite(data.y) || !finite(data.z) || !finite(data.w)) { fault(island, NONFINITE); return; }
            if kind != 0u && any(data != vec4<f32>(0.0)) { fault(island,INPUT); return; }
            if (data.y < 0.0 || data.x < data.y) { fault(island, INPUT); return; }
        }
        if kind != 0u && free_velocity[island*stride+2u*c+1u] != 0.0 { fault(island,INPUT); return; }
        swrite(modes + c, select(0.0, 1.0, kind == 0u && norm2(data.zw) > CAPTURE));
    }
    if (gram_scale > 0.0) {
        if !positive_semidefinite(island,stride,n,gram_scale,factor) { return; }
    }
    for (var c = 0u; c < count; c += 1u) {
        gram_scale = max(gram_scale,abs(solve_entry(island,stride,2u*c,2u*c)));
    }
    var iterations = 0u;
    for (var transition = 0u; transition <= count; transition += 1u) {
        var impulse_scale = 0.0;
        for (var c = 0u; c < count; c += 1u) {
            let data = contact_data[island * u.contacts + c];
            let kind = block_kind(island*u.contacts+c);
            var cap = 0.0;
            if kind == 0u { cap = select(data.x, data.y, sread(modes + c) != 0.0); }
            impulse_scale = max(impulse_scale, cap);
            if kind == 3u {
                impulse_scale = max(impulse_scale,abs(reference_impulse(island*u.contacts+c)));
            }
            if (block_kind(island*u.contacts+c) != 0u) {
                var estimate = max(1.0, abs(free_velocity[island*stride+2u*c]));
                if (gram_scale > 0.0) { estimate = abs(free_velocity[island*stride+2u*c]) / gram_scale; }
                if (!finite(estimate)) { fault(island, NONFINITE); return; }
                impulse_scale = max(impulse_scale, estimate);
            }
        }
        for (var c = 0u; c < count; c += 1u) {
            let data = contact_data[island * u.contacts + c];
            var cap = 0.0;
            if block_kind(island*u.contacts+c) == 0u { cap = select(data.x, data.y, sread(modes+c) != 0.0); }
            swrite(caps + c, cap);
            swrite(factor + 1u + c, 0.0);
            if (impulse_scale > 0.0) {
                if (cap > 0.0 && cap / impulse_scale == 0.0) { fault(island, NONFINITE); return; }
            }
        }
        var objective_scale = 0.0;
        for (var i = 0u; i < n; i += 1u) {
            var initial = 0.0;
            if i%2u == 0u && impulse_scale > 0.0 { initial = reference_impulse(island*u.contacts+i/2u)/impulse_scale; }
            swrite(x + i, initial);
            swrite(y + i, initial);
            objective_scale = max(objective_scale, abs(free_velocity[island * stride + i]));
            for (var j = 0u; j < n; j += 1u) {
                let original = solve_entry(island,stride,i,j);
                let value = original * impulse_scale;
                if (!finite(value) || (original != 0.0 && impulse_scale > 0.0 && value == 0.0)) { fault(island, NONFINITE); return; }
                swrite(matrix + i * stride + j, value);
                objective_scale = max(objective_scale, abs(value));
            }
        }
        var lipschitz = 0.0;
        if (objective_scale > 0.0) {
            for (var i = 0u; i < n; i += 1u) {
                var sum = 0.0;
                for (var j = 0u; j < n; j += 1u) {
                    swrite(matrix + i * stride + j, sread(matrix + i * stride + j) / (objective_scale));
                    sum += abs(sread(matrix + i * stride + j));
                }
                lipschitz = max(lipschitz, sum);
            }
        }
        var residual = kkt(island, count, x, caps, gradient, impulse_scale);
        if (atomicLoad(&status) != 0u) { return; }
        if (!finite(residual)) { fault(island, NONFINITE); return; }
        var momentum = 1.0;
        for (var iteration = 0u; iteration < u.iterations && residual > 0.999999 * u.tolerance; iteration += 1u) {
            iterations += 1u;
            for (var c = 0u; c < count; c += 1u) {
                let r = c * 2u;
                let kind = block_kind(island*u.contacts+c);
                var candidate: vec2<f32>;
                var boundary = 0.0;
                if (lipschitz == 0.0) {
                    let free = vec2<f32>(free_velocity[island * stride + r], free_velocity[island * stride + r + 1u]);
                    candidate = vec2<f32>(0.0);
                    if kind == 3u {
                        let bounds = contact_data[island*u.contacts+c].xy;
                        let selected = select(select(0.0,bounds.y,free.x < 0.0),bounds.x,free.x > 0.0);
                        if !finite(selected) { fault(island,CONVERGENCE); return; }
                        candidate.x = scalar_project(island*u.contacts+c,selected,1.0)/impulse_scale;
                    } else if (kind != 0u) {
                        let error = select(abs(free.x), max(-free.x, 0.0), kind == 2u);
                        if (error > 0.999999 * u.tolerance) { fault(island, CONVERGENCE); return; }
                    } else if (norm2(free) > 0.0) {
                        candidate = -unit2(free) * (sread(caps + c) / impulse_scale); boundary = 1.0;
                    }
                } else {
                    let g = vec2<f32>(product(matrix, stride, n, r, y) + free_velocity[island * stride + r] / objective_scale,
                        product(matrix, stride, n, r + 1u, y) + free_velocity[island * stride + r + 1u] / objective_scale);
                    candidate = vec2<f32>(sread(y + r), sread(y + r + 1u)) - g / lipschitz;
                    if (kind != 0u) {
                        candidate.y = 0.0;
                        candidate.x = scalar_project(island*u.contacts+c,candidate.x,impulse_scale);
                    } else {
                        boundary = select(0.0, 1.0, norm2(candidate) > sread(caps + c) / impulse_scale);
                        candidate = project(candidate, sread(caps + c) / impulse_scale);
                    }
                }
                if (!finite(candidate.x) || !finite(candidate.y)) { fault(island, NONFINITE); return; }
                swrite(next + r, candidate.x);
                swrite(next + r + 1u, candidate.y);
                swrite(factor + 1u + c, boundary);
            }
            var restart = 0.0;
            for (var i = 0u; i < n; i += 1u) { restart += (sread(y + i) - sread(next + i)) * (sread(next + i) - sread(x + i)); }
            var next_momentum = 0.5 * (1.0 + sqrt(1.0 + 4.0 * momentum * momentum));
            var beta = (momentum - 1.0) / next_momentum;
            if (restart > 0.0) { next_momentum = 1.0; beta = 0.0; }
            for (var i = 0u; i < n; i += 1u) {
                swrite(y + i, sread(next + i) + beta * (sread(next + i) - sread(x + i)));
                swrite(x + i, sread(next + i));
            }
            momentum = next_momentum;
            residual = kkt(island, count, x, caps, gradient, impulse_scale);
            if (atomicLoad(&status) != 0u) { return; }
            if (!finite(residual)) { fault(island, NONFINITE); return; }
            if (iteration % 32u == 31u && (has_scalar || residual < 16.0 * u.tolerance) && residual > 0.999999 * u.tolerance) {
                let polished = polish(island, count, next, caps, gradient, factor);
                if (polished <= 0.999999 * u.tolerance) { residual = polished; break; }
                residual = kkt(island, count, x, caps, gradient, impulse_scale);
            }
        }
        if (residual > 0.999999 * u.tolerance) {
            diagnostics[island] = vec4<u32>(CONVERGENCE, iterations, bitcast<u32>(residual), 1u);
            atomicOr(&status, CONVERGENCE); return;
        }
        var changed = false;
        for (var c = 0u; c < count; c += 1u) {
            let slip = vec2<f32>(sread(gradient + 2u * c), sread(gradient + 2u * c + 1u));
            if (block_kind(island*u.contacts+c) == 0u && sread(modes + c) == 0.0 && norm2(slip) > CAPTURE) { swrite(modes + c, 1.0); changed = true; }
        }
        if (changed) { continue; }
        for (var c=0u;c<count;c++) {
            if block_kind(island*u.contacts+c)!=0u { swrite(next+2u*c+1u,0.0); }
            swrite(residue_base(island)+2u*c,0.0);swrite(residue_base(island)+2u*c+1u,0.0);
        }
        if (!canonicalize(island, count, matrix, factor, x, y, next, gradient, caps)) {
            diagnostics[island] = vec4<u32>(CONVERGENCE, iterations, bitcast<u32>(sread(x)), 4u);
            atomicOr(&status, CONVERGENCE); return;
        }
        residual = refine_physical_impulses(island, count, next, caps, gradient, factor);
        if (residual > 0.999999 * u.tolerance) {
            diagnostics[island] = vec4<u32>(CONVERGENCE, iterations, bitcast<u32>(residual), 5u);
            atomicOr(&status, CONVERGENCE); return;
        }
        var energy = 0.0;
        var reference_energy = 0.0;
        var impulse_sum = 0.0;
        for (var i = 0u; i < n; i += 1u) {
            let impulse = sread(next + i);
            energy += 0.5 * impulse * (free_velocity[island * stride + i] + sread(gradient + i));
            if i%2u==1u && block_kind(island*u.contacts+i/2u)!=0u {
                energy += 0.5*impulse*(free_velocity[island*stride+i-1u]+sread(gradient+i-1u));
            }
            if block_kind(island*u.contacts+i/2u)==0u {
                energy+=0.5*sread(residue_base(island)+i)*(free_velocity[island*stride+i]+sread(gradient+i));
            }
            impulse_sum += abs(impulse);
            if i%2u == 0u {
                let reference = reference_impulse(island*u.contacts+i/2u);
                if reference != 0.0 {
                    var reference_gradient = free_velocity[island*stride+i];
                    for (var j = 0u; j < n; j += 2u) {
                        reference_gradient += solve_entry(island,stride,i,j)*reference_impulse(island*u.contacts+j/2u);
                    }
                    reference_energy += 0.5*reference*(free_velocity[island*stride+i]+reference_gradient);
                }
            }
        }
        if (!finite(energy) || !finite(reference_energy) || !finite(impulse_sum)) { fault(island, NONFINITE); return; }
        if (energy > reference_energy + u.tolerance * impulse_sum) {
            diagnostics[island] = vec4<u32>(CONVERGENCE, iterations, bitcast<u32>(energy), 2u);
            atomicOr(&status, CONVERGENCE); return;
        }
        diagnostics[island] = vec4<u32>(0u, iterations, bitcast<u32>(residual), 0u);
        for (var c = 0u; c < count; c += 1u) {
            let data = contact_data[island * u.contacts + c];
            let slip = vec2<f32>(sread(gradient + 2u * c), sread(gradient + 2u * c + 1u));
            let sliding = sread(modes + c) != 0.0;
            var result: Solution;
            result.impulse = vec2<f32>(sread(next + 2u * c), sread(next + 2u * c + 1u));
            result.slip = slip;
            result.cap = select(data.x, data.y, sliding);
            result.regime = select(3u, 2u, !sliding || norm2(slip) <= u.tolerance);
            if (data.x == 0.0) { result.regime = 0u; }
            let kind = block_kind(island*u.contacts+c);
            if (kind != 0u) {
                result.regime = 3u + kind; result.cap = 0.0;
                let physical_slip = slip_at(island,n,2u*c,next);
                result.slip = vec2<f32>(physical_slip.x+physical_slip.y,0.0);
            }
            result.iterations = iterations;
            result.residual = residual;
            store_solution(island*u.contacts+c,result,
                vec2<f32>(sread(residue_base(island)+2u*c),sread(residue_base(island)+2u*c+1u)));
        }
        return;
    }
    diagnostics[island] = vec4<u32>(CONVERGENCE, iterations, 0u, 3u);
    atomicOr(&status, CONVERGENCE);
}
