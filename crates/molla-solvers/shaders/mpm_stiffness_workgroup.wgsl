var<workgroup> wg_stiffness_tail: array<atomic<u32>, 64>;
var<workgroup> wg_stiffness_round: array<atomic<u32>, 64>;
var<private> stiffness_lane: u32;

fn scatter_stiffness_round(value: f32) -> f32 {
    atomicStore(&wg_stiffness_round[stiffness_lane], bitcast<u32>(value));
    return bitcast<f32>(atomicAdd(&wg_stiffness_round[stiffness_lane], 0u));
}

fn scatter_stiffness_tail(slot: u32, value: f32) {
    var old_bits = atomicLoad(&wg_stiffness_tail[slot]);
    loop {
        let next = bitcast<u32>(bitcast<f32>(old_bits) + value);
        let result = atomicCompareExchangeWeak(&wg_stiffness_tail[slot], old_bits, next);
        if result.exchanged { break; }
        old_bits = result.old_value;
    }
}

fn wg_atomic_add_kdiag(slot: u32, val: f32) {
    var old_bits = atomicLoad(&wg_kdiag[slot]);
    loop {
        let old = bitcast<f32>(old_bits);
        let next = scatter_stiffness_round(old + val);
        let result = atomicCompareExchangeWeak(&wg_kdiag[slot], old_bits, bitcast<u32>(next));
        if result.exchanged {
            let large = max(old, val);
            let small = min(old, val);
            let error = scatter_stiffness_round(scatter_stiffness_round(large - next) + small);
            scatter_stiffness_tail(slot, error);
            break;
        }
        old_bits = result.old_value;
    }
}
