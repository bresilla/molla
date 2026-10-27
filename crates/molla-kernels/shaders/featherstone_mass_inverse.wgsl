struct JointInverse { matrix: array<f32,36>, valid: bool }
fn joint_psd_inverse(input: array<f32,36>, width: u32) -> JointInverse {
    return joint_psd_inverse_rank(input,width,width);
}
fn joint_psd_inverse_rank(input: array<f32,36>, width: u32, rank: u32) -> JointInverse {
    var result: JointInverse;
    result.valid = true;
    var matrix = input;
    var vectors: array<f32,36>;
    var scale = 0.0;
    for (var r = 0u; r < width; r++) {
        vectors[r*6u+r] = 1.0;
        for (var c = 0u; c < width; c++) {
            if !(abs(matrix[r*6u+c]) <= 3.402823e38) { result.valid = false; return result; }
            scale = max(scale,abs(matrix[r*6u+c]));
        }
    }
    if scale == 0.0 { return result; }
    for (var i = 0u; i < 36u; i++) { matrix[i] /= scale; }
    for (var sweep = 0u; sweep < 48u; sweep++) {
        var changed = false;
        for (var p = 0u; p < width; p++) {
            for (var q = p+1u; q < width; q++) {
                let cross = matrix[p*6u+q];
                if abs(cross) <= 1e-12 { continue; }
                changed = true;
                let tau = (matrix[q*6u+q]-matrix[p*6u+p])/(2.0*cross);
                var tangent = 0.5/tau;
                if abs(tau) < 1e10 { tangent = select(-1.0,1.0,tau>=0.0)/(abs(tau)+sqrt(1.0+tau*tau)); }
                let cosine = inverseSqrt(1.0+tangent*tangent);
                let sine = tangent*cosine;
                matrix[p*6u+p] -= tangent*cross;
                matrix[q*6u+q] += tangent*cross;
                matrix[p*6u+q] = 0.0; matrix[q*6u+p] = 0.0;
                for (var k = 0u; k < width; k++) {
                    if k != p && k != q {
                        let kp = matrix[k*6u+p]; let kq = matrix[k*6u+q];
                        let vp = cosine*kp-sine*kq; let vq = sine*kp+cosine*kq;
                        matrix[k*6u+p] = vp; matrix[p*6u+k] = vp;
                        matrix[k*6u+q] = vq; matrix[q*6u+k] = vq;
                    }
                    let kp = vectors[k*6u+p]; let kq = vectors[k*6u+q];
                    vectors[k*6u+p] = cosine*kp-sine*kq;
                    vectors[k*6u+q] = sine*kp+cosine*kq;
                }
            }
        }
        if !changed { break; }
    }
    var peak = 0.0;
    for (var i = 0u; i < width; i++) { peak = max(peak,matrix[i*6u+i]); }
    for (var i = 0u; i < width; i++) {
        if matrix[i*6u+i] < -peak*1e-6 { result.valid = false; return result; }
        for (var j = i+1u; j < width; j++) {
            if abs(matrix[i*6u+j]) > peak*1e-6 { result.valid = false; return result; }
        }
        var order=0u;
        for (var j=0u;j<width;j++) {
            if matrix[j*6u+j]>matrix[i*6u+i] || (matrix[j*6u+j]==matrix[i*6u+i] && j<i) { order++; }
        }
        if order>=rank || matrix[i*6u+i] <= peak*1e-12 { continue; }
        let inverse = (1.0/scale)/matrix[i*6u+i];
        for (var r = 0u; r < width; r++) {
            for (var c = 0u; c < width; c++) {
                result.matrix[r*6u+c] = fma(vectors[r*6u+i]*vectors[c*6u+i],inverse,result.matrix[r*6u+c]);
                if !(abs(result.matrix[r*6u+c]) <= 3.402823e38) { result.valid = false; }
            }
        }
    }
    return result;
}

