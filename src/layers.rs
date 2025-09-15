use std::arch::x86_64::*;

// Coefficients for a quintic polynomial approximation of GELU
// This approximation is accurate within the typical range of [-6, 6].
const GELU_COEFF: [f32; 6] = [
    0.5f32,        // c0
    0.35849124f32, // c1
    0.0f32,        // c2 (no x^2 term in this approx)
    0.05037353f32, // c3
    0.0f32,        // c4 (no x^4 term)
    0.00625f32,    // c5 (x^5 coefficient)
];

// Safe wrapper for GELU forward with AVX2
pub fn gelu_forward(out: &mut [f32], inp: &[f32]) {
    if is_x86_feature_detected!("avx2") {
        unsafe {
            gelu_forward_avx2(out, inp);
        }
    } else {
        gelu_forward_scalar(out, inp);
    }
}

// Safe wrapper for GELU backward with AVX2
pub fn gelu_backward(dinp: &mut [f32], inp: &[f32], dout: &[f32]) {
    if is_x86_feature_detected!("avx2") {
        unsafe {
            gelu_backward_avx2(dinp, inp, dout);
        }
    } else {
        gelu_backward_scalar(dinp, inp, dout);
    }
}

// Scalar GELU forward
fn gelu_forward_scalar(out: &mut [f32], inp: &[f32]) {
    for i in 0..inp.len() {
        let x = inp[i];
        // Compute polynomial using Horner's method
        let poly = GELU_COEFF[0]
            + x * (GELU_COEFF[1]
                + x * (GELU_COEFF[2]
                    + x * (GELU_COEFF[3] + x * (GELU_COEFF[4] + x * GELU_COEFF[5]))));
        out[i] = x * poly;
    }
}

// Scalar GELU backward
fn gelu_backward_scalar(dinp: &mut [f32], inp: &[f32], dout: &[f32]) {
    for i in 0..inp.len() {
        let x = inp[i];
        // Recompute polynomial value
        let poly = GELU_COEFF[0]
            + x * (GELU_COEFF[1]
                + x * (GELU_COEFF[2]
                    + x * (GELU_COEFF[3] + x * (GELU_COEFF[4] + x * GELU_COEFF[5]))));

        // Compute derivative of the polynomial: poly'
        let poly_deriv = GELU_COEFF[1]
            + x * (2.0 * GELU_COEFF[2]
                + x * (3.0 * GELU_COEFF[3]
                    + x * (4.0 * GELU_COEFF[4] + x * (5.0 * GELU_COEFF[5]))));

        // local_grad = d/dx [x * poly] = poly + x * poly'
        let local_grad = poly + x * poly_deriv;

        dinp[i] += local_grad * dout[i];
    }
}

// AVX2 GELU forward
#[target_feature(enable = "avx2")]
unsafe fn gelu_forward_avx2(out: &mut [f32], inp: &[f32]) {
    assert!(out.len() == inp.len());
    let len = inp.len();

    let mut i = 0;
    while i + 8 <= len {
        let x = _mm256_loadu_ps(inp.as_ptr().add(i));

        let c0 = _mm256_set1_ps(GELU_COEFF[0]);
        let c1 = _mm256_set1_ps(GELU_COEFF[1]);
        let c2 = _mm256_set1_ps(GELU_COEFF[2]);
        let c3 = _mm256_set1_ps(GELU_COEFF[3]);
        let c4 = _mm256_set1_ps(GELU_COEFF[4]);
        let c5 = _mm256_set1_ps(GELU_COEFF[5]);

        // Horner's method: c0 + x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * c5))))
        let mut poly = c5;
        poly = _mm256_fmadd_ps(x, poly, c4);
        poly = _mm256_fmadd_ps(x, poly, c3);
        poly = _mm256_fmadd_ps(x, poly, c2);
        poly = _mm256_fmadd_ps(x, poly, c1);
        poly = _mm256_fmadd_ps(x, poly, c0);

        let result = _mm256_mul_ps(x, poly);
        _mm256_storeu_ps(out.as_mut_ptr().add(i), result);

        i += 8;
    }

    // Handle remainder
    for j in i..len {
        let x = inp[j];
        let poly = GELU_COEFF[0]
            + x * (GELU_COEFF[1]
                + x * (GELU_COEFF[2]
                    + x * (GELU_COEFF[3] + x * (GELU_COEFF[4] + x * GELU_COEFF[5]))));
        out[j] = x * poly;
    }
}

// AVX2 GELU backward
#[target_feature(enable = "avx2")]
unsafe fn gelu_backward_avx2(dinp: &mut [f32], inp: &[f32], dout: &[f32]) {
    assert!(dinp.len() == inp.len() && inp.len() == dout.len());
    let len = inp.len();

    let mut i = 0;
    while i + 8 <= len {
        let x = _mm256_loadu_ps(inp.as_ptr().add(i));
        let d = _mm256_loadu_ps(dout.as_ptr().add(i));

        let c0 = _mm256_set1_ps(GELU_COEFF[0]);
        let c1 = _mm256_set1_ps(GELU_COEFF[1]);
        let c2 = _mm256_set1_ps(GELU_COEFF[2]);
        let c3 = _mm256_set1_ps(GELU_COEFF[3]);
        let c4 = _mm256_set1_ps(GELU_COEFF[4]);
        let c5 = _mm256_set1_ps(GELU_COEFF[5]);

        // Recompute poly
        let mut poly = c5;
        poly = _mm256_fmadd_ps(x, poly, c4);
        poly = _mm256_fmadd_ps(x, poly, c3);
        poly = _mm256_fmadd_ps(x, poly, c2);
        poly = _mm256_fmadd_ps(x, poly, c1);
        poly = _mm256_fmadd_ps(x, poly, c0);

        // Constants for derivative
        let two = _mm256_set1_ps(2.0);
        let three = _mm256_set1_ps(3.0);
        let four = _mm256_set1_ps(4.0);
        let five = _mm256_set1_ps(5.0);

        let x2 = _mm256_mul_ps(x, x);
        let x3 = _mm256_mul_ps(x2, x);
        let x4 = _mm256_mul_ps(x2, x2);

        // poly' = c1 + 2*c2*x + 3*c3*x^2 + 4*c4*x^3 + 5*c5*x^4
        let term1 = c1;
        let term2 = _mm256_mul_ps(two, _mm256_mul_ps(c2, x));
        let term3 = _mm256_mul_ps(three, _mm256_mul_ps(c3, x2));
        let term4 = _mm256_mul_ps(four, _mm256_mul_ps(c4, x3));
        let term5 = _mm256_mul_ps(five, _mm256_mul_ps(c5, x4));

        let poly_deriv = _mm256_add_ps(
            term1,
            _mm256_add_ps(term2, _mm256_add_ps(term3, _mm256_add_ps(term4, term5))),
        );

        // local_grad = poly + x * poly_deriv
        let x_poly_deriv = _mm256_mul_ps(x, poly_deriv);
        let local_grad = _mm256_add_ps(poly, x_poly_deriv);

        // dinp += local_grad * dout
        let grad_update = _mm256_mul_ps(local_grad, d);
        let current_dinp = _mm256_loadu_ps(dinp.as_ptr().add(i));
        let new_dinp = _mm256_add_ps(current_dinp, grad_update);
        _mm256_storeu_ps(dinp.as_mut_ptr().add(i), new_dinp);

        i += 8;
    }

    // Handle remainder
    for j in i..len {
        let x = inp[j];
        let poly = GELU_COEFF[0]
            + x * (GELU_COEFF[1]
                + x * (GELU_COEFF[2]
                    + x * (GELU_COEFF[3] + x * (GELU_COEFF[4] + x * GELU_COEFF[5]))));

        let poly_deriv = GELU_COEFF[1]
            + x * (2.0 * GELU_COEFF[2]
                + x * (3.0 * GELU_COEFF[3]
                    + x * (4.0 * GELU_COEFF[4] + x * (5.0 * GELU_COEFF[5]))));

        let local_grad = poly + x * poly_deriv;
        dinp[j] += local_grad * dout[j];
    }
}

// Other layer functions (encoder, layernorm, matmul, attention, residual, softmax, crossentropy)

pub fn encoder_forward(
    out: &mut [f32],
    inp: &[i32],
    wte: &[f32],
    wpe: &[f32],
    B: usize,
    T: usize,
    C: usize,
) {
    for b in 0..B {
        for t in 0..T {
            let out_bt = &mut out[(b * T + t) * C..(b * T + t + 1) * C];
            let ix = inp[b * T + t] as usize;
            let wte_ix = &wte[ix * C..(ix + 1) * C];
            let wpe_t = &wpe[t * C..(t + 1) * C];

            for i in 0..C {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

pub fn encoder_backward(
    dwte: &mut [f32],
    dwpe: &mut [f32],
    dout: &[f32],
    inp: &[i32],
    B: usize,
    T: usize,
    C: usize,
) {
    for b in 0..B {
        for t in 0..T {
            let dout_bt = &dout[(b * T + t) * C..(b * T + t + 1) * C];
            let ix = inp[b * T + t] as usize;
            let dwte_ix = &mut dwte[ix * C..(ix + 1) * C];
            let dwpe_t = &mut dwpe[t * C..(t + 1) * C];

            for i in 0..C {
                let d = dout_bt[i];
                dwte_ix[i] += d;
                dwpe_t[i] += d;
            }
        }
    }
}

pub fn layernorm_forward(
    out: &mut [f32],
    mean: &mut [f32],
    rstd: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: &[f32],
    B: usize,
    T: usize,
    C: usize,
) {
    let eps = 1e-5f32;
    for b in 0..B {
        for t in 0..T {
            let x = &inp[(b * T + t) * C..(b * T + t + 1) * C];
            let out_bt = &mut out[(b * T + t) * C..(b * T + t + 1) * C];

            // Calculate mean
            let m: f32 = x.iter().sum::<f32>() / C as f32;

            // Calculate variance
            let v: f32 = x.iter().map(|&xi| (xi - m).powi(2)).sum::<f32>() / C as f32;

            // Calculate rstd
            let s = 1.0 / (v + eps).sqrt();

            // Normalize and scale/shift
            for i in 0..C {
                let n = s * (x[i] - m);
                out_bt[i] = n * weight[i] + bias[i];
            }

            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

pub fn layernorm_backward(
    dinp: &mut [f32],
    dweight: &mut [f32],
    dbias: &mut [f32],
    dout: &[f32],
    inp: &[f32],
    weight: &[f32],
    mean: &[f32],
    rstd: &[f32],
    B: usize,
    T: usize,
    C: usize,
) {
    for b in 0..B {
        for t in 0..T {
            let dout_bt = &dout[(b * T + t) * C..(b * T + t + 1) * C];
            let inp_bt = &inp[(b * T + t) * C..(b * T + t + 1) * C];
            let dinp_bt = &mut dinp[(b * T + t) * C..(b * T + t + 1) * C];
            let mean_bt = mean[b * T + t];
            let rstd_bt = rstd[b * T + t];

            // First: two reduce operations
            let mut dnorm_mean = 0.0f32;
            let mut dnorm_norm_mean = 0.0f32;
            for i in 0..C {
                let norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                let dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean /= C as f32;
            dnorm_norm_mean /= C as f32;

            // Now accumulate gradients
            for i in 0..C {
                let norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                let dnorm_i = weight[i] * dout_bt[i];

                dbias[i] += dout_bt[i];
                dweight[i] += norm_bti * dout_bt[i];

                let dval = (dnorm_i - dnorm_mean - norm_bti * dnorm_norm_mean) * rstd_bt;
                dinp_bt[i] += dval;
            }
        }
    }
}

pub fn matmul_forward(
    out: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) {
    for b in 0..B {
        for t in 0..T {
            let out_bt = &mut out[(b * T + t) * OC..(b * T + t + 1) * OC];
            let inp_bt = &inp[(b * T + t) * C..(b * T + t + 1) * C];

            for o in 0..OC {
                let mut val = bias.map_or(0.0, |b| b[o]);
                let wrow = &weight[o * C..(o + 1) * C];

                for i in 0..C {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

pub fn matmul_backward(
    dinp: &mut [f32],
    dweight: &mut [f32],
    mut dbias: Option<&mut [f32]>,
    dout: &[f32],
    inp: &[f32],
    weight: &[f32],
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) {
    // Backward into inp
    for b in 0..B {
        for t in 0..T {
            let dout_bt = &dout[(b * T + t) * OC..(b * T + t + 1) * OC];
            let dinp_bt = &mut dinp[(b * T + t) * C..(b * T + t + 1) * C];

            for o in 0..OC {
                let wrow = &weight[o * C..(o + 1) * C];
                let d = dout_bt[o];
                for i in 0..C {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }

    // Backward into weight/bias
    for o in 0..OC {
        for b in 0..B {
            for t in 0..T {
                let dout_bt = &dout[(b * T + t) * OC..(b * T + t + 1) * OC];
                let inp_bt = &inp[(b * T + t) * C..(b * T + t + 1) * C];
                let dwrow = &mut dweight[o * C..(o + 1) * C];
                let d = dout_bt[o];

                if let Some(db) = dbias.as_deref_mut() {
                    db[o] += d;
                }

                for i in 0..C {
                    dwrow[i] += inp_bt[i] * d;
                }
            }
        }
    }
}

pub fn attention_forward(
    out: &mut [f32],
    preatt: &mut [f32],
    att: &mut [f32],
    inp: &[f32],
    B: usize,
    T: usize,
    C: usize,
    NH: usize,
) {
    let C3 = C * 3;
    let hs = C / NH;
    let scale = 1.0 / (hs as f32).sqrt();

    for b in 0..B {
        for t in 0..T {
            for h in 0..NH {
                let query_t = &inp[b * T * C3 + t * C3 + h * hs..][..hs];
                let preatt_bth = &mut preatt[b * NH * T * T + h * T * T + t * T..][..T];
                let att_bth = &mut att[b * NH * T * T + h * T * T + t * T..][..T];

                // Pass 1: Q @ K and find max
                let mut maxval = -10000.0;
                for t2 in 0..=t {
                    let key_t2 = &inp[b * T * C3 + t2 * C3 + h * hs + C..][..hs]; // +C for key
                    let mut val = 0.0;
                    for i in 0..hs {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if val > maxval {
                        maxval = val;
                    }
                    preatt_bth[t2] = val;
                }

                // Pass 2: exp and sum
                let mut expsum = 0.0;
                for t2 in 0..=t {
                    let expv = (preatt_bth[t2] - maxval).exp();
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                let expsum_inv = if expsum == 0.0 { 0.0 } else { 1.0 / expsum };

                // Pass 3: softmax
                for t2 in 0..T {
                    if t2 <= t {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        att_bth[t2] = 0.0; // causal mask
                    }
                }

                // Pass 4: weighted sum of V
                let out_bth = &mut out[b * T * C + t * C + h * hs..][..hs];
                for i in 0..hs {
                    out_bth[i] = 0.0;
                }
                for t2 in 0..=t {
                    let value_t2 = &inp[b * T * C3 + t2 * C3 + h * hs + C * 2..][..hs]; // +C*2 for value
                    let att_btht2 = att_bth[t2];
                    for i in 0..hs {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

pub fn attention_backward(
    dinp: &mut [f32],
    dpreatt: &mut [f32],
    datt: &mut [f32],
    dout: &[f32],
    inp: &[f32],
    att: &[f32],
    B: usize,
    T: usize,
    C: usize,
    NH: usize,
) {
    let C3 = C * 3;
    let hs = C / NH;
    let scale = 1.0 / (hs as f32).sqrt();

    for b in 0..B {
        let base = b * T * C3;
        let att_base = b * NH * T * T;
        let dout_base = b * T * C;
        for t in 0..T {
            for h in 0..NH {
                let att_index = att_base + h * T * T + t * T;
                let att_slice = &att[att_index..att_index + T];
                let datt_slice = &mut datt[att_index..att_index + T];
                let dpreatt_slice = &mut dpreatt[att_index..att_index + T];

                let query_offset = base + t * C3 + h * hs;
                let query_slice = &inp[query_offset..query_offset + hs];
                let mut dquery_accum = vec![0.0f32; hs];

                let dout_offset = dout_base + t * C + h * hs;
                let dout_slice = &dout[dout_offset..dout_offset + hs];

                for t2 in 0..=t {
                    let value_offset = base + t2 * C3 + h * hs + C * 2;
                    let value_slice = &inp[value_offset..value_offset + hs];
                    let att_val = att_slice[t2];
                    let datt_entry = &mut datt_slice[t2];
                    for i in 0..hs {
                        let dout_val = dout_slice[i];
                        *datt_entry += value_slice[i] * dout_val;
                        dinp[value_offset + i] += att_val * dout_val;
                    }
                }

                for t2 in 0..=t {
                    let datt_val = datt_slice[t2];
                    for t3 in 0..=t {
                        let indicator = if t2 == t3 { 1.0 } else { 0.0 };
                        let local_derivative = att_slice[t2] * (indicator - att_slice[t3]);
                        dpreatt_slice[t3] += local_derivative * datt_val;
                    }
                }

                for t2 in 0..=t {
                    let key_offset = base + t2 * C3 + h * hs + C;
                    let dpreatt_val = dpreatt_slice[t2] * scale;
                    for i in 0..hs {
                        let key_val = inp[key_offset + i];
                        dquery_accum[i] += key_val * dpreatt_val;
                        dinp[key_offset + i] += query_slice[i] * dpreatt_val;
                    }
                }

                for i in 0..hs {
                    dinp[query_offset + i] += dquery_accum[i];
                }
            }
        }
    }
}

pub fn residual_forward(out: &mut [f32], inp1: &[f32], inp2: &[f32], N: usize) {
    for i in 0..N {
        out[i] = inp1[i] + inp2[i];
    }
}

pub fn residual_backward(dinp1: &mut [f32], dinp2: &mut [f32], dout: &[f32], N: usize) {
    for i in 0..N {
        dinp1[i] += dout[i];
        dinp2[i] += dout[i];
    }
}

pub fn softmax_forward(probs: &mut [f32], logits: &[f32], B: usize, T: usize, V: usize) {
    for b in 0..B {
        for t in 0..T {
            let logits_bt = &logits[(b * T + t) * V..(b * T + t + 1) * V];
            let probs_bt = &mut probs[(b * T + t) * V..(b * T + t + 1) * V];

            let maxval = logits_bt.iter().fold(-10000.0f32, |acc, &x| acc.max(x));

            let sum: f32 = logits_bt.iter().map(|&x| (x - maxval).exp()).sum();

            for i in 0..V {
                probs_bt[i] = (logits_bt[i] - maxval).exp() / sum;
            }
        }
    }
}

pub fn crossentropy_forward(
    losses: &mut [f32],
    probs: &[f32],
    targets: &[i32],
    B: usize,
    T: usize,
    V: usize,
) {
    for b in 0..B {
        for t in 0..T {
            let probs_bt = &probs[(b * T + t) * V..(b * T + t + 1) * V];
            let ix = targets[b * T + t] as usize;
            losses[b * T + t] = -probs_bt[ix].ln();
        }
    }
}

pub fn crossentropy_softmax_backward(
    dlogits: &mut [f32],
    dlosses: &[f32],
    probs: &[f32],
    targets: &[i32],
    B: usize,
    T: usize,
    V: usize,
) {
    for b in 0..B {
        for t in 0..T {
            let dlogits_bt = &mut dlogits[(b * T + t) * V..(b * T + t + 1) * V];
            let probs_bt = &probs[(b * T + t) * V..(b * T + t + 1) * V];
            let dloss = dlosses[b * T + t];
            let ix = targets[b * T + t] as usize;

            for i in 0..V {
                let p = probs_bt[i];
                let indicator = if i == ix { 1.0 } else { 0.0 };
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) {
        assert!(
            (a - b).abs() <= tol,
            "expected {b}, got {a} with tolerance {tol}"
        );
    }

    #[test]
    fn gelu_forward_matches_polynomial() {
        let inp = [-3.0, -1.0, 0.0, 1.0, 3.0];
        let mut out = [0.0; 5];
        super::gelu_forward(&mut out, &inp);

        for (o, &x) in out.iter().zip(inp.iter()) {
            let poly = GELU_COEFF[0]
                + x
                    * (GELU_COEFF[1]
                        + x
                            * (GELU_COEFF[2]
                                + x
                                    * (GELU_COEFF[3]
                                        + x * (GELU_COEFF[4] + x * GELU_COEFF[5]))));
            approx_eq(*o, x * poly, 1e-5);
        }
    }

    #[test]
    fn gelu_backward_matches_manual() {
        let inp = [-1.0, 0.5, 2.0];
        let dout = [0.7, -0.2, 1.1];
        let mut dinp = [0.3, -0.4, 0.1];
        let dinp_before = dinp;

        super::gelu_backward(&mut dinp, &inp, &dout);

        for ((&x, &d), (out_grad, initial)) in inp
            .iter()
            .zip(dout.iter())
            .zip(dinp.iter().zip(dinp_before.iter()))
        {
            let poly = GELU_COEFF[0]
                + x
                    * (GELU_COEFF[1]
                        + x
                            * (GELU_COEFF[2]
                                + x
                                    * (GELU_COEFF[3]
                                        + x * (GELU_COEFF[4] + x * GELU_COEFF[5]))));
            let poly_deriv = GELU_COEFF[1]
                + x
                    * (2.0 * GELU_COEFF[2]
                        + x
                            * (3.0 * GELU_COEFF[3]
                                + x
                                    * (4.0 * GELU_COEFF[4] + x * (5.0 * GELU_COEFF[5]))));
            let expected = initial + (poly + x * poly_deriv) * d;
            approx_eq(*out_grad, expected, 1e-4);
        }
    }

    #[test]
    fn encoder_forward_adds_embeddings() {
        let B = 1;
        let T = 2;
        let C = 2;
        let inp = [1, 0];
        let wte = [0.1, 0.2, 0.4, 0.5];
        let wpe = [0.01, 0.02, 0.03, 0.04];
        let mut out = [0.0f32; 4];

        super::encoder_forward(&mut out, &inp, &wte, &wpe, B, T, C);

        let expected = [0.41, 0.52, 0.13, 0.24];
        for (o, &e) in out.iter().zip(expected.iter()) {
            approx_eq(*o, e, 1e-6);
        }
    }

    #[test]
    fn encoder_backward_accumulates_gradients() {
        let B = 1;
        let T = 2;
        let C = 2;
        let inp = [1, 0];
        let dout = [0.5, -0.5, 1.0, 2.0];
        let mut dwte = vec![0.0f32; 4];
        let mut dwpe = vec![0.0f32; 4];

        super::encoder_backward(&mut dwte, &mut dwpe, &dout, &inp, B, T, C);

        let expected_dwte = [1.0, 2.0, 0.5, -0.5];
        let expected_dwpe = [0.5, -0.5, 1.0, 2.0];

        for (g, &e) in dwte.iter().zip(expected_dwte.iter()) {
            approx_eq(*g, e, 1e-6);
        }
        for (g, &e) in dwpe.iter().zip(expected_dwpe.iter()) {
            approx_eq(*g, e, 1e-6);
        }
    }

    #[test]
    fn layernorm_forward_normalizes_input() {
        let B = 1;
        let T = 1;
        let C = 3;
        let inp = [1.0, 2.0, 3.0];
        let weight = [1.0, 1.0, 1.0];
        let bias = [0.0, 0.0, 0.0];
        let mut out = [0.0f32; 3];
        let mut mean = [0.0f32; 1];
        let mut rstd = [0.0f32; 1];

        super::layernorm_forward(&mut out, &mut mean, &mut rstd, &inp, &weight, &bias, B, T, C);

        let expected_mean = 2.0;
        let diff0 = 1.0f32 - expected_mean;
        let diff1 = 2.0f32 - expected_mean;
        let diff2 = 3.0f32 - expected_mean;
        let var = (diff0 * diff0 + diff1 * diff1 + diff2 * diff2) / 3.0f32;
        let expected_rstd = 1.0 / (var + 1e-5).sqrt();
        let expected_out: Vec<f32> = inp
            .iter()
            .map(|&x| (x - expected_mean) * expected_rstd)
            .collect();

        approx_eq(mean[0], expected_mean, 1e-6);
        approx_eq(rstd[0], expected_rstd, 1e-6);
        for (o, e) in out.iter().zip(expected_out.iter()) {
            approx_eq(*o, *e, 1e-5);
        }
    }

    #[test]
    fn layernorm_backward_matches_manual_computation() {
        let B = 1;
        let T = 1;
        let C = 2;
        let inp = [1.0, 3.0];
        let weight = [1.5, -0.5];
        let bias = [0.2, -0.3];
        let mut out = [0.0f32; 2];
        let mut mean = [0.0f32; 1];
        let mut rstd = [0.0f32; 1];

        super::layernorm_forward(&mut out, &mut mean, &mut rstd, &inp, &weight, &bias, B, T, C);

        let dout = [0.4, -0.6];
        let mut dinp = [0.1f32, -0.2f32];
        let mut dweight = [0.3f32, 0.5f32];
        let mut dbias = [-0.1f32, 0.2f32];
        let dinp_initial = dinp;
        let dweight_initial = dweight;
        let dbias_initial = dbias;

        super::layernorm_backward(
            &mut dinp,
            &mut dweight,
            &mut dbias,
            &dout,
            &inp,
            &weight,
            &mean,
            &rstd,
            B,
            T,
            C,
        );

        let mut expected_dbias = dbias_initial;
        let mut expected_dweight = dweight_initial;
        let mut expected_dinp = dinp_initial;

        let mut dnorm_mean = 0.0f32;
        let mut dnorm_norm_mean = 0.0f32;
        for i in 0..C {
            let norm = (inp[i] - mean[0]) * rstd[0];
            let dnorm = weight[i] * dout[i];
            dnorm_mean += dnorm;
            dnorm_norm_mean += dnorm * norm;

            expected_dbias[i] += dout[i];
            expected_dweight[i] += norm * dout[i];
        }
        dnorm_mean /= C as f32;
        dnorm_norm_mean /= C as f32;

        for i in 0..C {
            let norm = (inp[i] - mean[0]) * rstd[0];
            let dnorm = weight[i] * dout[i];
            let dval = (dnorm - dnorm_mean - norm * dnorm_norm_mean) * rstd[0];
            expected_dinp[i] += dval;
        }

        for (g, &e) in dinp.iter().zip(expected_dinp.iter()) {
            approx_eq(*g, e, 1e-5);
        }
        for (g, &e) in dweight.iter().zip(expected_dweight.iter()) {
            approx_eq(*g, e, 1e-6);
        }
        for (g, &e) in dbias.iter().zip(expected_dbias.iter()) {
            approx_eq(*g, e, 1e-6);
        }
    }

    #[test]
    fn matmul_forward_multiplies_and_adds_bias() {
        let B = 1;
        let T = 2;
        let C = 2;
        let OC = 3;
        let inp = [1.0, 2.0, -1.0, 0.5];
        let weight = [
            0.2, -0.1,
            0.4, 0.3,
            -0.5, 0.6,
        ];
        let bias = [0.1, -0.2, 0.3];
        let mut out = [0.0f32; 6];

        super::matmul_forward(
            &mut out,
            &inp,
            &weight,
            Some(&bias),
            B,
            T,
            C,
            OC,
        );

        let expected = [0.1, 0.8, 1.0, -0.15, -0.45, 1.1];
        for (g, &e) in out.iter().zip(expected.iter()) {
            approx_eq(*g, e, 1e-6);
        }
    }

    #[test]
    fn matmul_backward_matches_manual_computation() {
        let B = 1;
        let T = 1;
        let C = 2;
        let OC = 2;
        let inp = [1.0, -2.0];
        let weight = [0.5, -0.3, 0.1, 0.4];
        let dout = [0.7, -1.1];
        let mut dinp = [0.2f32, 0.3f32];
        let mut dweight = [0.4f32, -0.5f32, 0.6f32, 0.7f32];
        let mut dbias = [0.1f32, -0.2f32];

        let dinp_initial = dinp;
        let dweight_initial = dweight;
        let dbias_initial = dbias;

        super::matmul_backward(
            &mut dinp,
            &mut dweight,
            Some(&mut dbias),
            &dout,
            &inp,
            &weight,
            B,
            T,
            C,
            OC,
        );

        let mut expected_dinp = dinp_initial;
        let mut expected_dweight = dweight_initial;
        let mut expected_dbias = dbias_initial;

        for o in 0..OC {
            let wrow = &weight[o * C..(o + 1) * C];
            for i in 0..C {
                expected_dinp[i] += wrow[i] * dout[o];
            }
        }
        for o in 0..OC {
            expected_dbias[o] += dout[o];
            for i in 0..C {
                expected_dweight[o * C + i] += inp[i] * dout[o];
            }
        }

        for (g, &e) in dinp.iter().zip(expected_dinp.iter()) {
            approx_eq(*g, e, 1e-6);
        }
        for (g, &e) in dweight.iter().zip(expected_dweight.iter()) {
            approx_eq(*g, e, 1e-6);
        }
        for (g, &e) in dbias.iter().zip(expected_dbias.iter()) {
            approx_eq(*g, e, 1e-6);
        }
    }

    #[test]
    fn attention_forward_produces_expected_outputs() {
        let B = 1;
        let T = 2;
        let C = 2;
        let NH = 1;
        let hs = C / NH;
        let mut inp = vec![0.0f32; B * T * 3 * C];

        // t = 0
        inp[0] = 0.2;
        inp[1] = -0.4;
        inp[2] = 0.1;
        inp[3] = 0.3;
        inp[4] = 0.5;
        inp[5] = -0.1;

        // t = 1
        inp[6] = -0.3;
        inp[7] = 0.7;
        inp[8] = 0.4;
        inp[9] = -0.2;
        inp[10] = -0.6;
        inp[11] = 0.9;

        let mut out = vec![0.0f32; B * T * C];
        let mut preatt = vec![0.0f32; B * NH * T * T];
        let mut att = vec![0.0f32; B * NH * T * T];

        super::attention_forward(&mut out, &mut preatt, &mut att, &inp, B, T, C, NH);

        let scale = 1.0 / (hs as f32).sqrt();
        // Manual computation for t=0
        let dot00 = (0.2 * 0.1 + -0.4 * 0.3) * scale;
        let exp00 = (dot00 - dot00).exp();
        let sum0 = exp00;
        let att00 = exp00 / sum0;
        let v0 = [0.5, -0.1];
        let expected_out0 = [att00 * v0[0], att00 * v0[1]];

        // Manual computation for t=1
        let dot10 = ( -0.3 * 0.1 + 0.7 * 0.3) * scale;
        let dot11 = (-0.3 * 0.4 + 0.7 * -0.2) * scale;
        let max1 = dot10.max(dot11);
        let exp10 = (dot10 - max1).exp();
        let exp11 = (dot11 - max1).exp();
        let sum1 = exp10 + exp11;
        let att10 = exp10 / sum1;
        let att11 = exp11 / sum1;
        let v1 = [-0.6, 0.9];
        let expected_out1 = [att10 * v0[0] + att11 * v1[0], att10 * v0[1] + att11 * v1[1]];

        let expected_preatt_all = vec![dot00, 0.0, dot10, dot11];
        let expected_att_all = vec![att00, 0.0, att10, att11];

        for (v, &e) in preatt.iter().zip(expected_preatt_all.iter()) {
            approx_eq(*v, e, 1e-5);
        }
        for (v, &e) in att.iter().zip(expected_att_all.iter()) {
            approx_eq(*v, e, 1e-5);
        }
        let expected_out_all = vec![expected_out0[0], expected_out0[1], expected_out1[0], expected_out1[1]];
        for (v, &e) in out.iter().zip(expected_out_all.iter()) {
            approx_eq(*v, e, 1e-5);
        }
    }

    #[test]
    fn attention_backward_zero_dout_produces_zero_gradients() {
        let B = 1;
        let T = 2;
        let C = 2;
        let NH = 1;
        let inp = vec![0.1f32; B * T * 3 * C];
        let mut out = vec![0.0f32; B * T * C];
        let mut preatt = vec![0.0f32; B * NH * T * T];
        let mut att = vec![0.0f32; B * NH * T * T];

        super::attention_forward(&mut out, &mut preatt, &mut att, &inp, B, T, C, NH);

        let mut dinp = vec![0.0f32; B * T * 3 * C];
        let mut dpreatt = vec![0.0f32; B * NH * T * T];
        let mut datt = vec![0.0f32; B * NH * T * T];
        let dout = vec![0.0f32; B * T * C];

        super::attention_backward(
            &mut dinp,
            &mut dpreatt,
            &mut datt,
            &dout,
            &inp,
            &att,
            B,
            T,
            C,
            NH,
        );

        assert!(dinp.iter().all(|&x| x == 0.0));
        assert!(dpreatt.iter().all(|&x| x == 0.0));
        assert!(datt.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn attention_backward_matches_numeric_gradient() {
        let B = 1;
        let T = 2;
        let C = 2;
        let NH = 1;
        let mut inp = vec![
            0.2, -0.4, 0.1, 0.3, 0.5, -0.1,
            -0.3, 0.7, 0.4, -0.2, -0.6, 0.9,
        ];
        let mut out = vec![0.0f32; B * T * C];
        let mut preatt = vec![0.0f32; B * NH * T * T];
        let mut att = vec![0.0f32; B * NH * T * T];

        super::attention_forward(&mut out, &mut preatt, &mut att, &inp, B, T, C, NH);

        let dout = vec![0.2, -0.1, 0.05, 0.5];
        let mut dinp = vec![0.0f32; B * T * 3 * C];
        let mut dpreatt = vec![0.0f32; B * NH * T * T];
        let mut datt = vec![0.0f32; B * NH * T * T];

        super::attention_backward(
            &mut dinp,
            &mut dpreatt,
            &mut datt,
            &dout,
            &inp,
            &att,
            B,
            T,
            C,
            NH,
        );

        let mut numerical = vec![0.0f32; dinp.len()];
        let eps = 1e-3;
        for i in 0..inp.len() {
            let original = inp[i];
            inp[i] = original + eps;
            let mut out_plus = vec![0.0f32; B * T * C];
            let mut preatt_plus = vec![0.0f32; B * NH * T * T];
            let mut att_plus = vec![0.0f32; B * NH * T * T];
            super::attention_forward(&mut out_plus, &mut preatt_plus, &mut att_plus, &inp, B, T, C, NH);
            let loss_plus: f32 = out_plus.iter().zip(dout.iter()).map(|(o, &d)| o * d).sum();

            inp[i] = original - eps;
            let mut out_minus = vec![0.0f32; B * T * C];
            let mut preatt_minus = vec![0.0f32; B * NH * T * T];
            let mut att_minus = vec![0.0f32; B * NH * T * T];
            super::attention_forward(&mut out_minus, &mut preatt_minus, &mut att_minus, &inp, B, T, C, NH);
            let loss_minus: f32 = out_minus.iter().zip(dout.iter()).map(|(o, &d)| o * d).sum();

            numerical[i] = (loss_plus - loss_minus) / (2.0 * eps);
            inp[i] = original;
        }

        for (analytical, numerical) in dinp.iter().zip(numerical.iter()) {
            approx_eq(*analytical, *numerical, 2e-2);
        }
    }

    #[test]
    fn residual_forward_and_backward_work() {
        let inp1 = [1.0, -2.0, 3.0];
        let inp2 = [0.5, 0.5, -1.0];
        let mut out = [0.0f32; 3];
        super::residual_forward(&mut out, &inp1, &inp2, inp1.len());
        let expected_out = [1.5, -1.5, 2.0];
        for (o, &e) in out.iter().zip(expected_out.iter()) {
            approx_eq(*o, e, 1e-6);
        }

        let dout = [0.2, -0.4, 0.6];
        let mut dinp1 = [0.1f32, 0.0f32, -0.1f32];
        let mut dinp2 = [-0.2f32, 0.3f32, 0.4f32];
        super::residual_backward(&mut dinp1, &mut dinp2, &dout, inp1.len());
        let expected_dinp1 = [0.30000004, -0.4, 0.5];
        let expected_dinp2 = [0.0, -0.10000001, 1.0];
        for (g, &e) in dinp1.iter().zip(expected_dinp1.iter()) {
            approx_eq(*g, e, 1e-6);
        }
        for (g, &e) in dinp2.iter().zip(expected_dinp2.iter()) {
            approx_eq(*g, e, 1e-6);
        }
    }

    #[test]
    fn softmax_forward_produces_probabilities() {
        let B = 1;
        let T = 1;
        let V = 3;
        let logits = [1.0, 2.0, 0.5];
        let mut probs = [0.0f32; 3];

        super::softmax_forward(&mut probs, &logits, B, T, V);

        let sum: f32 = probs.iter().sum();
        approx_eq(sum, 1.0, 1e-6);
        assert!(probs.iter().all(|&p| p > 0.0));
    }

    #[test]
    fn crossentropy_forward_matches_log_prob() {
        let B = 1;
        let T = 2;
        let V = 3;
        let probs = [0.1, 0.7, 0.2, 0.25, 0.25, 0.5];
        let targets = [1, 2];
        let mut losses = [0.0f32; 2];

        super::crossentropy_forward(&mut losses, &probs, &targets, B, T, V);

        approx_eq(losses[0], -0.7f32.ln(), 1e-6);
        approx_eq(losses[1], -0.5f32.ln(), 1e-6);
    }

    #[test]
    fn crossentropy_softmax_backward_matches_theory() {
        let B = 1;
        let T = 1;
        let V = 3;
        let probs = [0.2, 0.5, 0.3];
        let targets = [1];
        let dlosses = [2.0];
        let mut dlogits = [0.5f32, -0.3f32, 0.7f32];
        let dlogits_initial = dlogits;

        super::crossentropy_softmax_backward(&mut dlogits, &dlosses, &probs, &targets, B, T, V);

        let mut expected = dlogits_initial;
        for i in 0..V {
            let indicator = if i == 1 { 1.0 } else { 0.0 };
            expected[i] += (probs[i] - indicator) * dlosses[0];
        }

        for (g, &e) in dlogits.iter().zip(expected.iter()) {
            approx_eq(*g, e, 1e-6);
        }
    }
}
