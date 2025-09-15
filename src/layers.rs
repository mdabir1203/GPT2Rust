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
    dbias: Option<&mut [f32]>,
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

                if let Some(db) = dbias {
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
        for t in 0..T {
            for h in 0..NH {
                let att_bth = &att[b * NH * T * T + h * T * T + t * T..][..T];
                let datt_bth = &mut datt[b * NH * T * T + h * T * T + t * T..][..T];
                let dpreatt_bth = &mut dpreatt[b * NH * T * T + h * T * T + t * T..][..T];
                let dquery_t = &mut dinp[b * T * C3 + t * C3 + h * hs..][..hs];
                let query_t = &inp[b * T * C3 + t * C3 + h * hs..][..hs];

                // Backward pass 4: through value accumulation
                let dout_bth = &dout[b * T * C + t * C + h * hs..][..hs];
                for t2 in 0..=t {
                    let value_t2 = &inp[b * T * C3 + t2 * C3 + h * hs + C * 2..][..hs]; // +C*2 for value
                    let dvalue_t2 = &mut dinp[b * T * C3 + t2 * C3 + h * hs + C * 2..][..hs];
                    for i in 0..hs {
                        datt_bth[t2] += value_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }

                // Backward pass 2 & 3: through softmax
                for t2 in 0..=t {
                    for t3 in 0..=t {
                        let indicator = if t2 == t3 { 1.0 } else { 0.0 };
                        let local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                    }
                }

                // Backward pass 1: through Q @ K matmul
                for t2 in 0..=t {
                    let key_t2 = &inp[b * T * C3 + t2 * C3 + h * hs + C..][..hs]; // +C for key
                    let dkey_t2 = &mut dinp[b * T * C3 + t2 * C3 + h * hs + C..][..hs];
                    for i in 0..hs {
                        dquery_t[i] += key_t2[i] * dpreatt_bth[t2] * scale;
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale;
                    }
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

            let maxval = logits_bt.iter().fold(-10000.0, |acc, &x| acc.max(x));

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
