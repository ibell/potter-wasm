//! Genuine multi-dimensional adaptive cubature (hcubature), pure Rust.
//!
//! Uses the degree-7 Genz-Malik fully-symmetric rule with its embedded degree-5
//! rule for a local error estimate, and a global region heap: repeatedly bisect
//! the single region with the largest estimated error, along the axis whose
//! fourth difference is largest. This is the scheme behind Steven Johnson's
//! `hcubature` (and DCUHRE) — fundamentally better than nesting 1-D adaptive
//! rules, because the error is estimated on the true d-dimensional integrand and
//! refinement is driven globally to where it actually helps.
//!
//! Genz-Malik point sets over a box [c-h, c+h] (lambda * halfwidth offsets):
//!   set 0: center                                   (1 point)
//!   set 2: +/- lambda2 e_i                          (2d points)
//!   set 3: +/- lambda4 e_i                          (2d points)
//!   set 4: +/- lambda4 e_i +/- lambda4 e_j (i<j)    (2d(d-1) points)
//!   set 5: (+/- lambda5, ..., +/- lambda5)          (2^d points)
//! For d=3 that is 33 integrand evaluations per region.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

struct Region {
    center: Vec<f64>,
    half: Vec<f64>,
    value: f64,
    err: f64,
    split: usize,
}

impl PartialEq for Region {
    fn eq(&self, o: &Self) -> bool {
        self.err == o.err
    }
}
impl Eq for Region {}
impl PartialOrd for Region {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Region {
    fn cmp(&self, o: &Self) -> Ordering {
        // max-heap on error
        self.err.partial_cmp(&o.err).unwrap_or(Ordering::Equal)
    }
}

/// Evaluate the Genz-Malik rule on the box centered at `c` with half-widths `h`.
/// Returns (degree-7 estimate, error estimate, axis to bisect).
fn rule<F: Fn(&[f64]) -> f64>(
    f: &F,
    dim: usize,
    c: &[f64],
    h: &[f64],
    p: &mut [f64],
    nevals: &mut usize,
) -> (f64, f64, usize) {
    let lambda2 = (9.0_f64 / 70.0).sqrt();
    let lambda4 = (9.0_f64 / 10.0).sqrt();
    let lambda5 = (9.0_f64 / 19.0).sqrt();
    let ratio = (9.0 / 70.0) / (9.0 / 10.0); // lambda2^2 / lambda4^2 = 1/7

    let d = dim as f64;
    let w1 = (12824.0 - 9120.0 * d + 400.0 * d * d) / 19683.0;
    let w2 = 980.0 / 6561.0;
    let w3 = (1820.0 - 400.0 * d) / 19683.0;
    let w4 = 200.0 / 19683.0;
    let w5 = 6859.0 / 19683.0 / 2.0_f64.powi(dim as i32);
    let we1 = (729.0 - 950.0 * d + 50.0 * d * d) / 729.0;
    let we2 = 245.0 / 486.0;
    let we3 = (265.0 - 100.0 * d) / 1458.0;
    let we4 = 25.0 / 729.0;

    p.copy_from_slice(c);
    let f0 = f(p);
    *nevals += 1;

    let (mut sum2, mut sum3, mut sum4, mut sum5) = (0.0, 0.0, 0.0, 0.0);
    let mut maxdiff = -1.0;
    let mut split = 0usize;

    // sets 2 and 3 (single-axis offsets) + fourth-difference split heuristic
    for i in 0..dim {
        p.copy_from_slice(c);
        p[i] = c[i] + lambda2 * h[i];
        let f2p = f(p);
        p[i] = c[i] - lambda2 * h[i];
        let f2m = f(p);
        p[i] = c[i] + lambda4 * h[i];
        let f3p = f(p);
        p[i] = c[i] - lambda4 * h[i];
        let f3m = f(p);
        *nevals += 4;
        sum2 += f2p + f2m;
        sum3 += f3p + f3m;
        let diff = ((f2p + f2m - 2.0 * f0) - ratio * (f3p + f3m - 2.0 * f0)).abs();
        if diff > maxdiff {
            maxdiff = diff;
            split = i;
        }
    }

    // set 4: all pairs of axes, +/- lambda4 on both
    for i in 0..dim {
        for j in (i + 1)..dim {
            for &si in &[-1.0, 1.0] {
                for &sj in &[-1.0, 1.0] {
                    p.copy_from_slice(c);
                    p[i] = c[i] + si * lambda4 * h[i];
                    p[j] = c[j] + sj * lambda4 * h[j];
                    sum4 += f(p);
                    *nevals += 1;
                }
            }
        }
    }

    // set 5: 2^dim corner points at +/- lambda5 on every axis
    for mask in 0..(1usize << dim) {
        for i in 0..dim {
            let s = if (mask >> i) & 1 == 1 { 1.0 } else { -1.0 };
            p[i] = c[i] + s * lambda5 * h[i];
        }
        sum5 += f(p);
        *nevals += 1;
    }

    let mut vol = 1.0;
    for hi in h.iter().take(dim) {
        vol *= 2.0 * hi;
    }

    let result = vol * (w1 * f0 + w2 * sum2 + w3 * sum3 + w4 * sum4 + w5 * sum5);
    let result_e = vol * (we1 * f0 + we2 * sum2 + we3 * sum3 + we4 * sum4);
    (result, (result - result_e).abs(), split)
}

/// Adaptively integrate `f` over the box [a, b] (d-dimensional) to the requested
/// tolerances. Returns (integral, error estimate, number of integrand evals).
pub fn hcubature<F: Fn(&[f64]) -> f64>(
    dim: usize,
    f: &F,
    a: &[f64],
    b: &[f64],
    abstol: f64,
    reltol: f64,
    maxevals: usize,
) -> (f64, f64, usize) {
    let mut p = vec![0.0; dim];
    let mut nevals = 0usize;

    let center: Vec<f64> = (0..dim).map(|i| 0.5 * (a[i] + b[i])).collect();
    let half: Vec<f64> = (0..dim).map(|i| 0.5 * (b[i] - a[i])).collect();
    let (v0, e0, s0) = rule(f, dim, &center, &half, &mut p, &mut nevals);

    let mut total_val = v0;
    let mut total_err = e0;
    let mut heap = BinaryHeap::new();
    heap.push(Region {
        center,
        half,
        value: v0,
        err: e0,
        split: s0,
    });

    // Combined tolerance: absolute floor + relative term (the SUNDIALS/scipy
    // convention, err <= abstol + reltol*|I|) — a single smooth criterion rather
    // than a hard max() switch between the two regimes.
    while total_err > abstol + reltol * total_val.abs() && nevals < maxevals {
        let r = match heap.pop() {
            Some(r) => r,
            None => break,
        };
        total_val -= r.value;
        total_err -= r.err;

        let d = r.split;
        let mut half_c = r.half.clone();
        half_c[d] *= 0.5; // child half-width along the split axis
        let mut c1 = r.center.clone();
        c1[d] -= half_c[d];
        let mut c2 = r.center.clone();
        c2[d] += half_c[d];

        let (v1, e1, s1) = rule(f, dim, &c1, &half_c, &mut p, &mut nevals);
        let (v2, e2, s2) = rule(f, dim, &c2, &half_c, &mut p, &mut nevals);
        total_val += v1 + v2;
        total_err += e1 + e2;

        heap.push(Region {
            center: c1,
            half: half_c.clone(),
            value: v1,
            err: e1,
            split: s1,
        });
        heap.push(Region {
            center: c2,
            half: half_c,
            value: v2,
            err: e2,
            split: s2,
        });
    }

    (total_val, total_err, nevals)
}

// ---- vector (3-component) cubature: integrate [f0,f1,f2] on one shared grid ----

struct Region3 {
    center: Vec<f64>,
    half: Vec<f64>,
    value: [f64; 3],
    err: [f64; 3],
    maxerr: f64,
    split: usize,
}
impl PartialEq for Region3 {
    fn eq(&self, o: &Self) -> bool {
        self.maxerr == o.maxerr
    }
}
impl Eq for Region3 {}
impl PartialOrd for Region3 {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Region3 {
    fn cmp(&self, o: &Self) -> Ordering {
        self.maxerr.partial_cmp(&o.maxerr).unwrap_or(Ordering::Equal)
    }
}

/// Genz-Malik degree-7 (+ embedded degree-5 error) rule for a 3-vector integrand,
/// evaluated at the SAME nodes for all components. Returns (value[3], err[3], split).
fn rule3<F: Fn(&[f64]) -> [f64; 3]>(
    f: &F,
    dim: usize,
    c: &[f64],
    h: &[f64],
    p: &mut [f64],
    nevals: &mut usize,
) -> ([f64; 3], [f64; 3], usize) {
    let lambda2 = (9.0_f64 / 70.0).sqrt();
    let lambda4 = (9.0_f64 / 10.0).sqrt();
    let lambda5 = (9.0_f64 / 19.0).sqrt();
    let ratio = (9.0 / 70.0) / (9.0 / 10.0);

    let d = dim as f64;
    let w1 = (12824.0 - 9120.0 * d + 400.0 * d * d) / 19683.0;
    let w2 = 980.0 / 6561.0;
    let w3 = (1820.0 - 400.0 * d) / 19683.0;
    let w4 = 200.0 / 19683.0;
    let w5 = 6859.0 / 19683.0 / 2.0_f64.powi(dim as i32);
    let we1 = (729.0 - 950.0 * d + 50.0 * d * d) / 729.0;
    let we2 = 245.0 / 486.0;
    let we3 = (265.0 - 100.0 * d) / 1458.0;
    let we4 = 25.0 / 729.0;

    p.copy_from_slice(c);
    let f0 = f(p);
    *nevals += 1;

    let mut sum2 = [0.0; 3];
    let mut sum3 = [0.0; 3];
    let mut sum4 = [0.0; 3];
    let mut sum5 = [0.0; 3];
    let mut maxdiff = -1.0;
    let mut split = 0usize;

    for i in 0..dim {
        p.copy_from_slice(c);
        p[i] = c[i] + lambda2 * h[i];
        let f2p = f(p);
        p[i] = c[i] - lambda2 * h[i];
        let f2m = f(p);
        p[i] = c[i] + lambda4 * h[i];
        let f3p = f(p);
        p[i] = c[i] - lambda4 * h[i];
        let f3m = f(p);
        *nevals += 4;
        let mut diff = 0.0;
        for k in 0..3 {
            sum2[k] += f2p[k] + f2m[k];
            sum3[k] += f3p[k] + f3m[k];
            diff += ((f2p[k] + f2m[k] - 2.0 * f0[k]) - ratio * (f3p[k] + f3m[k] - 2.0 * f0[k])).abs();
        }
        if diff > maxdiff {
            maxdiff = diff;
            split = i;
        }
    }

    for i in 0..dim {
        for j in (i + 1)..dim {
            for &si in &[-1.0, 1.0] {
                for &sj in &[-1.0, 1.0] {
                    p.copy_from_slice(c);
                    p[i] = c[i] + si * lambda4 * h[i];
                    p[j] = c[j] + sj * lambda4 * h[j];
                    let fv = f(p);
                    for k in 0..3 {
                        sum4[k] += fv[k];
                    }
                    *nevals += 1;
                }
            }
        }
    }

    for mask in 0..(1usize << dim) {
        for i in 0..dim {
            let s = if (mask >> i) & 1 == 1 { 1.0 } else { -1.0 };
            p[i] = c[i] + s * lambda5 * h[i];
        }
        let fv = f(p);
        for k in 0..3 {
            sum5[k] += fv[k];
        }
        *nevals += 1;
    }

    let mut vol = 1.0;
    for hi in h.iter().take(dim) {
        vol *= 2.0 * hi;
    }
    let mut value = [0.0; 3];
    let mut err = [0.0; 3];
    for k in 0..3 {
        let res = vol * (w1 * f0[k] + w2 * sum2[k] + w3 * sum3[k] + w4 * sum4[k] + w5 * sum5[k]);
        let res_e = vol * (we1 * f0[k] + we2 * sum2[k] + we3 * sum3[k] + we4 * sum4[k]);
        value[k] = res;
        err[k] = (res - res_e).abs();
    }
    (value, err, split)
}

/// Vector analog of `hcubature`: integrate a `[f64;3]` integrand on one shared region
/// subdivision. Refines the region with the largest max-component error until EVERY
/// component meets `err_k <= abstol + reltol*|val_k|`. Returns (value[3], err[3], nevals).
pub fn hcubature3<F: Fn(&[f64]) -> [f64; 3]>(
    dim: usize,
    f: &F,
    a: &[f64],
    b: &[f64],
    abstol: f64,
    reltol: f64,
    maxevals: usize,
) -> ([f64; 3], [f64; 3], usize) {
    let mut p = vec![0.0; dim];
    let mut nevals = 0usize;
    let center: Vec<f64> = (0..dim).map(|i| 0.5 * (a[i] + b[i])).collect();
    let half: Vec<f64> = (0..dim).map(|i| 0.5 * (b[i] - a[i])).collect();
    let (v0, e0, s0) = rule3(f, dim, &center, &half, &mut p, &mut nevals);

    let mut total_val = v0;
    let mut total_err = e0;
    let mut heap = BinaryHeap::new();
    let maxerr0 = e0.iter().cloned().fold(0.0, f64::max);
    heap.push(Region3 { center, half, value: v0, err: e0, maxerr: maxerr0, split: s0 });

    let converged = |tv: &[f64; 3], te: &[f64; 3]| -> bool {
        (0..3).all(|k| te[k] <= abstol + reltol * tv[k].abs())
    };

    while !converged(&total_val, &total_err) && nevals < maxevals {
        let r = match heap.pop() {
            Some(r) => r,
            None => break,
        };
        for k in 0..3 {
            total_val[k] -= r.value[k];
            total_err[k] -= r.err[k];
        }
        let dd = r.split;
        let mut half_c = r.half.clone();
        half_c[dd] *= 0.5;
        let mut c1 = r.center.clone();
        c1[dd] -= half_c[dd];
        let mut c2 = r.center.clone();
        c2[dd] += half_c[dd];

        let (v1, e1, s1) = rule3(f, dim, &c1, &half_c, &mut p, &mut nevals);
        let (v2, e2, s2) = rule3(f, dim, &c2, &half_c, &mut p, &mut nevals);
        for k in 0..3 {
            total_val[k] += v1[k] + v2[k];
            total_err[k] += e1[k] + e2[k];
        }
        let m1 = e1.iter().cloned().fold(0.0, f64::max);
        let m2 = e2.iter().cloned().fold(0.0, f64::max);
        heap.push(Region3 { center: c1, half: half_c.clone(), value: v1, err: e1, maxerr: m1, split: s1 });
        heap.push(Region3 { center: c2, half: half_c, value: v2, err: e2, maxerr: m2, split: s2 });
    }
    (total_val, total_err, nevals)
}
