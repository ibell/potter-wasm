//! 1-D adaptive quadrature — the "cubature in Rust" piece (1-D case, since the
//! Lennard-Jones potential is spherically symmetric so B2 is a 1-D integral).
//!
//! Adaptive Simpson with local error control: each panel is compared against the
//! sum of its two halves; if they disagree by more than the tolerance the panel
//! is bisected. This is the same adaptive-refinement idea as hcubature, minus the
//! multi-dimensional region bookkeeping.

/// Adaptive Simpson integration of `f` over `[a, b]` to absolute tolerance `tol`.
pub fn adaptive_simpson<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64, tol: f64, max_depth: u32) -> f64 {
    let m = 0.5 * (a + b);
    let fa = f(a);
    let fb = f(b);
    let fm = f(m);
    let whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb);
    recur(f, a, b, fa, fb, fm, whole, tol, max_depth)
}

#[allow(clippy::too_many_arguments)]
fn recur<F: Fn(f64) -> f64>(
    f: &F,
    a: f64,
    b: f64,
    fa: f64,
    fb: f64,
    fm: f64,
    whole: f64,
    tol: f64,
    depth: u32,
) -> f64 {
    let m = 0.5 * (a + b);
    let lm = 0.5 * (a + m);
    let rm = 0.5 * (m + b);
    let flm = f(lm);
    let frm = f(rm);
    let left = (m - a) / 6.0 * (fa + 4.0 * flm + fm);
    let right = (b - m) / 6.0 * (fm + 4.0 * frm + fb);
    let delta = left + right - whole;
    if depth == 0 || delta.abs() <= 15.0 * tol {
        // Richardson extrapolation: the (left+right) Simpson estimate plus a
        // correction term that cancels the leading error.
        return left + right + delta / 15.0;
    }
    recur(f, a, m, fa, fm, flm, left, 0.5 * tol, depth - 1)
        + recur(f, m, b, fm, fb, frm, right, 0.5 * tol, depth - 1)
}

/// Simpson estimate of a vector integrand over `[a, b]` from its three samples.
#[inline]
fn simpson3(a: f64, b: f64, fa: [f64; 3], fm: [f64; 3], fb: [f64; 3]) -> [f64; 3] {
    let c = (b - a) / 6.0;
    [
        c * (fa[0] + 4.0 * fm[0] + fb[0]),
        c * (fa[1] + 4.0 * fm[1] + fb[1]),
        c * (fa[2] + 4.0 * fm[2] + fb[2]),
    ]
}

/// Adaptive Simpson for a vector-valued integrand `f: x -> [f64; 3]`, refining on a
/// SHARED subdivision until *every* component is resolved (mixed abs/rel test per
/// component). Because all three components are evaluated at the same nodes, their
/// quadrature errors are coherent — essential when the results are combined into a
/// ratio such as n_eff(B2, B2', B2'').
pub fn adaptive_simpson3<F: Fn(f64) -> [f64; 3]>(
    f: &F,
    a: f64,
    b: f64,
    tol: f64,
    max_depth: u32,
) -> [f64; 3] {
    let m = 0.5 * (a + b);
    let fa = f(a);
    let fb = f(b);
    let fm = f(m);
    let whole = simpson3(a, b, fa, fm, fb);
    recur3(f, a, b, fa, fb, fm, whole, tol, max_depth)
}

#[allow(clippy::too_many_arguments)]
fn recur3<F: Fn(f64) -> [f64; 3]>(
    f: &F,
    a: f64,
    b: f64,
    fa: [f64; 3],
    fb: [f64; 3],
    fm: [f64; 3],
    whole: [f64; 3],
    tol: f64,
    depth: u32,
) -> [f64; 3] {
    let m = 0.5 * (a + b);
    let lm = 0.5 * (a + m);
    let rm = 0.5 * (m + b);
    let flm = f(lm);
    let frm = f(rm);
    let left = simpson3(a, m, fa, flm, fm);
    let right = simpson3(m, b, fm, frm, fb);

    let mut out = [0.0f64; 3];
    let mut converged = true;
    for k in 0..3 {
        let lr = left[k] + right[k];
        let delta = lr - whole[k];
        out[k] = lr + delta / 15.0; // Richardson extrapolation, per component
        // mixed absolute/relative tolerance so each component is resolved on the
        // shared grid (the B2'' integrand peaks hardest at the wall and so votes
        // on where to refine).
        let scale = tol * (1.0 + lr.abs());
        if delta.abs() > 15.0 * scale {
            converged = false;
        }
    }
    if depth == 0 || converged {
        return out;
    }
    let l = recur3(f, a, m, fa, fm, flm, left, 0.5 * tol, depth - 1);
    let r = recur3(f, m, b, fm, fb, frm, right, 0.5 * tol, depth - 1);
    [l[0] + r[0], l[1] + r[1], l[2] + r[2]]
}

/// Fixed-grid composite Simpson over `[a, b]` with `n` panels (rounded up to even).
/// Used only as an *independent* high-resolution reference to check the adaptive
/// routine — not in the hot path.
pub fn composite_simpson<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64, n: usize) -> f64 {
    let n = if n % 2 == 1 { n + 1 } else { n };
    let h = (b - a) / n as f64;
    let mut s = f(a) + f(b);
    for i in 1..n {
        let x = a + i as f64 * h;
        s += if i % 2 == 1 { 4.0 } else { 2.0 } * f(x);
    }
    s * h / 3.0
}
