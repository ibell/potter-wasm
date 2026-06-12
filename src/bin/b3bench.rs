//! B3 method comparison: nested 1-D adaptive Simpson vs genuine Genz-Malik
//! hcubature, same hard-coded LJ potential. Also a smooth-integrand sanity check
//! to separate "the rule is efficient" from "this integrand is hard".

use potter_poc::b3_v;
use potter_poc::cubature::hcubature;
use std::time::Instant;

const C: f64 = -(8.0 * std::f64::consts::PI * std::f64::consts::PI) / 3.0;

fn lj(r: f64) -> f64 {
    let inv = 1.0 / r;
    let s2 = inv * inv;
    let s6 = s2 * s2 * s2;
    4.0 * (s6 * s6 - s6)
}

// B3 integrand on the unit cube (transform + Mayer factors), for a given V, T.
fn b3_integrand(t: f64) -> impl Fn(&[f64]) -> f64 {
    move |x: &[f64]| {
        let mayer = |r: f64| {
            let vv = lj(r);
            if vv.is_finite() {
                (-vv / t).exp() - 1.0
            } else {
                -1.0
            }
        };
        let (s1, s2, u3) = (x[0], x[1], x[2]);
        let (om1, om2) = (1.0 - s1, 1.0 - s2);
        if om1 <= 0.0 || om2 <= 0.0 {
            return 0.0;
        }
        let r1 = s1 / om1;
        let r2 = s2 / om2;
        let lo = (r1 - r2).abs();
        let hi = r1 + r2;
        let r3 = lo + (hi - lo) * u3;
        let val = r1 * r2 * r3 * mayer(r1) * mayer(r2) * mayer(r3)
            * (1.0 / (om1 * om1)) * (1.0 / (om2 * om2)) * (hi - lo);
        if val.is_finite() {
            val
        } else {
            0.0
        }
    }
}

fn timed<T>(reps: usize, mut f: impl FnMut() -> T) -> (T, f64) {
    let mut best = f64::INFINITY;
    let mut v = f();
    for _ in 0..reps {
        let t0 = Instant::now();
        v = f();
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        if ms < best {
            best = ms;
        }
    }
    (v, best)
}

fn main() {
    // --- sanity: a smooth integrand should converge in very few evals ---
    let g = |x: &[f64]| (x[0] + x[1] + x[2]).exp();
    let exact = (std::f64::consts::E - 1.0).powi(3);
    let (v, e, n) = hcubature(3, &g, &[0.0; 3], &[1.0; 3], 1e-12, 1e-10, 5_000_000);
    println!("sanity (smooth exp(x+y+z) over [0,1]^3):");
    println!("  hcubature = {v:.10}  exact = {exact:.10}  |err|={:.1e}  errEst={e:.1e}  evals={n}\n", (v - exact).abs());

    println!("B3 of Lennard-Jones (12-6): nested-1D adaptive vs Genz-Malik hcubature");
    for &t in &[1.0_f64, 1.5, 2.0] {
        let f = b3_integrand(t);
        // best-estimate reference (tight, large budget)
        let (ri, _re, _rn) = hcubature(3, &f, &[0.0; 3], &[1.0; 3], 1e-13, 1e-9, 40_000_000);
        let rref = C * ri;

        println!("\nT* = {t:.1}   (reference B3 = {rref:.8})");
        let (vn, tn) = timed(3, || b3_v(&lj, t, 1e-7));
        println!("  nested-1D  tol=1e-7 : err={:.1e}  [{tn:7.1} ms]", (vn - rref).abs());
        for &rel in &[1e-4, 1e-5, 1e-6] {
            let ((vc, nev), tc) = timed(3, || {
                let (i, _e, n) = hcubature(3, &f, &[0.0; 3], &[1.0; 3], 1e-13, rel, 5_000_000);
                (C * i, n)
            });
            let conv = if nev < 5_000_000 { "converged" } else { "hit cap" };
            println!(
                "  hcubature rel={rel:.0e}: err={:.1e}  [{tc:7.1} ms, {nev:>8} evals, {conv}]",
                (vc - rref).abs()
            );
        }
    }
}
