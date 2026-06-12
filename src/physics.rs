//! Virial coefficients for a spherically symmetric pair potential V(r).
//!
//! B2 (1-D):
//!   B2(T) = -2*pi * \int_0^\infty ( e^{-V/T} - 1 ) r^2 dr
//!
//! B3 (3-D), in triangle (bipolar) coordinates with the triangle-inequality
//! constraint on the inner variable:
//!   B3(T) = -(8*pi^2/3) \int_0^\infty dr1 \int_0^\infty dr2
//!                       \int_{|r1-r2|}^{r1+r2} dr3  r1 r2 r3 f1 f2 f3
//! where f_i = e^{-V(r_i)/T} - 1.
//!
//! The semi-infinite r1, r2 axes are mapped to [0,1) via r = s/(1-s) so the tail
//! is captured with no potential-specific truncation (keeps it generic across DSL
//! potentials). The integrators are generic over any `Fn(f64) -> f64` potential,
//! so the hand-rolled DSL, fasteval, a hard-coded closure, or a hard-sphere step
//! can all be plugged in and compared.

use crate::dsl::{self, Expr};
use crate::integrate::{adaptive_simpson, composite_simpson};

pub const PI: f64 = std::f64::consts::PI;

/// LJ Boyle temperature in reduced units (where B2 crosses zero).
pub const LJ_BOYLE_TSTAR: f64 = 3.417_928;

/// A compiled spherically symmetric pair potential V(r) (hand-rolled DSL backend).
/// Environment variable order is [r, eps, sig].
pub struct Potential {
    expr: Expr,
    eps: f64,
    sig: f64,
}

impl Potential {
    pub fn compile(src: &str, eps: f64, sig: f64) -> Result<Self, String> {
        let expr = dsl::compile(src, &["r", "eps", "sig"])?;
        Ok(Potential { expr, eps, sig })
    }

    /// Compile without the integer-power optimization (keeps `**` as `powf`).
    pub fn compile_unoptimized(src: &str, eps: f64, sig: f64) -> Result<Self, String> {
        let expr = dsl::compile_unoptimized(src, &["r", "eps", "sig"])?;
        Ok(Potential { expr, eps, sig })
    }

    #[inline]
    pub fn v(&self, r: f64) -> f64 {
        dsl::eval(&self.expr, &[r, self.eps, self.sig])
    }
}

/// A potential compiled to a CSE'd flat program (integer powers + common
/// subexpression elimination + no tree-walk dispatch). Holds a reusable scratch
/// buffer so the hot loop allocates nothing.
pub struct CsePotential {
    prog: dsl::Program,
    eps: f64,
    sig: f64,
    scratch: std::cell::RefCell<Vec<f64>>,
}

impl CsePotential {
    pub fn compile(src: &str, eps: f64, sig: f64) -> Result<Self, String> {
        let prog = dsl::compile_program(src, &["r", "eps", "sig"])?;
        let scratch = std::cell::RefCell::new(vec![0.0; dsl::program_slots(&prog)]);
        Ok(CsePotential {
            prog,
            eps,
            sig,
            scratch,
        })
    }

    /// Number of ops after CSE (smaller than the tree node count).
    pub fn ops(&self) -> usize {
        self.prog.len()
    }

    #[inline]
    pub fn v(&self, r: f64) -> f64 {
        let mut s = self.scratch.borrow_mut();
        dsl::eval_program(&self.prog, &[r, self.eps, self.sig], &mut s)
    }
}

/// Mayer factor f(r) = e^{-V/T} - 1, robust to the repulsive core where V
/// overflows to +inf / NaN (there f = -1).
#[inline]
fn mayer<V: Fn(f64) -> f64>(v: &V, t: f64, r: f64) -> f64 {
    let vv = v(r);
    if vv.is_finite() {
        (-vv / t).exp() - 1.0
    } else {
        -1.0
    }
}

// ------------------------------- B2 -------------------------------

#[inline]
fn b2_integrand_s<V: Fn(f64) -> f64>(v: &V, t: f64, s: f64) -> f64 {
    let om = 1.0 - s;
    if om <= 0.0 {
        return 0.0; // s = 1 -> r = inf
    }
    let r = s / om;
    let jac = 1.0 / (om * om);
    let val = mayer(v, t, r) * r * r * jac;
    if val.is_finite() {
        val
    } else {
        0.0
    }
}

/// B2 for any potential closure, via adaptive Simpson.
pub fn b2_v<V: Fn(f64) -> f64>(v: &V, t: f64, tol: f64) -> f64 {
    -2.0 * PI * adaptive_simpson(&|s| b2_integrand_s(v, t, s), 0.0, 1.0, tol, 60)
}

/// B2 via an independent fixed `n`-panel grid (reference for the adaptive routine).
pub fn b2_v_grid<V: Fn(f64) -> f64>(v: &V, t: f64, n: usize) -> f64 {
    -2.0 * PI * composite_simpson(&|s| b2_integrand_s(v, t, s), 0.0, 1.0, n)
}

pub fn b2(pot: &Potential, t: f64, tol: f64) -> f64 {
    b2_v(&|r| pot.v(r), t, tol)
}
pub fn b2_finegrid(pot: &Potential, t: f64, n: usize) -> f64 {
    b2_v_grid(&|r| pot.v(r), t, n)
}

// ------------------------------- B3 -------------------------------

/// B3 for any potential closure, via nested adaptive Simpson (3-D).
pub fn b3_v<V: Fn(f64) -> f64>(v: &V, t: f64, tol: f64) -> f64 {
    let f = |r: f64| mayer(v, t, r);

    let outer = |s1: f64| -> f64 {
        let om1 = 1.0 - s1;
        if om1 <= 0.0 {
            return 0.0;
        }
        let r1 = s1 / om1;
        let j1 = 1.0 / (om1 * om1);
        let f1 = f(r1);

        let mid = |s2: f64| -> f64 {
            let om2 = 1.0 - s2;
            if om2 <= 0.0 {
                return 0.0;
            }
            let r2 = s2 / om2;
            let j2 = 1.0 / (om2 * om2);
            let f2 = f(r2);
            // inner r3 integral over the triangle-inequality window
            let lo = (r1 - r2).abs();
            let hi = r1 + r2;
            let i3 = adaptive_simpson(&|r3| r3 * f(r3), lo, hi, tol, 28);
            let val = r2 * j2 * f2 * i3;
            if val.is_finite() {
                val
            } else {
                0.0
            }
        };

        let i2 = adaptive_simpson(&mid, 0.0, 1.0, tol, 28);
        let val = r1 * j1 * f1 * i2;
        if val.is_finite() {
            val
        } else {
            0.0
        }
    };

    -(8.0 * PI * PI / 3.0) * adaptive_simpson(&outer, 0.0, 1.0, tol, 28)
}

/// B3 via nested fixed `n`-panel grids (independent reference for B3).
pub fn b3_v_grid<V: Fn(f64) -> f64>(v: &V, t: f64, n: usize) -> f64 {
    let f = |r: f64| mayer(v, t, r);
    let outer = |s1: f64| -> f64 {
        let om1 = 1.0 - s1;
        if om1 <= 0.0 {
            return 0.0;
        }
        let r1 = s1 / om1;
        let j1 = 1.0 / (om1 * om1);
        let f1 = f(r1);
        let mid = |s2: f64| -> f64 {
            let om2 = 1.0 - s2;
            if om2 <= 0.0 {
                return 0.0;
            }
            let r2 = s2 / om2;
            let j2 = 1.0 / (om2 * om2);
            let f2 = f(r2);
            let lo = (r1 - r2).abs();
            let hi = r1 + r2;
            let i3 = composite_simpson(&|r3| r3 * f(r3), lo, hi, n);
            let val = r2 * j2 * f2 * i3;
            if val.is_finite() {
                val
            } else {
                0.0
            }
        };
        let i2 = composite_simpson(&mid, 0.0, 1.0, n);
        let val = r1 * j1 * f1 * i2;
        if val.is_finite() {
            val
        } else {
            0.0
        }
    };
    -(8.0 * PI * PI / 3.0) * composite_simpson(&outer, 0.0, 1.0, n)
}

pub fn b3(pot: &Potential, t: f64, tol: f64) -> f64 {
    b3_v(&|r| pot.v(r), t, tol)
}

/// B3 via genuine 3-D adaptive cubature (Genz-Malik hcubature). Same integral as
/// `b3_v`, but the triangle domain is mapped to the unit cube [0,1]^3 — r1,r2 via
/// r = s/(1-s), and r3 linearly across [|r1-r2|, r1+r2] — and integrated as one
/// 3-D region tree instead of nested 1-D sweeps. Returns (B3, integrand evals).
pub fn b3_cubature_v<V: Fn(f64) -> f64>(v: &V, t: f64, reltol: f64) -> (f64, usize) {
    let f = |r: f64| mayer(v, t, r);
    let integrand = |x: &[f64]| -> f64 {
        let (s1, s2, u3) = (x[0], x[1], x[2]);
        let (om1, om2) = (1.0 - s1, 1.0 - s2);
        if om1 <= 0.0 || om2 <= 0.0 {
            return 0.0;
        }
        let r1 = s1 / om1;
        let r2 = s2 / om2;
        let j1 = 1.0 / (om1 * om1);
        let j2 = 1.0 / (om2 * om2);
        let lo = (r1 - r2).abs();
        let hi = r1 + r2;
        let r3 = lo + (hi - lo) * u3;
        let jr3 = hi - lo;
        let val = r1 * r2 * r3 * f(r1) * f(r2) * f(r3) * j1 * j2 * jr3;
        if val.is_finite() {
            val
        } else {
            0.0
        }
    };
    let (i, _e, nev) = crate::cubature::hcubature(
        3,
        &integrand,
        &[0.0, 0.0, 0.0],
        &[1.0, 1.0, 1.0],
        1e-13,
        reltol,
        5_000_000,
    );
    (-(8.0 * PI * PI / 3.0) * i, nev)
}

pub fn b3_cubature(pot: &Potential, t: f64, reltol: f64) -> (f64, usize) {
    b3_cubature_v(&|r| pot.v(r), t, reltol)
}

// --------------------- closed-form LJ B2 reference ---------------------

/// Closed-form Lennard-Jones (12-6) second virial coefficient in reduced units
/// (sigma = epsilon = 1). Hirschfelder-Curtiss-Bird Gamma-function series:
///   B2(T*) = (2*pi/3) * sum_{j>=0} -( 2^{(2j+1)/2}/(4 j!) ) Gamma((2j-1)/4) T*^{-(2j+1)/4}
/// (`libm::tgamma` is used here only — not in any hot loop.)
pub fn b2_lj_series(tstar: f64, nterms: usize) -> f64 {
    let mut factorial = 1.0f64;
    let mut sum = 0.0f64;
    for j in 0..nterms {
        let jf = j as f64;
        if j > 0 {
            factorial *= jf;
        }
        let p = (2.0 * jf + 1.0) / 2.0;
        let coeff = -(2.0f64.powf(p) / (4.0 * factorial)) * libm::tgamma((2.0 * jf - 1.0) / 4.0);
        let term = coeff * tstar.powf(-(2.0 * jf + 1.0) / 4.0);
        if term.is_finite() {
            sum += term;
        }
    }
    (2.0 * PI / 3.0) * sum
}
