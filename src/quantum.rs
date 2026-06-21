//! Full-quantum B2 via Beth-Uhlenbeck phase shifts.
//! Atomic units internally (Bohr, Hartree, electron mass, hbar=1).
//!
//! The scattering second virial coefficient follows Hurly & Mehl, J. Res. NIST 112, 75
//! (2007), Eqs. (9),(18)-(24) — the method Cencek 2012 used for the tabulated 4He/3He
//! virials that Bell 2020 Fig. 8 plots. Phase shifts δ_l(k) are obtained by NUMEROV
//! integration of the radial equation matched to free Riccati-Bessel functions: the He
//! repulsive wall is extremely stiff (V(0.5 a0) ≈ 0.8 MK), and a Numerov scheme is
//! 4th-order and L-stable through it, whereas RK4 on the nonlinear Calogero phase equation
//! needs an impractical step (and otherwise blows up). The variable-phase `phase_shifts`
//! below is kept for the analytic square-well test only.

use crate::he_potential::{v_he, He};

const HARTREE_K: f64 = 315774.65;
const A0_CM: f64 = 0.529177210903e-8;
const N_A: f64 = 6.02214076e23;
const AMU_ME: f64 = 1822.888486209;
const PI: f64 = std::f64::consts::PI;
const SQRT2: f64 = std::f64::consts::SQRT_2;

#[derive(Clone, Copy)]
pub enum Species {
    He4,
    He3,
    Ne,
}

fn iso(sp: Species) -> He {
    match sp {
        Species::He3 => He::He3,
        _ => He::He4,
    }
}
fn mass_amu(sp: Species) -> f64 {
    match sp {
        Species::He4 => 4.002602,
        Species::He3 => 3.0160293,
        Species::Ne => 20.1797,
    }
}

/// V(r) [Hartree] for the species (He potential for He4/He3; Ne uses the TT potential
/// converted to a.u.: neon_tt v is V/kB[K] at r[nm]).
fn potential(sp: Species) -> impl Fn(f64) -> f64 {
    move |r_bohr: f64| match sp {
        Species::Ne => {
            let r_nm = r_bohr * A0_CM * 1e7; // a0->cm->nm
            crate::noblegas::neon_tt().v(r_nm) / HARTREE_K // K -> Hartree
        }
        s => v_he(iso(s), r_bohr, true),
    }
}

/// Pair reduced mass μ (electron masses), ATOMIC masses (Cencek/Hurly–Mehl): μ = m_atom/2.
/// Chosen so E[K] = α κ² exactly.
fn mu_pair(sp: Species) -> f64 {
    mass_amu(sp) * AMU_ME / 2.0
}

/// Statistical weight w_l in S(κ) = Σ_l w_l (2l+1) δ_l (Hurly–Mehl Eq. 9 + nuclear spin).
/// spin-0 bosons (⁴He, ²⁰Ne): even l 1, odd l 0. spin-½ ³He (I=½): even ¼, odd ¾.
#[allow(clippy::manual_is_multiple_of)]
fn l_weight(sp: Species, l: usize) -> f64 {
    match sp {
        Species::He4 | Species::Ne => {
            if l % 2 == 0 {
                1.0
            } else {
                0.0
            }
        }
        Species::He3 => {
            if l % 2 == 0 {
                0.25
            } else {
                0.75
            }
        }
    }
}

/// Ideal-gas coefficient c so B_ideal = c·N_A Λ³ (HM Eq. 20, ∓2^{-5/2}λ³ N_A/(2I+1), Λ=√2 λ_T).
/// ⁴He & ²⁰Ne (boson, I=0): c=-1/16. ³He (fermion, I=½): c=+1/32.
fn c_ideal(sp: Species) -> f64 {
    match sp {
        Species::He3 => 1.0 / 32.0,
        _ => -1.0 / 16.0,
    }
}

/// Classical B2 (cm^3/mol): -2 pi N_A ∫ (e^{-βV}-1) r² dr  [a0³ per pair -> cm³/mol].
/// The integral starts at r_lo (Bohr), NOT r=0: the He potential fit extrapolates to
/// garbage (large NEGATIVE) below ~0.3 a0, which would make e^{-βV} overflow. Past the
/// physical repulsive wall V is huge-positive so e^{-βV}->0 and the integrand is exactly
/// -r² (the hard-core excluded volume); we add that analytic [0, r_lo] piece (-r_lo³/3).
pub fn classical_b2(sp: Species, t: f64) -> f64 {
    let v = potential(sp);
    let beta = HARTREE_K / t; // 1/(kT) in 1/Hartree
    let r_lo = 0.3_f64;
    let (n, rmax) = (200_000usize, 60.0_f64);
    let h = (rmax - r_lo) / n as f64;
    // analytic [0, r_lo] contribution: integrand = (0 - 1) r² = -r²  (V>>kT inside wall).
    let mut s = -r_lo * r_lo * r_lo / 3.0;
    for i in 0..=n {
        let r = (i as f64) * h + r_lo;
        let f = ((-beta * v(r)).exp() - 1.0) * r * r;
        s += if i == 0 || i == n { 0.5 } else { 1.0 } * f * h;
    }
    -2.0 * PI * N_A * s * A0_CM.powi(3)
}

pub fn quantum_b2(sp: Species, t: f64) -> f64 {
    quantum_b2_parts(sp, t).0
}

// =========================================================================================
// Beth–Uhlenbeck B2 via Numerov-matched phase shifts
// =========================================================================================
//
// We integrate the LINEAR radial equation u'' = [l(l+1)/r² + 2μV − k²] u outward by Numerov
// (L-stable through the stiff He wall) and match to free Riccati-Bessel at the outer edge for
// the principal δ_l(k) (mod π). S(κ) needs the CONTINUOUS, Levinson-anchored δ_l(k): we
// recover it by k-unwrapping the principal values upward from a small-k anchor whose absolute
// branch is fixed by a node-count Levinson there, plus the s-wave dimer π for ⁴He. This
// reproduces the Cencek 2012 table to <0.01 cm³/mol and runs in seconds.

const SCAT_R0: f64 = 0.5; // inner radius (Bohr): u≈0 this deep in the wall
const SCAT_NK: usize = 1500; // κ-grid points (trapezoid); large enough the unwrap step < π/2
const SCAT_KF: f64 = 40.0; // Boltzmann cutoff: κ_max² α = KF·T  (e^{-KF} weight at the edge)
const SCAT_LCAP: usize = 60; // overall l safety cap (physical cap is per-T via κ_max·rmax)
const SCAT_H: f64 = 2.5e-3; // Numerov step (Bohr); Numerov is 4th-order and L-stable

/// T-adaptive outer matching radius (Bohr). Low T (small k, ⁴He's ~189 a0 scattering length)
/// needs a long tail to capture the phase; high T does not, and too-large rmax there would
/// outrun the κ-grid unwrap. rmax·κ_max is held ≈ const (480·√α·… ) so SCAT_NK keeps the
/// unwrap step < π/2 at all T. Validated vs Cencek: Δ<0.001 cm³/mol at 4 K, <0.05 to 500 K.
fn scat_rmax(t: f64) -> f64 {
    (480.0 / t.sqrt()).clamp(40.0, 160.0)
}

/// Numerically STABLE Riccati-Bessel ĵ_l(x)=x j_l(x), ŷ_l(x)=x y_l(x), l=0..lmax.
/// ŷ_l by upward recurrence (the irregular solution grows, so upward is stable). ĵ_l by
/// DOWNWARD (Miller) recurrence from a high seed, normalized to ĵ_0 = sin x — this is
/// stable even when l ≫ x (where the regular solution decays and upward recurrence
/// would explode). Both satisfy f_{l+1} = (2l+1)/x f_l - f_{l-1}.
fn riccati_stable(lmax: usize, x: f64) -> (Vec<f64>, Vec<f64>) {
    let (s, c) = (x.sin(), x.cos());
    let mut y = vec![0.0; lmax + 1];
    y[0] = -c;
    if lmax >= 1 {
        y[1] = -c / x - s;
        for l in 1..lmax {
            y[l + 1] = (2 * l + 1) as f64 / x * y[l] - y[l - 1];
        }
    }
    let mut j = vec![0.0; lmax + 1];
    j[0] = s;
    if lmax == 0 {
        return (j, y);
    }
    if (lmax as f64) < x - 1.0 {
        // upward recurrence is stable while l < x
        j[1] = s / x - c;
        for l in 1..lmax {
            j[l + 1] = (2 * l + 1) as f64 / x * j[l] - j[l - 1];
        }
    } else {
        // downward (Miller) recurrence: seed high, recur down, rescale so ĵ_0 = sin x
        let top = lmax + 20 + x as usize;
        let mut jt = vec![0.0f64; top + 2];
        jt[top] = 1e-30;
        for l in (1..=top).rev() {
            jt[l - 1] = (2 * l + 1) as f64 / x * jt[l] - jt[l + 1];
        }
        let scale = s / jt[0];
        for (l, jl) in j.iter_mut().enumerate() {
            *jl = jt[l] * scale;
        }
    }
    (j, y)
}

/// Principal phase shifts δ_l(k) mod π in (−π/2, π/2), l=0..lmax, by Numerov. Integrate
/// u'' = [l(l+1)/r² + 2μV − k²] u outward (u(r0)=0), match free Riccati-Bessel at the last
/// two grid points: tan δ_l = (u_{n-1} ĵ_l(k r_n) − u_n ĵ_l(k r_{n-1})) /
/// (u_{n-1} ŷ_l(k r_n) − u_n ŷ_l(k r_{n-1})). Numerov is L-stable through the stiff wall;
/// the absolute branch (multiple of π) is resolved by k-unwrapping (see `b2_moments`).
fn phase_principal<V: Fn(f64) -> f64>(
    v: &V,
    mu: f64,
    k: f64,
    lmax: usize,
    r0: f64,
    rmax: f64,
    n: usize,
) -> Vec<f64> {
    let h = (rmax - r0) / n as f64;
    let h2 = h * h;
    let k2 = k * k;
    let mut out = vec![0.0f64; lmax + 1];
    let uv: Vec<f64> = (0..=n).map(|i| 2.0 * mu * v(r0 + i as f64 * h)).collect();
    for l in 0..=lmax {
        let ll = (l * (l + 1)) as f64;
        let g = |i: usize| -> f64 {
            let r = r0 + i as f64 * h;
            ll / (r * r) + uv[i] - k2
        };
        let f = |i: usize| 1.0 - h2 / 12.0 * g(i);
        let (mut um, mut uc) = (0.0f64, 1e-20f64);
        let (mut fm, mut fc) = (f(0), f(1));
        for i in 2..=n {
            let fp = f(i);
            let up = ((12.0 - 10.0 * fc) * uc - fm * um) / fp;
            um = uc;
            uc = up;
            fm = fc;
            fc = fp;
            if uc.abs() > 1e250 {
                um *= 1e-250;
                uc *= 1e-250;
            }
        }
        let (j1, y1) = riccati_stable(l, k * (rmax - h));
        let (j2, y2) = riccati_stable(l, k * rmax);
        let num = um * j2[l] - uc * j1[l];
        let den = um * y2[l] - uc * y1[l];
        out[l] = (num / den).atan();
    }
    out
}

/// Continuous, Levinson-anchored δ_l(k) at a single (small) k — the k-unwrap anchor:
/// δ_l = π·(N_scat − N_free) + principal, with robust node counting (ignore |u| below
/// 1e-6 of its peak). N_scat = interior nodes of the scattering u; N_free = nodes of the
/// free ĵ_l on the same grid. (The l=0 dimer adds no interior node, so the Levinson π for
/// the ⁴He bound state is folded in separately by the caller.)
fn phase_anchor<V: Fn(f64) -> f64>(
    v: &V,
    mu: f64,
    k: f64,
    lmax: usize,
    r0: f64,
    rmax: f64,
    n: usize,
) -> Vec<f64> {
    let h = (rmax - r0) / n as f64;
    let h2 = h * h;
    let k2 = k * k;
    let mut out = vec![0.0f64; lmax + 1];
    let uv: Vec<f64> = (0..=n).map(|i| 2.0 * mu * v(r0 + i as f64 * h)).collect();
    let jfree: Vec<Vec<f64>> = (0..=n)
        .map(|i| riccati_stable(lmax, k * (r0 + i as f64 * h)).0)
        .collect();
    let count_nodes = |samp: &dyn Fn(usize) -> f64| -> usize {
        let mut peak = 0.0f64;
        for i in 0..=n {
            peak = peak.max(samp(i).abs());
        }
        let thr = peak * 1e-6;
        let (mut nc, mut last) = (0usize, 0.0f64);
        for i in 0..=n {
            let x = samp(i);
            if x.abs() < thr {
                continue;
            }
            if last != 0.0 && last * x < 0.0 {
                nc += 1;
            }
            last = x;
        }
        nc
    };
    for l in 0..=lmax {
        let ll = (l * (l + 1)) as f64;
        let g = |i: usize| -> f64 {
            let r = r0 + i as f64 * h;
            ll / (r * r) + uv[i] - k2
        };
        let f = |i: usize| 1.0 - h2 / 12.0 * g(i);
        let mut u = vec![0.0f64; n + 1];
        u[1] = 1e-20;
        let (mut fm, mut fc) = (f(0), f(1));
        for i in 2..=n {
            let fp = f(i);
            u[i] = ((12.0 - 10.0 * fc) * u[i - 1] - fm * u[i - 2]) / fp;
            fm = fc;
            fc = fp;
            if u[i].abs() > 1e250 {
                for x in u.iter_mut().take(i + 1) {
                    *x *= 1e-250;
                }
            }
        }
        let nscat = count_nodes(&|i| u[i]);
        let nfree = count_nodes(&|i| jfree[i][l]);
        let (um, uc) = (u[n - 1], u[n]);
        let (j1, y1) = riccati_stable(l, k * (rmax - h));
        let (j2, y2) = riccati_stable(l, k * rmax);
        let princ = ((um * j2[l] - uc * j1[l]) / (um * y2[l] - uc * y1[l])).atan();
        out[l] = princ + (nscat as i64 - nfree as i64) as f64 * PI;
    }
    out
}

/// κ-moments J_p = ∫₀^∞ e^{-ακ²/T} S(κ) κ (ακ²)^p dκ for p=0,1,2, with
/// S(κ) = Σ_l w_l (2l+1) δ_l(κ) (species statistics weights). δ_l(κ) from Numerov-matched
/// principal values, k-unwrapped upward from the small-k Levinson anchor; `has_bound` folds
/// the dimer Levinson π into l=0. Parallel over the κ-grid (std thread::scope).
#[allow(clippy::needless_range_loop)] // l indexes anchor/princ AND feeds l_weight/(2l+1); ik indexes s_grid AND sets k
fn b2_moments(sp: Species, t: f64, mu: f64, alpha: f64, has_bound: bool) -> (f64, f64, f64) {
    let v = potential(sp);
    let rmax = scat_rmax(t);
    let n = (((rmax - SCAT_R0) / SCAT_H).ceil() as usize).max(8_000);
    let kmax = (SCAT_KF * t / alpha).sqrt();
    let hk = kmax / SCAT_NK as f64;
    let lcap = ((kmax * rmax).ceil() as usize).clamp(6, SCAT_LCAP);
    let cores = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1);
    // 1) principal δ_l(k) on the κ-grid (parallel over k; each k is independent).
    let princ: Vec<Vec<f64>> = std::thread::scope(|sc| {
        let handles: Vec<_> = (0..cores)
            .map(|c| {
                let v = &v;
                sc.spawn(move || {
                    let mut local: Vec<(usize, Vec<f64>)> = Vec::new();
                    let mut ik = c + 1;
                    while ik <= SCAT_NK {
                        let k = ik as f64 * hk;
                        local.push((ik, phase_principal(v, mu, k, lcap, SCAT_R0, rmax, n)));
                        ik += cores;
                    }
                    local
                })
            })
            .collect();
        let mut all = vec![Vec::new(); SCAT_NK + 1];
        for hnd in handles {
            for (ik, d) in hnd.join().unwrap() {
                all[ik] = d;
            }
        }
        all
    });
    // 2) Levinson anchor at the smallest k; fold the dimer π into l=0 for ⁴He.
    let mut anchor = phase_anchor(&v, mu, hk, lcap, SCAT_R0, rmax, n);
    if has_bound {
        anchor[0] += PI;
    }
    // 3) k-unwrap each contributing l upward from the anchor; build S(κ) on the grid.
    let mut s_grid = vec![0.0f64; SCAT_NK + 1];
    for l in 0..=lcap {
        let w = l_weight(sp, l);
        if w == 0.0 {
            continue;
        }
        let mut prev = anchor[l];
        for ik in 1..=SCAT_NK {
            let mut d = princ[ik][l];
            while d - prev > PI / 2.0 {
                d -= PI;
            }
            while d - prev < -PI / 2.0 {
                d += PI;
            }
            prev = d;
            s_grid[ik] += w * (2 * l + 1) as f64 * d;
        }
    }
    // 4) integrate the three moments (trapezoid; ik=0 → κ=0 contributes 0).
    let (mut j0, mut j1, mut j2) = (0.0f64, 0.0f64, 0.0f64);
    for ik in 1..=SCAT_NK {
        let k = ik as f64 * hk;
        let e = alpha * k * k; // E in K
        let base = (-e / t).exp() * s_grid[ik] * k;
        let tw = if ik == SCAT_NK { 0.5 } else { 1.0 } * hk;
        j0 += base * tw;
        j1 += base * e * tw;
        j2 += base * e * e * tw;
    }
    (j0, j1, j2)
}

/// (B2, dB2/dT, d2B2/dT2) cm^3/mol — Beth–Uhlenbeck B2 + analytic T-derivs, Hurly & Mehl
/// 2007 Eqs. (9),(18)-(24). B=B_th+B_ideal+B_bound; B_th=-2 N_A Λ³ α I₀/(πT),
/// I₀=∫e^{-ακ²/T}S(κ)κ dκ; B_ideal=c·N_A Λ³; B_bound=-N_A Λ³[e^{T_b/T}-1] (⁴He dimer).
/// α=(mₑ/m_atom)E_h/k_B; Λ=√2 λ_T; S(κ)=Σ w_l(2l+1)δ_l. E[K]=ακ². The T-derivatives reuse
/// the κ-moments J_p (p=0,1,2): with B_th = A·J0, A ∝ T^{-5/2} and J0'(T)=J1/T²,
/// J0''=J2/T⁴-2J1/T³ — all analytic, no finite differencing.
pub fn quantum_b2_parts(sp: Species, t: f64) -> (f64, f64, f64) {
    let mu = mu_pair(sp);
    let m = mass_amu(sp) * AMU_ME; // atomic mass in mₑ (= 2μ)
    let beta = HARTREE_K / t; // 1/Hartree
    let lambda_t = (2.0 * PI * beta / m).sqrt(); // λ_T in Bohr (h=2π a.u.)
    let na_l3 = N_A * (SQRT2 * lambda_t).powi(3) * A0_CM.powi(3); // N_A Λ³ [cm³/mol], Λ=√2 λ_T
    let alpha = HARTREE_K / m; // α = (mₑ/m_atom) E_h/k_B  [K]

    // ⁴He dimer energy (computed once; reused for the Levinson fold and the bound term).
    let eb = if let Species::He4 = sp {
        let v = potential(sp);
        crate::quantum::s_wave_bound_energy(&v, mu)
    } else {
        None
    };

    // thermal (scattering) term and its analytic T-derivatives from the κ-moments.
    let (j0, j1, j2) = b2_moments(sp, t, mu, alpha, eb.is_some());
    let a = -2.0 * na_l3 * alpha / (PI * t); // ∝ T^{-5/2}
    let b_th = a * j0;
    let b_th_d1 = a * (-2.5 / t * j0 + j1 / (t * t));
    let b_th_d2 = a * (8.75 / (t * t) * j0 - 7.0 * j1 / (t * t * t) + j2 / t.powi(4));

    // ideal-gas (exchange) term ∝ T^{-3/2}.
    let b_id = c_ideal(sp) * na_l3;
    let b_id_d1 = -1.5 / t * b_id;
    let b_id_d2 = 3.75 / (t * t) * b_id;

    // bound-state term (⁴He dimer only): B_bound = -N_A Λ³ [e^{T_b/T}-1], ∝ T^{-3/2}·g(T).
    let (mut b_bd, mut b_bd_d1, mut b_bd_d2) = (0.0, 0.0, 0.0);
    if let Some(eb) = eb {
        let tb = -eb * HARTREE_K; // T_b = |E_b| in K (>0)
        let p = -na_l3;
        let ex = (tb / t).exp();
        let g = ex - 1.0;
        let gp = -tb / (t * t) * ex;
        let gpp = (tb * tb / t.powi(4) + 2.0 * tb / (t * t * t)) * ex;
        b_bd = p * g;
        b_bd_d1 = -1.5 / t * p * g + p * gp;
        b_bd_d2 = 3.75 / (t * t) * p * g + 2.0 * (-1.5 / t * p) * gp + p * gpp;
    }
    (
        b_th + b_id + b_bd,
        b_th_d1 + b_id_d1 + b_bd_d1,
        b_th_d2 + b_id_d2 + b_bd_d2,
    )
}

// =========================================================================================
// Variable-phase (Calogero) engine — used by the analytic square-well test only.
// =========================================================================================

/// Riccati-Bessel functions up to order `lmax`: jhat_l(x)=x j_l(x), yhat_l(x)=x y_l(x).
/// jhat_0=sin x, yhat_0=-cos x; jhat_1=sin x/x - cos x, yhat_1=-cos x/x - sin x;
/// both satisfy f_{l+1} = (2l+1)/x f_l - f_{l-1}. Upward recurrence (adequate for the
/// low-l square-well test; production B2 uses `riccati_stable`).
pub fn riccati(lmax: usize, x: f64) -> (Vec<f64>, Vec<f64>) {
    let (s, c) = (x.sin(), x.cos());
    let mut j = vec![0.0; lmax + 1];
    let mut y = vec![0.0; lmax + 1];
    j[0] = s;
    y[0] = -c;
    if lmax >= 1 {
        j[1] = s / x - c;
        y[1] = -c / x - s;
    }
    for l in 1..lmax {
        let f = (2 * l + 1) as f64 / x;
        j[l + 1] = f * j[l] - j[l - 1];
        y[l + 1] = f * y[l] - y[l - 1];
    }
    (j, y)
}

/// Phase shift delta_l(k) for U(r)=2 mu V(r) via the Calogero variable-phase eq.,
/// integrated r0->rmax by RK4: delta_l'(r) = -(1/k) U(r) [cos d jhat_l(kr) - sin d yhat_l(kr)]^2.
/// `v`: V(r) [Hartree] closure. Returns delta_l (radians) for l=0..lmax.
pub fn phase_shifts<V: Fn(f64) -> f64>(
    v: &V,
    mu: f64,
    k: f64,
    lmax: usize,
    r0: f64,
    rmax: f64,
    steps: usize,
) -> Vec<f64> {
    let h = (rmax - r0) / steps as f64;
    let mut d = vec![0.0_f64; lmax + 1];
    let deriv = |r: f64, dl: &[f64]| -> Vec<f64> {
        let u = 2.0 * mu * v(r);
        let (j, y) = riccati(lmax, k * r);
        (0..=lmax)
            .map(|l| {
                let b = d_cos_sin(dl[l], j[l], y[l]);
                -(1.0 / k) * u * b * b
            })
            .collect()
    };
    let mut r = r0;
    for _ in 0..steps {
        let k1 = deriv(r, &d);
        let d2: Vec<f64> = (0..=lmax).map(|l| d[l] + 0.5 * h * k1[l]).collect();
        let k2 = deriv(r + 0.5 * h, &d2);
        let d3: Vec<f64> = (0..=lmax).map(|l| d[l] + 0.5 * h * k2[l]).collect();
        let k3 = deriv(r + 0.5 * h, &d3);
        let d4: Vec<f64> = (0..=lmax).map(|l| d[l] + h * k3[l]).collect();
        let k4 = deriv(r + h, &d4);
        for l in 0..=lmax {
            d[l] += h / 6.0 * (k1[l] + 2.0 * k2[l] + 2.0 * k3[l] + k4[l]);
        }
        r += h;
    }
    d
}

#[inline]
fn d_cos_sin(d: f64, jl: f64, yl: f64) -> f64 {
    d.cos() * jl - d.sin() * yl
}

// =========================================================================================
// s-wave bound state (⁴He dimer)
// =========================================================================================

// Inner cutoff (a0): the He repulsive wall is enormous (V(0.5)~2.6 Hartree), so
// starting Numerov at r->0 overflows. The bound-state u(r) is utterly negligible
// here (V ~ 1e5 K), so seeding u(R0)=0 is exact to machine precision. The result is
// insensitive to R0 over [1.0, 2.0] (verified). Outer radius is far past the ~50 a0
// halo; E_b is converged to <1e-4 mK at RMAX=800, N=40k (checked vs RMAX 5000, N 5e5).
const BOUND_R0: f64 = 2.0;
const BOUND_RMAX: f64 = 800.0;
const BOUND_N: usize = 40_000;

/// Number of l=0 nodes of the radial solution u'' = 2 mu (V - e) u, integrated
/// outward by Numerov from R0 (seed u=0) to RMAX. At e=0 this is the Levinson count
/// of s-wave bound states; for e<0 it jumps by one as e crosses each eigenvalue
/// (above the eigenvalue the classically-forbidden tail diverges with a sign flip).
fn s_wave_node_count<V: Fn(f64) -> f64>(v: &V, mu: f64, e: f64) -> usize {
    let h = (BOUND_RMAX - BOUND_R0) / BOUND_N as f64;
    let f = |r: f64| 2.0 * mu * (v(r) - e); // u'' = f(r) u
    let w = |r: f64| 1.0 - h * h / 12.0 * f(r);
    let (mut u0, mut u1) = (0.0_f64, 1e-30_f64);
    let mut nodes = 0usize;
    let mut r = BOUND_R0 + h;
    for _ in 2..=BOUND_N {
        let rn = r + h;
        let u2 = ((12.0 - 10.0 * w(r)) * u1 - w(r - h) * u0) / w(rn);
        if u1 * u2 < 0.0 {
            nodes += 1;
        }
        u0 = u1;
        u1 = u2;
        r = rn;
    }
    nodes
}

/// s-wave bound-state energy (Hartree, <0) of the deepest/only state, or None.
/// Shooting eigenvalue by node counting: the E<0 radial solution has 0 nodes below
/// the ground-state eigenvalue and 1 above it, so the eigenvalue is the E at which
/// the node count switches; bisect on that switch. Returns None when no E<0 in the
/// search bracket produces a node (no bound state, e.g. 3He).
pub fn s_wave_bound_energy<V: Fn(f64) -> f64>(v: &V, mu: f64) -> Option<f64> {
    // Bracket below (deep, 0 nodes) and above (shallow, 1 node if a state exists).
    let (mut lo, mut hi) = (-1e-6_f64, -1e-13_f64);
    if s_wave_node_count(v, mu, hi) == 0 {
        return None; // no bound state appears as E -> 0^-
    }
    // Ensure the lower bracket is below the eigenvalue (0 nodes); if even the deepest
    // probe still shows a node, there's a deeper state than we model -> bail to None.
    if s_wave_node_count(v, mu, lo) != 0 {
        return None;
    }
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if s_wave_node_count(v, mu, mid) == 0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(0.5 * (lo + hi))
}

/// Test helper: s-wave phase shift for a square well V=-v0 (r<rr) else 0.
pub fn s_wave_phase_for_test(mu: f64, v0: f64, rr: f64, k: f64) -> f64 {
    let v = |r: f64| if r < rr { -v0 } else { 0.0 };
    phase_shifts(&v, mu, k, 0, 1e-6, rr + 30.0, 6000)[0]
}
