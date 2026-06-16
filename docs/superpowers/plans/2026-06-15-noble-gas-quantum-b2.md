# Noble-gas B₂ with Wigner–Kirkwood quantum corrections — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A native Rust module computing classical + Wigner–Kirkwood-corrected (to 3rd order) second virial coefficients and `n_eff` for the Tang–Toennies noble-gas potentials (Ne, Ar, Kr, Xe), in cm³/mol, reproducing `integrate_potentials.py`.

**Architecture:** A `TangToennies` struct whose potential value function is generic over `num_dual::DualNum<f64>`; spatial derivatives `V′/V″/V‴` come from a `Dual3` third-derivative, and the B₂ temperature derivatives (for `n_eff`) from a `Dual2` second-derivative threaded through a fixed-log-grid integral. No hand-coded derivatives.

**Tech Stack:** Rust, new dependency `num-dual = "0.14"`. New file `src/noblegas.rs`. Tests in `src/main.rs`'s `#[cfg(test)] mod tests`.

**Reference spec:** `docs/superpowers/specs/2026-06-15-noble-gas-quantum-b2-design.md`

**Reference values (computed from `integrate_potentials.py`'s analytic derivatives — the test targets):**

Potential V/k_B [K] (TT value at the given R in nm):
| gas | anchor 1 | anchor 2 |
|---|---|---|
| Ne | `v(0.16)=26860.903` | `v(0.56)=−1.632` |
| Ar | `v(0.20)=51376.994` | `v(0.9)=−0.9169` |
| Kr | `v(0.24)=27869.811` | `v(1.00)=−0.9816` |
| Xe | `v(0.26)=37582.271` | `v(0.9)=−4.3452` |

Argon analytic derivatives at R=0.5 nm (K/nmᵏ): `V=−38.818381, V′=492.757085, V″=−6719.247296, V‴=83513.124167`.

B₂ [cm³/mol] (classical / WK-order-3):
| gas | T=50 | T=100 | T=300 | T=1000 |
|---|---|---|---|---|
| Ne cl | −38.5466 | −4.9747 | 11.4662 | 13.8749 |
| Ne WK3 | −36.4805 | −4.4326 | 11.5664 | 13.8950 |
| Ar cl | −774.1828 | −183.8396 | −15.2992 | 20.2995 |
| Ar WK3 | −756.9259 | −182.3749 | −15.1793 | 20.3171 |
| Kr cl | −2507.8925 | −427.6767 | −50.2435 | 18.5870 |
| Kr WK3 | −2473.9746 | −426.1043 | −50.1607 | 18.5973 |
| Xe cl | −12473.4207 | −1146.4559 | −128.5996 | 12.2082 |
| Xe WK3 | −12335.0404 | −1143.5702 | −128.5143 | 12.2166 |

**Conventions:** tests live in `src/main.rs` `#[cfg(test)] mod tests` and import from `potter_poc::…`. Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## Task 1: Add `num-dual` and pin its derivative API

**Files:** Modify `Cargo.toml`; Test `src/main.rs`.

This establishes the exact `num-dual` 0.14 idioms every later task uses (`DualNum::from`, `.exp()`, `.powi()`, `.recip()`, arithmetic with `f64`, and the `first/second/third_derivative` helpers + their tuple order).

- [ ] **Step 1: Write the failing test** (in `src/main.rs` `mod tests`)

```rust
    #[test]
    fn num_dual_derivative_api() {
        use num_dual::{second_derivative, third_derivative, DualNum};
        // generic f(x) = x^3 - 2x, using the idioms the noble-gas code needs
        fn f<D: DualNum<f64> + Copy>(x: D) -> D { x.powi(3) - x * 2.0 }
        // third_derivative -> (value, f', f'', f''')
        let (v, d1, d2, d3) = third_derivative(f, 2.0);
        assert!((v - 4.0).abs() < 1e-12);   // 8 - 4
        assert!((d1 - 10.0).abs() < 1e-12);  // 3x^2 - 2 = 10
        assert!((d2 - 12.0).abs() < 1e-12);  // 6x
        assert!((d3 - 6.0).abs() < 1e-12);   // 6
        // second_derivative -> (value, f', f'')
        let (v2, e1, e2) = second_derivative(f, 2.0);
        assert!((v2 - 4.0).abs() < 1e-12 && (e1 - 10.0).abs() < 1e-12 && (e2 - 12.0).abs() < 1e-12);
        // exp/recip/from on a dual via first_derivative of g(x)=exp(-x)/x at x=1
        use num_dual::first_derivative;
        let (g, _) = first_derivative(|x: num_dual::Dual64| (-x).exp() * x.recip(), 1.0);
        assert!((g - std::f64::consts::E.recip()).abs() < 1e-12); // e^{-1}/1
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test num_dual_derivative_api`
Expected: FAIL — `num_dual` crate not found (unresolved import).

- [ ] **Step 3: Add the dependency**

In `Cargo.toml`, under `[dependencies]`, add:
```toml
num-dual = "0.14"
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test num_dual_derivative_api`
Expected: PASS. If a helper name or tuple arity differs in the installed 0.14.x, adjust the test to the actual API (e.g. `num_dual::third_derivative` signature) and re-run — this task exists precisely to discover and pin those idioms for later tasks. Report the exact working idioms in your summary.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock src/main.rs
git commit -m "$(cat <<'EOF'
Add num-dual dependency; pin its derivative API

Smoke test for second/third_derivative tuple order and DualNum idioms
(exp, powi, recip, f64 arithmetic) used by the noble-gas module.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `TangToennies` struct, potential value, and constructors

**Files:** Create `src/noblegas.rs`; Modify `src/lib.rs`; Test `src/main.rs`.

- [ ] **Step 1: Write the failing test** (in `src/main.rs` `mod tests`)

```rust
    #[test]
    fn noblegas_potential_anchors() {
        use potter_poc::noblegas::{neon_tt, argon_tt, krypton_tt, xenon_tt};
        // V/k_B [K] at R [nm] — TT values matching integrate_potentials.py
        assert!((neon_tt().v(0.16) - 26860.903).abs() / 26860.903 < 1e-4);
        assert!((neon_tt().v(0.56) - (-1.632)).abs() < 2e-3);
        assert!((argon_tt().v(0.20) - 51376.994).abs() / 51376.994 < 1e-4);
        assert!((argon_tt().v(0.9) - (-0.9169)).abs() < 2e-3);
        assert!((krypton_tt().v(0.24) - 27869.811).abs() / 27869.811 < 1e-4);
        assert!((krypton_tt().v(1.00) - (-0.9816)).abs() < 2e-3);
        assert!((xenon_tt().v(0.26) - 37582.271).abs() / 37582.271 < 1e-4);
        assert!((xenon_tt().v(0.9) - (-4.3452)).abs() < 2e-3);
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test noblegas_potential_anchors`
Expected: FAIL — `potter_poc::noblegas` does not exist.

- [ ] **Step 3: Create `src/noblegas.rs`**

```rust
//! Noble-gas (Ne, Ar, Kr, Xe) Tang-Toennies pair potentials and their second
//! virial coefficients with Wigner-Kirkwood quantum corrections (to 3rd order),
//! reproducing potter's `integrate_potentials.py`. Real units: R in nm, V/k_B in
//! K, B2 in cm^3/mol. Spatial and temperature derivatives use `num-dual` autodiff.

use num_dual::DualNum;

// physical constants (SI), matching integrate_potentials.py
const KB: f64 = 1.380649e-23; // J/K
const HBAR: f64 = 1.054571817e-34; // J s
const U_AMU: f64 = 1.66053906660e-27; // kg
const N_A: f64 = 8.314462618 / KB; // 1/mol  (= 6.02214076e23)
const PI: f64 = std::f64::consts::PI;

/// A Tang-Toennies noble-gas potential. `V/k_B = A·exp(a1 R + a2 R² + an1/R + an2/R²)
/// − Σ_{n=3}^{8} C[2n]·[1 − e^{−bR}Σ_{k=0}^{2n}(bR)^k/k!]/R^{2n}`, R in nm. Below
/// `rcutoff·repsilon` a short-range `tilde_amp/R·e^{−tilde_exp·R}` form is used.
pub struct TangToennies {
    pub a: f64,
    pub a1: f64,
    pub a2: f64,
    pub an1: f64,
    pub an2: f64,
    pub b: f64,
    pub nmax: usize,
    pub c: [f64; 6], // C[6], C[8], C[10], C[12], C[14], C[16] — index n-3
    pub tilde_amp: f64,
    pub tilde_exp: f64,
    pub rcutoff: f64,
    pub repsilon: f64,
    pub mass_rel: f64,
}

impl TangToennies {
    /// Fill C[12], C[14], C[16] from C[6..10] by the recurrence
    /// C[2n] = C[2n-6]·(C[2n-2]/C[2n-4])³ for n = 6, 7, 8.
    fn add_recursive(&mut self) {
        for n in 6..=8 {
            let i = n - 3;
            self.c[i] = self.c[i - 3] * (self.c[i - 1] / self.c[i - 2]).powi(3);
        }
    }

    /// V/k_B [K] generic over the dual scalar (used for both value and derivatives).
    pub fn v_full<D: DualNum<f64> + Copy>(&self, r: D) -> D {
        if r.re() < self.rcutoff * self.repsilon {
            // short-range: tilde_amp/r · exp(-tilde_exp·r)
            r.recip() * self.tilde_amp * (r * (-self.tilde_exp)).exp()
        } else {
            let ri = r.recip();
            let mut out = (r * self.a1 + r * r * self.a2 + ri * self.an1 + ri * ri * self.an2)
                .exp()
                * self.a;
            for n in 3..=self.nmax {
                let br = r * self.b;
                // sum_{k=0}^{2n} br^k / k!
                let mut s = D::from(1.0); // k = 0
                let mut term = D::from(1.0);
                let mut fact = 1.0;
                for k in 1..=(2 * n) {
                    term = term * br;
                    fact *= k as f64;
                    s = s + term * (1.0 / fact);
                }
                let bracket = -((-br).exp() * s) + 1.0; // 1 - e^{-br}·s
                out = out - bracket * self.c[n - 3] * r.powi(2 * n as i32).recip();
            }
            out
        }
    }

    /// V/k_B [K] at R [nm].
    pub fn v(&self, r_nm: f64) -> f64 {
        self.v_full(r_nm)
    }
}

fn ne() -> TangToennies {
    TangToennies {
        a: 0.402915058383e8, a1: -0.428654039586e2, a2: -0.333818674327e1,
        an1: -0.534644860719e-1, an2: 0.501774999419e-2, b: 0.492438731676e2, nmax: 8,
        c: [0.440676750157e-1, 0.164892507701e-2, 0.790473640524e-4,
            0.485489170103e-5, 0.382012334054e-6, 0.385106552963e-7],
        tilde_amp: 2.36770343e6, tilde_exp: 3.93124973e1,
        rcutoff: 0.4, repsilon: 0.30894556, mass_rel: 20.1797,
    }
}
pub fn neon_tt() -> TangToennies { ne() }

pub fn argon_tt() -> TangToennies {
    TangToennies {
        a: 4.61330146e7, a1: -2.98337630e1, a2: -9.71208881,
        an1: 2.75206827e-2, an2: -1.01489050e-2, b: 4.02517211e1, nmax: 8,
        c: [4.42812017e-1, 3.26707684e-2, 2.45656537e-3,
            1.88246247e-4, 1.47012192e-5, 1.17006343e-6],
        tilde_amp: 9.36167467e5, tilde_exp: 2.15969557e1,
        rcutoff: 0.4, repsilon: 0.376182, mass_rel: 39.948,
    }
}

pub fn krypton_tt() -> TangToennies {
    let mut k = TangToennies {
        a: 0.3200711798e8, a1: -0.2430565544e1 * 10.0, a2: -0.1435536209 * 100.0,
        an1: -0.4532273868 / 10.0, an2: 0.0, b: 0.2786344368e1 * 10.0, nmax: 8,
        c: [0.8992209265e6 / 1e6, 0.7316713603e7 / 1e8, 0.7835488511e8 / 1e10, 0.0, 0.0, 0.0],
        tilde_amp: 0.8268005465e7 / 10.0, tilde_exp: 0.1682493666e1 * 10.0,
        rcutoff: 0.3, repsilon: 4.015802 / 10.0, mass_rel: 83.798,
    };
    k.add_recursive();
    k
}

pub fn xenon_tt() -> TangToennies {
    let mut x = TangToennies {
        a: 0.579317071e8, a1: -0.208311994e1 * 10.0, a2: -0.147746919 * 100.0,
        an1: -0.289687722e1 / 10.0, an2: 0.258976595e1 / 100.0, b: 0.244337880e1 * 10.0, nmax: 8,
        c: [0.200298034e7 / 1e6, 0.199130481e8 / 1e8, 0.286841040e9 / 1e10, 0.0, 0.0, 0.0],
        tilde_amp: 4.18081481e6, tilde_exp: 2.38954061e1,
        rcutoff: 0.3, repsilon: 4.37798 / 10.0, mass_rel: 131.293,
    };
    x.add_recursive();
    x
}
```

(`N_A`, `HBAR`, `U_AMU`, `PI` are unused until Task 4 — that's fine; add `#[allow(dead_code)]` on them if the build warns, or leave them since they're used shortly.)

- [ ] **Step 4: Register the module in `src/lib.rs`**

Add to `src/lib.rs` (next to the other `pub mod` lines, e.g. after `pub mod molecule;`):
```rust
pub mod noblegas;
```

- [ ] **Step 5: Run to verify it passes**

Run: `cargo test noblegas_potential_anchors`
Expected: PASS. Run `cargo build` — clean (unused-const warnings for N_A/HBAR/U_AMU/PI are acceptable here; they are used in Task 4).

- [ ] **Step 6: Commit**

```bash
git add src/noblegas.rs src/lib.rs src/main.rs
git commit -m "$(cat <<'EOF'
Add TangToennies noble-gas potential (Ne/Ar/Kr/Xe value + constructors)

Generic-over-DualNum V/k_B with the short-range tilde branch; Kr/Xe use
the C2n recurrence. Validated vs the published TT potential anchors.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Spatial derivatives via `Dual3`

**Files:** Modify `src/noblegas.rs`; Test `src/main.rs`.

- [ ] **Step 1: Write the failing test**

```rust
    #[test]
    fn noblegas_v_derivs_match_analytic() {
        use potter_poc::noblegas::argon_tt;
        // Argon at R=0.5 nm (TT branch); analytic reference from integrate_potentials.py
        let (v, vp, vpp, vppp) = argon_tt().v_derivs(0.5);
        assert!((v - (-38.818381)).abs() < 1e-4, "V {v}");
        assert!((vp - 492.757085).abs() / 492.757085 < 1e-6, "V' {vp}");
        assert!((vpp - (-6719.247296)).abs() / 6719.247296 < 1e-6, "V'' {vpp}");
        assert!((vppp - 83513.124167).abs() / 83513.124167 < 1e-6, "V''' {vppp}");
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test noblegas_v_derivs_match_analytic`
Expected: FAIL — no method `v_derivs`.

- [ ] **Step 3: Implement** — add to `impl TangToennies` in `src/noblegas.rs`:

```rust
    /// (V, V', V'', V''') in K/nmᵏ at R [nm], via num-dual third derivative.
    pub fn v_derivs(&self, r_nm: f64) -> (f64, f64, f64, f64) {
        num_dual::third_derivative(|r| self.v_full(r), r_nm)
    }
```

(If Task 1 found `third_derivative` returns the tuple in a different order/arity, match it.)

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test noblegas_v_derivs_match_analytic`
Expected: PASS (dual derivatives equal the analytic ones to ~1e-6).

- [ ] **Step 5: Commit**

```bash
git add src/noblegas.rs src/main.rs
git commit -m "$(cat <<'EOF'
Add Dual3 spatial derivatives V'/V''/V''' for TangToennies

Validated against integrate_potentials.py's analytic potprime/2/3 (Ar).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Classical B₂ on a fixed log grid (`grid_potvals`, `b2_generic`, `b2`)

**Files:** Modify `src/noblegas.rs`; Test `src/main.rs`.

- [ ] **Step 1: Write the failing test**

```rust
    #[test]
    fn noblegas_classical_b2() {
        use potter_poc::noblegas::{neon_tt, argon_tt, krypton_tt, xenon_tt};
        // classical B2 [cm^3/mol] vs integrate_potentials.py reference
        let cases: &[(&str, fn() -> potter_poc::noblegas::TangToennies, [(f64, f64); 4])] = &[
            ("Ne", neon_tt, [(50.0, -38.5466), (100.0, -4.9747), (300.0, 11.4662), (1000.0, 13.8749)]),
            ("Ar", argon_tt, [(50.0, -774.1828), (100.0, -183.8396), (300.0, -15.2992), (1000.0, 20.2995)]),
            ("Kr", krypton_tt, [(50.0, -2507.8925), (100.0, -427.6767), (300.0, -50.2435), (1000.0, 18.5870)]),
            ("Xe", xenon_tt, [(50.0, -12473.4207), (100.0, -1146.4559), (300.0, -128.5996), (1000.0, 12.2082)]),
        ];
        for (nm, ctor, refs) in cases {
            let g = ctor();
            for (t, b2ref) in refs {
                let b2 = g.b2(*t, 0);
                assert!((b2 - b2ref).abs() / b2ref.abs() < 2e-3, "{nm} T={t}: {b2} vs {b2ref}");
            }
        }
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test noblegas_classical_b2`
Expected: FAIL — no method `b2` (and the `pub` type alias in the `fn()` pointer requires `TangToennies` exported, which it is).

- [ ] **Step 3: Implement** — add to `impl TangToennies` in `src/noblegas.rs`:

```rust
    /// Precompute the T-independent grid: per point `[R_m, V_J, V'_{J/m}, V''_{J/m²},
    /// V'''_{J/m³}]` on a 10000-point log grid in R, [0.01·rε, 1e4·rε] nm.
    fn grid_potvals(&self) -> Vec<[f64; 5]> {
        let n = 10000usize;
        let (lo, hi) = ((0.01 * self.repsilon).ln(), (1e4 * self.repsilon).ln());
        (0..n)
            .map(|i| {
                let r_nm = (lo + (hi - lo) * (i as f64) / ((n - 1) as f64)).exp();
                let (v, vp, vpp, vppp) = self.v_derivs(r_nm);
                [r_nm * 1e-9, v * KB, vp * KB * 1e9, vpp * KB * 1e18, vppp * KB * 1e27]
            })
            .collect()
    }

    /// WK integrand bracket × R² at one grid point (generic over the dual T scalar).
    /// `order` ∈ {0,1,2,3}. β = 1/(k_B T), λ = ħ²β/(12 m).
    fn integrand<D: DualNum<f64> + Copy>(&self, pt: &[f64; 5], beta: D, lam: D, order: u8) -> D {
        let (rm, vj, vp, vpp, vppp) = (pt[0], pt[1], pt[2], pt[3], pt[4]);
        let e = (-(beta * vj)).exp();
        let p = beta * vp;
        let p2 = beta * vpp;
        let p3 = beta * vppp;
        let mut g = -(e - 1.0); // order 0 (classical)
        if order >= 1 {
            g = g + lam * e * p.powi(2);
        }
        if order >= 2 {
            g = g - lam.powi(2) * e
                * (p2.powi(2) * (6.0 / 5.0)
                    + p.powi(2) * (12.0 / (5.0 * rm * rm))
                    + p.powi(3) * (4.0 / (3.0 * rm))
                    - p.powi(4) * (1.0 / 6.0));
        }
        if order >= 3 {
            g = g + lam.powi(3) * e
                * (p3.powi(2) * (36.0 / 35.0)
                    + p2.powi(2) * (216.0 / (35.0 * rm * rm))
                    + p2.powi(3) * (24.0 / 21.0)
                    + p * p2.powi(2) * (24.0 / (5.0 * rm))
                    + p.powi(3) * (288.0 / (315.0 * rm * rm * rm))
                    - p.powi(2) * p2.powi(2) * (6.0 / 5.0)
                    - p.powi(4) * (2.0 / (15.0 * rm * rm))
                    - p.powi(5) * (2.0 / (5.0 * rm))
                    + p.powi(6) * (1.0 / 30.0));
        }
        g * (rm * rm)
    }

    /// B₂ [cm³/mol] generic over the dual temperature `t`, integrating the precomputed
    /// grid by the trapezoidal rule. `order` selects the WK truncation.
    fn b2_generic<D: DualNum<f64> + Copy>(&self, t: D, order: u8, pv: &[[f64; 5]]) -> D {
        let m = self.mass_rel * U_AMU;
        let beta = (t * KB).recip();
        let lam = beta * (HBAR * HBAR / (12.0 * m));
        let mut integ = D::from(0.0);
        for w in pv.windows(2) {
            let gi = self.integrand(&w[0], beta, lam, order);
            let gj = self.integrand(&w[1], beta, lam, order);
            integ = integ + (gi + gj) * (0.5 * (w[1][0] - w[0][0]));
        }
        integ * (2.0 * PI * N_A * 1e6)
    }

    /// B₂ [cm³/mol] at temperature `t` [K], WK truncation `order` (0 = classical).
    pub fn b2(&self, t: f64, order: u8) -> f64 {
        let pv = self.grid_potvals();
        self.b2_generic(t, order, &pv)
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test noblegas_classical_b2`
Expected: PASS (all 16 within 0.2%). If any is marginal, the cause is grid resolution — confirm `n = 10000` matches the reference grid; do not loosen below 5e-3 without flagging.

- [ ] **Step 5: Commit**

```bash
git add src/noblegas.rs src/main.rs
git commit -m "$(cat <<'EOF'
Add classical noble-gas B2 (fixed log grid, cm^3/mol)

grid_potvals precomputes R/V/V'/V''/V''' once (dual); b2_generic
trapz-integrates the (order-0) integrand. Validated vs the file for
Ne/Ar/Kr/Xe at 50-1000 K.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Wigner–Kirkwood corrections (orders 1–3)

**Files:** Test `src/main.rs`. (The integrand from Task 4 already implements orders 1–3; this task adds the validating test, confirming the quantum terms.)

- [ ] **Step 1: Write the failing test**

```rust
    #[test]
    fn noblegas_wk3_b2() {
        use potter_poc::noblegas::{neon_tt, argon_tt, krypton_tt, xenon_tt};
        // WK order-3 B2 [cm^3/mol] vs integrate_potentials.py reference
        let cases: &[(&str, fn() -> potter_poc::noblegas::TangToennies, [(f64, f64); 4])] = &[
            ("Ne", neon_tt, [(50.0, -36.4805), (100.0, -4.4326), (300.0, 11.5664), (1000.0, 13.8950)]),
            ("Ar", argon_tt, [(50.0, -756.9259), (100.0, -182.3749), (300.0, -15.1793), (1000.0, 20.3171)]),
            ("Kr", krypton_tt, [(50.0, -2473.9746), (100.0, -426.1043), (300.0, -50.1607), (1000.0, 18.5973)]),
            ("Xe", xenon_tt, [(50.0, -12335.0404), (100.0, -1143.5702), (300.0, -128.5143), (1000.0, 12.2166)]),
        ];
        for (nm, ctor, refs) in cases {
            let g = ctor();
            for (t, b2ref) in refs {
                let b2 = g.b2(*t, 3);
                assert!((b2 - b2ref).abs() / b2ref.abs() < 2e-3, "{nm} T={t}: {b2} vs {b2ref}");
            }
            // the quantum correction is a small, well-defined shift off classical at low T
            assert!((g.b2(50.0, 3) - g.b2(50.0, 0)).abs() > 0.0);
        }
    }
```

- [ ] **Step 2: Run to verify it (the WK terms already exist from Task 4 — confirm it passes)**

Run: `cargo test noblegas_wk3_b2`
Expected: PASS. (If it does not, the bug is in the order≥1/2/3 branches of `integrand` from Task 4 — compare term-by-term with the spec §4 / `get_integrand`.)

- [ ] **Step 3: Commit**

```bash
git add src/main.rs
git commit -m "$(cat <<'EOF'
Validate Wigner-Kirkwood B2 to 3rd order vs the reference (Ne-Xe)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Temperature derivatives & n_eff via `Dual2`

**Files:** Modify `src/noblegas.rs`; Test `src/main.rs`.

- [ ] **Step 1: Write the failing test**

```rust
    #[test]
    fn noblegas_b2_neff_dual_t() {
        use potter_poc::noblegas::argon_tt;
        let g = argon_tt();
        let (b2, db2, d2b2, neff) = g.b2_neff(300.0, 3);
        // b2 from the dual-T path equals the plain b2()
        assert!((b2 - g.b2(300.0, 3)).abs() / b2.abs() < 1e-9, "b2 {b2}");
        // dB2/dT matches a central finite difference of b2()
        let h = 0.5;
        let fd = (g.b2(300.0 + h, 3) - g.b2(300.0 - h, 3)) / (2.0 * h);
        assert!((db2 - fd).abs() / fd.abs() < 1e-3, "dB2/dT {db2} vs FD {fd}");
        // n_eff finite, positive, and consistent with the returned derivatives
        let chk = -3.0 * (b2 + 300.0 * db2) / (2.0 * 300.0 * db2 + 300.0 * 300.0 * d2b2);
        assert!(neff.is_finite() && neff > 0.0 && (neff - chk).abs() < 1e-9, "neff {neff}");
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test noblegas_b2_neff_dual_t`
Expected: FAIL — no method `b2_neff`.

- [ ] **Step 3: Implement** — add to `impl TangToennies` in `src/noblegas.rs`:

```rust
    /// B₂ [cm³/mol], dB₂/dT, d²B₂/dT² (via dual-T), and n_eff (paper Eq. 11), at
    /// temperature `t` [K] and WK truncation `order`.
    pub fn b2_neff(&self, t: f64, order: u8) -> (f64, f64, f64, f64) {
        let pv = self.grid_potvals();
        let (b2, db2_dt, d2b2_dt2) =
            num_dual::second_derivative(|tt| self.b2_generic(tt, order, &pv), t);
        let neff = -3.0 * (b2 + t * db2_dt) / (2.0 * t * db2_dt + t * t * d2b2_dt2);
        (b2, db2_dt, d2b2_dt2, neff)
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test noblegas_b2_neff_dual_t`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/noblegas.rs src/main.rs
git commit -m "$(cat <<'EOF'
Add b2_neff: B2 T-derivatives via dual-T and n_eff for noble gases

second_derivative threads a Dual2 temperature through the fixed-grid
integral -> B2, dB2/dT, d2B2/dT2 -> n_eff (paper Eq. 11). dB2/dT matches
a central finite difference of b2().

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Regression + re-exports

**Files:** Modify `src/lib.rs`; full suite.

- [ ] **Step 1: Add convenience re-exports** in `src/lib.rs` (optional but tidy) — after the `pub mod noblegas;` line, add:

```rust
pub use noblegas::{argon_tt, krypton_tt, neon_tt, xenon_tt, TangToennies};
```

- [ ] **Step 2: Full regression**

Run: `cargo test`
Expected: all prior tests plus the 6 new noble-gas tests pass. (B3-cubature / CO2-QFH tests are slow — allow ~4 min.) Paste the final `test result:` line. If anything else regressed, fix before continuing.

- [ ] **Step 3: Confirm the wasm build is unaffected**

Run: `cargo build --lib --target wasm32-unknown-unknown`
Expected: compiles. `num-dual` is pure Rust and should build for wasm; if it (or a transitive dep) fails on wasm, gate the module: change `pub mod noblegas;` to `#[cfg(not(target_arch = "wasm32"))] pub mod noblegas;` and likewise gate the re-export, then re-run. Report which path was needed.

- [ ] **Step 4: Commit**

```bash
git add src/lib.rs
git commit -m "$(cat <<'EOF'
Re-export noble-gas API; regression green

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review (completed against the spec)

**Spec coverage:**
- §2 TT potential + params (Ne/Ar/Kr/Xe, add_recursive, tilde branch) → Task 2. ✓
- §3 dual-r derivatives (Dual3, validated vs analytic) → Tasks 1, 3. ✓
- §4 WK-series integrand orders 0–3 (exact formula) → Task 4 (`integrand`) + Task 5 (validation). ✓
- §5 fixed-log-grid integration, cm³/mol → Task 4 (`grid_potvals`, `b2_generic`, `b2`). ✓
- §6 dual-T T-derivatives + n_eff → Task 6. ✓
- §7 public API (`v`, `v_derivs`, `b2`, `b2_neff`, constructors) → Tasks 2–6. ✓
- §8 validation (pot anchors, dual vs analytic derivs, B₂ vs file, n_eff sanity) → Tasks 2, 3, 4, 5, 6. ✓
- §9 out of scope (He, QFH, web) → not in any task. ✓
- §10 files (noblegas.rs, Cargo.toml +num-dual, lib.rs, tests) → Tasks 1, 2, 7. ✓

**Placeholder scan:** none — every step has complete code and exact commands; the reference test targets are concrete numbers generated from the reference script. The two flagged uncertainties (exact `num-dual` tuple API; wasm-gating) are resolved by Task 1's smoke test and Task 7's build check respectively, with the fallback action spelled out.

**Type consistency:** `TangToennies` fields and `v_full<D: DualNum<f64>+Copy>` are defined in Task 2 and used unchanged in Tasks 3–6. `v(r)`, `v_derivs(r)->(f64,f64,f64,f64)`, `grid_potvals()->Vec<[f64;5]>`, `integrand(&[f64;5], D, D, u8)->D`, `b2_generic(D,u8,&[[f64;5]])->D`, `b2(f64,u8)->f64`, `b2_neff(f64,u8)->(f64,f64,f64,f64)` are consistent across tasks and the test call sites. Constructors `neon_tt/argon_tt/krypton_tt/xenon_tt` consistent throughout.
