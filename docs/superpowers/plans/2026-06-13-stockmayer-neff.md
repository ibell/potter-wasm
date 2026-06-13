# Stockmayer (12-6-3) n_eff — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute `n_eff(T*)` for the Stockmayer 12-6-3 potential vs. reduced dipole `(μ*)²`, exposed in the web app, reproducing Bell JCP 2020 Fig. 6.

**Architecture:** A `Stockmayer` pair energy plugs into the existing 4-D Genz-Malik cubature; a new shared-grid vector cubature (`hcubature3`) integrates B₂, B₂′, B₂″ together (analytic Boltzmann-factor derivatives) for a coherent `n_eff`. A `poc_stockmayer` wasm export feeds a reduced-unit web preset, computed by a time-sliced incremental runner so the heavy 4-D sweep fills the table progressively instead of freezing.

**Tech Stack:** Pure Rust (no new deps); `src/cubature.rs` (Genz-Malik), `src/molecule.rs` (4-D orientational), `src/lib.rs` (wasm C-ABI), `web/index.html.in` (+ Plotly). Tests in `src/main.rs`'s `#[cfg(test)] mod tests`.

**Reference spec:** `docs/superpowers/specs/2026-06-13-stockmayer-neff-design.md`

**Reduced potential (ε=σ=1), parameter (μ*)²:**
```
V*(r,θ1,θ2,φ) = 4[(1/r)^12 − (1/r)^6] − (μ*)²(1/r)^3 [2cosθ1 cosθ2 − sinθ1 sinθ2 cosφ]
```

**Conventions (from existing code):** tests import from `potter_poc::…`; reduced LJ closure `let lj = |r: f64| { let s6 = (1.0_f64/r).powi(6); 4.0*(s6*s6 - s6) };`. The orientational B₂ normalization is `-0.25*i` (validated by the single-site→spherical limit). `B2Derivs { b2, db2_dt, d2b2_dt2 }` with `.neff(t)` is in `physics.rs`. Every commit ends with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

**File responsibilities:**
- `src/cubature.rs` — add `hcubature3` (vector `[f64;3]` Genz-Malik) alongside scalar `hcubature`.
- `src/molecule.rs` — `Stockmayer` struct, `energy`, vector orientational integrand, `b2_and_derivs`.
- `src/lib.rs` — `stockmayer_b2_derivs` (native) + `poc_stockmayer` (wasm export).
- `web/index.html.in` — incremental runner + Stockmayer preset.

---

## Task 1: `hcubature3` — shared-grid vector Genz-Malik cubature

**Files:**
- Modify: `src/cubature.rs` (append after `hcubature`, end of file)
- Test: `src/main.rs` (in `#[cfg(test)] mod tests`)

- [ ] **Step 1: Write the failing test**

In `src/main.rs` `mod tests`, add:

```rust
    #[test]
    fn hcubature3_integrates_each_component_on_shared_grid() {
        use potter_poc::cubature::hcubature3;
        // f(x) = [1, x0, x0^2] over [0,1]^4 -> [1, 1/2, 1/3] (Genz-Malik is exact here).
        let f = |x: &[f64]| [1.0, x[0], x[0] * x[0]];
        let (v, _e, _n) = hcubature3(4, &f, &[0.0; 4], &[1.0; 4], 1e-10, 1e-10, 5_000_000);
        assert!((v[0] - 1.0).abs() < 1e-9, "got {}", v[0]);
        assert!((v[1] - 0.5).abs() < 1e-9, "got {}", v[1]);
        assert!((v[2] - 1.0 / 3.0).abs() < 1e-9, "got {}", v[2]);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test hcubature3_integrates_each_component_on_shared_grid`
Expected: FAIL — `hcubature3` not found in `potter_poc::cubature`.

- [ ] **Step 3: Write minimal implementation**

In `src/cubature.rs`, append at the end of the file:

```rust
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test hcubature3_integrates_each_component_on_shared_grid`
Expected: PASS. Also run `cargo build` (clean).

- [ ] **Step 5: Commit**

```bash
git add src/cubature.rs src/main.rs
git commit -m "$(cat <<'EOF'
Add hcubature3: shared-grid [f64;3] vector Genz-Malik cubature

Integrates B2/B2'/B2'' on one region subdivision (max-component error
drives refinement) so the derivatives' errors stay coherent for n_eff.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `Stockmayer` struct + `energy`

**Files:**
- Modify: `src/molecule.rs` (append near the other potential structs, e.g. end of file)
- Test: `src/main.rs`

- [ ] **Step 1: Write the failing test**

In `src/main.rs` `mod tests`, add:

```rust
    #[test]
    fn stockmayer_energy_lj_plus_dipole() {
        use potter_poc::molecule::Stockmayer;
        // (μ*)²=0 -> pure LJ, angle-independent.
        let lj = Stockmayer { eps: 1.0, sig: 1.0, mu2: 0.0 };
        let r = 1.5_f64;
        let s6 = (1.0_f64 / r).powi(6);
        let lj_ref = 4.0 * (s6 * s6 - s6);
        assert!((lj.energy(r, 0.7, 1.1, 0.3) - lj_ref).abs() < 1e-12);
        // (μ*)²=1, fully aligned head-to-tail (θ1=θ2=0): ang = 2, so U = LJ - 1*(1/r)^3*2.
        let sm = Stockmayer { eps: 1.0, sig: 1.0, mu2: 1.0 };
        let dip = 2.0 * (1.0_f64 / r).powi(3);
        assert!((sm.energy(r, 0.0, 0.0, 0.0) - (lj_ref - dip)).abs() < 1e-12);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test stockmayer_energy_lj_plus_dipole`
Expected: FAIL — `Stockmayer` not found in `potter_poc::molecule`.

- [ ] **Step 3: Write minimal implementation**

In `src/molecule.rs`, append at the end of the file:

```rust
/// Stockmayer 12-6-3 potential: a Lennard-Jones centre with a point dipole.
/// Reduced form (eps=sig=1) with parameter `mu2 = (μ*)²`:
///   V = 4 eps [(sig/r)^12 - (sig/r)^6]
///       - eps mu2 (sig/r)^3 [2 cosθ1 cosθ2 - sinθ1 sinθ2 cosφ]
/// (Bell, J. Chem. Phys. 152, 164508 (2020), Eqs. 47-48.)
pub struct Stockmayer {
    pub eps: f64,
    pub sig: f64,
    pub mu2: f64,
}

impl Stockmayer {
    /// Pair energy at COM separation `r`, orientations (θ1; θ2, φ).
    pub fn energy(&self, r: f64, th1: f64, th2: f64, phi: f64) -> f64 {
        let sr = self.sig / r;
        let sr3 = sr * sr * sr;
        let sr6 = sr3 * sr3;
        let lj = 4.0 * self.eps * (sr6 * sr6 - sr6);
        let ang = 2.0 * th1.cos() * th2.cos() - th1.sin() * th2.sin() * phi.cos();
        lj - self.eps * self.mu2 * sr3 * ang
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test stockmayer_energy_lj_plus_dipole`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/molecule.rs src/main.rs
git commit -m "$(cat <<'EOF'
Add Stockmayer 12-6-3 pair energy (LJ centre + point dipole)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `Stockmayer::b2_and_derivs` (4-D B₂ + T-derivatives) + validation

**Files:**
- Modify: `src/molecule.rs` (add imports + the vector integrand + method)
- Test: `src/main.rs`

- [ ] **Step 1: Write the failing tests**

In `src/main.rs` `mod tests`, add:

```rust
    #[test]
    fn stockmayer_zero_dipole_matches_lj() {
        use potter_poc::b2_and_derivs_v;
        use potter_poc::molecule::Stockmayer;
        let lj = |r: f64| {
            let s6 = (1.0_f64 / r).powi(6);
            4.0 * (s6 * s6 - s6)
        };
        let sm = Stockmayer { eps: 1.0, sig: 1.0, mu2: 0.0 };
        for &t in &[2.0_f64, 5.0] {
            let (d, _n) = sm.b2_and_derivs(t, 1e-4);
            let r = b2_and_derivs_v(&lj, t, 1e-11);
            assert!((d.b2 - r.b2).abs() / r.b2.abs() < 5e-3, "B2 T*={t}: {} vs {}", d.b2, r.b2);
            assert!((d.db2_dt - r.db2_dt).abs() / r.db2_dt.abs() < 5e-3, "B2' T*={t}");
            assert!((d.d2b2_dt2 - r.d2b2_dt2).abs() / r.d2b2_dt2.abs() < 5e-3, "B2'' T*={t}");
            assert!((d.neff(t) - r.neff(t)).abs() < 1e-2, "neff T*={t}: {} vs {}", d.neff(t), r.neff(t));
        }
    }

    #[test]
    fn stockmayer_dipole_lowers_neff_and_b2() {
        use potter_poc::molecule::Stockmayer;
        // At a fixed T*, a stronger dipole adds attraction: B2 more negative, n_eff lower.
        let t = 3.0_f64;
        let mut prev_b2 = f64::INFINITY;
        let mut prev_neff = f64::INFINITY;
        for &mu2 in &[0.0_f64, 2.0, 4.0] {
            let sm = Stockmayer { eps: 1.0, sig: 1.0, mu2 };
            let (d, _n) = sm.b2_and_derivs(t, 1e-4);
            let ne = d.neff(t);
            assert!(d.b2 < prev_b2, "B2 not decreasing at mu2={mu2}: {} !< {}", d.b2, prev_b2);
            assert!(ne < prev_neff, "n_eff not decreasing at mu2={mu2}: {} !< {}", ne, prev_neff);
            prev_b2 = d.b2;
            prev_neff = ne;
        }
        // mu2=0 must equal the LJ n_eff at T*=3 (~9.08).
        let sm0 = Stockmayer { eps: 1.0, sig: 1.0, mu2: 0.0 };
        assert!((sm0.b2_and_derivs(t, 1e-4).0.neff(t) - 9.077).abs() < 0.1);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test stockmayer_zero_dipole_matches_lj stockmayer_dipole_lowers_neff_and_b2`
Expected: FAIL — `b2_and_derivs` method not found on `Stockmayer`.

- [ ] **Step 3: Write minimal implementation**

In `src/molecule.rs`, change the imports at the top. The current lines are:
```rust
use crate::cubature::hcubature;
use std::collections::HashMap;
use std::f64::consts::PI;
```
Replace with:
```rust
use crate::cubature::{hcubature, hcubature3};
use crate::physics::B2Derivs;
use std::collections::HashMap;
use std::f64::consts::PI;
```

Then, in the `impl Stockmayer { ... }` block (from Task 2), add the `b2_and_derivs` method after `energy`:

```rust
    /// Reduced B₂ and its first two T*-derivatives (σ³ units; ε=σ=1 expected for the
    /// web). 4-D orientational average via `hcubature3`, differentiating e^{-U/T}
    /// analytically (U is T-independent). Returns (B2Derivs, integrand evals).
    pub fn b2_and_derivs(&self, t: f64, reltol: f64) -> (B2Derivs, usize) {
        let integrand = |x: &[f64]| -> [f64; 3] {
            let (s, th1, th2, phi) = (x[0], x[1], x[2], x[3]);
            let om = 1.0 - s;
            if om <= 0.0 {
                return [0.0; 3];
            }
            let r = s / om;
            let w = r * r / (om * om) * th1.sin() * th2.sin(); // r^2 * Jacobian * angular
            let u = self.energy(r, th1, th2, phi);
            let (f0, f1, f2) = if u.is_finite() {
                let e = (-u / t).exp();
                let t2 = t * t;
                (e - 1.0, e * u / t2, e * (u * u / (t2 * t2) - 2.0 * u / (t2 * t)))
            } else {
                (-1.0, 0.0, 0.0)
            };
            let mut out = [f0 * w, f1 * w, f2 * w];
            for v in out.iter_mut() {
                if !v.is_finite() {
                    *v = 0.0;
                }
            }
            out
        };
        let (i, _e, nev) = hcubature3(
            4,
            &integrand,
            &[0.0, 0.0, 0.0, 0.0],
            &[1.0, PI, PI, 2.0 * PI],
            1e-3,
            reltol,
            5_000_000,
        );
        // Same -0.25 orientational normalization as `b2_orientational`; reduced units
        // (σ³), so NO ANG3_TO_CM3MOL conversion here.
        let d = B2Derivs {
            b2: -0.25 * i[0],
            db2_dt: -0.25 * i[1],
            d2b2_dt2: -0.25 * i[2],
        };
        (d, nev)
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test stockmayer_zero_dipole_matches_lj stockmayer_dipole_lowers_neff_and_b2`
Expected: PASS (both). If `stockmayer_zero_dipole_matches_lj` is marginally outside 5e-3, tighten the test's cubature `reltol` from `1e-4` to `1e-5` and re-run.

- [ ] **Step 5: Commit**

```bash
git add src/molecule.rs src/main.rs
git commit -m "$(cat <<'EOF'
Add Stockmayer::b2_and_derivs (4-D B2 + analytic T-derivatives)

Vector orientational integrand on the shared-grid hcubature3; reduced
B2Derivs via the validated -0.25 normalization. Validated: (mu*)^2=0
reproduces LJ B2/derivs/n_eff; stronger dipole lowers B2 and n_eff.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: native `stockmayer_b2_derivs` + `poc_stockmayer` wasm export

**Files:**
- Modify: `src/lib.rs`
- Test: `src/main.rs`

- [ ] **Step 1: Write the failing test**

In `src/main.rs` `mod tests`, add:

```rust
    #[test]
    fn stockmayer_b2_derivs_reduced_matches_molecule() {
        use potter_poc::stockmayer_b2_derivs;
        use potter_poc::molecule::Stockmayer;
        let d = stockmayer_b2_derivs(3.0, 2.0, 1e-4);
        let (m, _n) = Stockmayer { eps: 1.0, sig: 1.0, mu2: 2.0 }.b2_and_derivs(3.0, 1e-4);
        assert!((d.b2 - m.b2).abs() < 1e-12);
        assert!((d.db2_dt - m.db2_dt).abs() < 1e-12);
        assert!((d.d2b2_dt2 - m.d2b2_dt2).abs() < 1e-12);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test stockmayer_b2_derivs_reduced_matches_molecule`
Expected: FAIL — `stockmayer_b2_derivs` not found.

- [ ] **Step 3: Write minimal implementation**

In `src/lib.rs`, after `b2_derivs_from_dsl` (the function added in the n_eff work), add:

```rust
/// Reduced Stockmayer (ε=σ=1) B₂ and its first two T*-derivatives at reduced
/// temperature `tstar` and dipole strength `mu2 = (μ*)²`.
pub fn stockmayer_b2_derivs(tstar: f64, mu2: f64, reltol: f64) -> B2Derivs {
    let sm = crate::molecule::Stockmayer { eps: 1.0, sig: 1.0, mu2 };
    sm.b2_and_derivs(tstar, reltol).0
}
```

Then, inside `mod wasm_exports`, update the `use super::{...};` line to include `stockmayer_b2_derivs`, e.g.:
```rust
    use super::{b2_derivs_from_dsl, b2_from_dsl, b3_from_dsl, stockmayer_b2_derivs};
```
and add this export (next to `poc_b2_derivs`):

```rust
    /// Reduced Stockmayer: write [B2*, dB2*/dT*, d2B2*/dT*2, n_eff] (4 f64) into the
    /// caller `out` array. NaN-fill on a non-finite result. Uses unaligned writes so
    /// `out` need not be 8-byte aligned.
    #[no_mangle]
    pub extern "C" fn poc_stockmayer(tstar: f64, mu2: f64, reltol: f64, out: *mut f64) {
        let d = stockmayer_b2_derivs(tstar, mu2, reltol);
        let vals = [d.b2, d.db2_dt, d.d2b2_dt2, d.neff(tstar)];
        unsafe {
            for (k, v) in vals.iter().enumerate() {
                out.add(k).write_unaligned(*v);
            }
        }
    }
```

- [ ] **Step 4: Verify**

Run: `cargo test stockmayer_b2_derivs_reduced_matches_molecule`  -> PASS.
Run: `cargo build --lib --target wasm32-unknown-unknown`  -> compiles cleanly (only the pre-existing `aot.rs`/series `unused` warnings). Report the result. (If the target is missing: `rustup target add wasm32-unknown-unknown`.)

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/main.rs
git commit -m "$(cat <<'EOF'
Add stockmayer_b2_derivs + poc_stockmayer wasm export

Reduced-unit (eps=sig=1) Stockmayer B2/derivs/n_eff; the export writes
[B2,dB2dT,d2B2dT2,neff] into a caller f64 array (unaligned writes).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Web — time-sliced incremental, cancellable runner

**Files:**
- Modify: `web/index.html.in` (refactor `compute()`; add `computeRow`, run token)

This task refactors the existing synchronous `compute()` into an async, time-sliced
runner that fills the table/plot progressively. It must keep the existing presets
(LJ/Mie/inv/custom) producing identical results. Verification is build + JS syntax +
node-numeric (the numeric core is unchanged) + a manual smoke note (DOM/Plotly can't be
unit-tested here).

- [ ] **Step 1: Refactor `compute()` into an incremental runner**

In `web/index.html.in`, replace the entire existing `function compute() { ... }` (the one
that loops temperatures and calls `renderTable`/`renderPlotly`) with:

```javascript
let runGen = 0; // generation token: a newer run cancels an older in-flight one

// Build one result row for temperature T with the current preset/params.
function computeRow(preset, pot, T, eps, sig) {
  if (preset === "stockmayer") {
    const mu2 = parseFloat($("mu2").value) || 0;
    const r = stockmayerReduced(T / eps, mu2);        // reduced (ε=σ=1)
    const s3 = sig ** 3;
    return { T, num: s3 * r.b2, db2: (s3 * r.db2) / eps,
             d2b2: (s3 * r.d2b2) / (eps * eps), neff: r.neff, ex: null };
  }
  const d = b2derivs(pot.fn, eps, sig, T);
  return { T, num: d.b2, db2: d.db2, d2b2: d.d2b2, neff: d.neff, ex: exactFor(T, eps, sig) };
}

async function compute() {
  const gen = ++runGen;
  const preset = presetSel.value;
  const eps = parseFloat($("eps").value), sig = parseFloat($("sig").value);
  let tmin = parseFloat($("tmin").value), tmax = parseFloat($("tmax").value);
  const np = Math.max(2, parseInt($("npts").value) || 2);
  const log = $("tspace").value === "log";
  if (log && tmin <= 0) tmin = 1e-3;

  let pot = null;
  if (preset !== "stockmayer") {
    try { pot = makePotential(dslEl.value); }
    catch (e) { $("status").innerHTML = `<span class="err">✗ ${e.message}</span>`; return; }
  }

  const Ts = tGrid(tmin, tmax, np, log);
  const rows = [];
  let i = 0;
  while (i < Ts.length) {
    if (gen !== runGen) return; // a newer run started — abandon this one
    const sliceStart = performance.now();
    do {
      rows.push(computeRow(preset, pot, Ts[i], eps, sig));
      i++;
    } while (i < Ts.length && performance.now() - sliceStart < 16);

    const hasExact = rows.some(r => r.ex);
    lastRows = rows.slice();
    lastHasExact = hasExact;
    renderTable(rows, hasExact);
    renderPlotly(rows, hasExact, log);
    $("status").innerHTML = i < Ts.length
      ? `computing ${i}/${Ts.length}…`
      : `✓ ${preset === "stockmayer" ? "Stockmayer 4-D cubature" : "compiled <code>" +
          dslEl.value.replace(/</g, "&lt;") + "</code>"} — ${np} temperatures ` +
        `(${log ? "log" : "linear"} spacing): B₂, its first two T-derivatives, and n_eff.` +
        (hasExact ? "" : " <span class=note>(no closed-form overlay)</span>");
    if (i < Ts.length) await new Promise(r => setTimeout(r, 0));
  }
}
```

- [ ] **Step 2: Add a placeholder `stockmayerReduced` so the file parses**

`computeRow` references `stockmayerReduced` (implemented fully in Task 6). Add a stub now
so Task 5 builds/parses on its own. In `web/index.html.in`, just before `function compute`,
add:

```javascript
// Reduced Stockmayer via the wasm export (filled in for the Stockmayer preset).
function stockmayerReduced(tstar, mu2, reltol = 1e-3) {
  const out = ex.poc_alloc(32); // 4 * f64
  ex.poc_stockmayer(tstar, mu2, reltol, out);
  const dv = new DataView(ex.memory.buffer);
  const r = {
    b2: dv.getFloat64(out, true),
    db2: dv.getFloat64(out + 8, true),
    d2b2: dv.getFloat64(out + 16, true),
    neff: dv.getFloat64(out + 24, true),
  };
  ex.poc_dealloc(out, 32);
  return r;
}
```

(`poc_stockmayer` exists from Task 4, so this is functional already — no real stub needed.)

- [ ] **Step 3: Build and verify the existing presets still work**

Run: `./web/build.sh`  (rebuilds wasm incl. `poc_stockmayer`, regenerates `docs/index.html`).
Then syntax-check the embedded JS:
```bash
python3 - <<'PY'
import re
html=open("web/index.html.in").read()
m=re.search(r'<script>\n(.*)</script>', html, re.S)
open('/tmp/webjs.js','w').write(m.group(1))
PY
node --check /tmp/webjs.js && echo "JS SYNTAX OK"
```
Expected: build succeeds, `JS SYNTAX OK`.

Manual smoke (note for the executor; not automated): open `docs/index.html`, confirm the
LJ and Mie presets still plot and the table fills (now via the async runner), and that
changing inputs still recomputes.

- [ ] **Step 4: Commit**

```bash
git add web/index.html.in docs/index.html
git commit -m "$(cat <<'EOF'
web: time-sliced incremental, cancellable compute runner

compute() is now async: rows are computed on a ~16ms time budget and the
table/plot fill in progressively; a generation token cancels a stale run
when inputs change mid-sweep. Cheap presets still finish in one tick.
Adds computeRow() + stockmayerReduced() (wasm) ahead of the preset.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Web — Stockmayer preset + (μ*)² control

**Files:**
- Modify: `web/index.html.in` (preset option, `(μ*)²` input, `syncPreset`, `repExponent`, reactivity)

- [ ] **Step 1: Add the preset option and the (μ*)² input**

In `web/index.html.in`, in the preset `<select id="preset">`, add the Stockmayer option
after the Mie option:
```html
          <option value="mie">Mie  (n–m)</option>
          <option value="stockmayer">Stockmayer (12-6-3)</option>
```
And after the `mwrap` label, add the (μ*)² field:
```html
      <label id="mwrap" hidden>m <input id="mexp" type="number" value="6" min="4" step="1" /></label>
      <label id="mu2wrap" hidden>(μ*)² <input id="mu2" type="number" value="2" min="0" step="0.5" /></label>
```

- [ ] **Step 2: Wire the input ref, `syncPreset`, reactivity, and the reference line**

In `web/index.html.in`:

(a) Add the element ref. The existing line:
```javascript
const nwrap = $("nwrap"), nexp = $("nexp"), mwrap = $("mwrap"), mexp = $("mexp");
```
becomes:
```javascript
const nwrap = $("nwrap"), nexp = $("nexp"), mwrap = $("mwrap"), mexp = $("mexp");
const mu2wrap = $("mu2wrap"), mu2el = $("mu2");
```

(b) In `syncPreset()`, handle the Stockmayer case and the `(μ*)²` visibility. Replace the
body of `syncPreset` with:
```javascript
function syncPreset() {
  const p = presetSel.value;
  nwrap.hidden = !(p === "inv" || p === "mie");
  mwrap.hidden = p !== "mie";
  mu2wrap.hidden = p !== "stockmayer";
  if (p === "lj") {
    dslEl.value = "4*eps*((sig/r)**12 - (sig/r)**6)"; dslEl.readOnly = true;
  } else if (p === "inv") {
    dslEl.value = `eps*(sig/r)**${parseInt(nexp.value) || 12}`; dslEl.readOnly = true;
  } else if (p === "mie") {
    const n = parseInt(nexp.value) || 12, m = parseInt(mexp.value) || 6;
    dslEl.value = `${mieC(n, m)}*eps*((sig/r)**${n} - (sig/r)**${m})`; dslEl.readOnly = true;
  } else if (p === "stockmayer") {
    dslEl.value = "4*eps*((sig/r)**12 - (sig/r)**6) - eps*mu2*(sig/r)**3*[2cosθ₁cosθ₂ - sinθ₁sinθ₂cosφ]";
    dslEl.readOnly = true; // informational only — computed by 4-D cubature, not the DSL
  } else {
    dslEl.readOnly = false;
  }
}
```

(c) Make the `(μ*)²` field reactive. After the existing `mexp.onchange = …;` handler line,
add:
```javascript
mu2el.oninput = () => { if (presetSel.value === "stockmayer") syncPreset(); };
mu2el.onchange = () => { if (presetSel.value === "stockmayer") { syncPreset(); recompute(); } };
```

(d) Extend `repExponent()` so the Stockmayer panel keeps the n=12 high-T reference. The
existing function:
```javascript
function repExponent() {
  const p = presetSel.value;
  if (p === "lj") return 12;
  if (p === "mie" || p === "inv") return parseInt(nexp.value) || 12;
  return NaN;
}
```
becomes:
```javascript
function repExponent() {
  const p = presetSel.value;
  if (p === "lj" || p === "stockmayer") return 12; // LJ repulsive core -> high-T limit 12
  if (p === "mie" || p === "inv") return parseInt(nexp.value) || 12;
  return NaN;
}
```

- [ ] **Step 2.5: Verify reduced units note**

`computeRow` (Task 5) already calls `stockmayerReduced(T/eps, mu2)` and scales by `sig³` /
the `eps` chain rule, and `repExponent()` returns 12. No further compute wiring is needed
here — the preset is now fully functional.

- [ ] **Step 3: Build + syntax check**

Run: `./web/build.sh`
```bash
python3 - <<'PY'
import re
html=open("web/index.html.in").read()
m=re.search(r'<script>\n(.*)</script>', html, re.S)
open('/tmp/webjs.js','w').write(m.group(1))
PY
node --check /tmp/webjs.js && echo "JS SYNTAX OK"
```
Expected: build OK, `JS SYNTAX OK`. Confirm the generated `docs/index.html` contains
`value="stockmayer"` and `id="mu2"`:
```bash
grep -c 'value="stockmayer"\|id="mu2"\|stockmayerReduced\|poc_stockmayer' docs/index.html
```
Expected: a non-zero count (≥4).

- [ ] **Step 4: Commit**

```bash
git add web/index.html.in docs/index.html
git commit -m "$(cat <<'EOF'
web: add Stockmayer (12-6-3) preset with a (mu*)^2 control

Reactive (mu*)^2 field; computed via the poc_stockmayer 4-D cubature
export (reduced units, scaled by sig^3 / eps chain rule); n_eff panel
keeps the n=12 high-T reference. The incremental runner fills the table
progressively so the 4-D sweep does not freeze the page.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: End-to-end validation (node) + final regression

**Files:**
- Test: a node script (transient, under `/tmp`); full Rust suite

- [ ] **Step 1: Verify `poc_stockmayer` through the built wasm in node**

Create `/tmp/stockmayer_e2e.mjs`:
```javascript
import fs from "node:fs";
const bytes = fs.readFileSync("target/wasm32-unknown-unknown/release/potter_poc.wasm");
const { instance } = await WebAssembly.instantiate(bytes, {});
const ex = instance.exports;
function stock(tstar, mu2, reltol = 1e-4) {
  const out = ex.poc_alloc(32);
  ex.poc_stockmayer(tstar, mu2, reltol, out);
  const dv = new DataView(ex.memory.buffer);
  const r = [0, 1, 2, 3].map(k => dv.getFloat64(out + 8 * k, true));
  ex.poc_dealloc(out, 32);
  return { b2: r[0], db2: r[1], d2b2: r[2], neff: r[3] };
}
// LJ reference (1-D) for the mu2=0 check
const neffFrom = (b2, db2, d2b2, T) => (-3 * (b2 + T * db2)) / (2 * T * db2 + T * T * d2b2);
function lj(T, n = 40000) { let s0 = 0, s1 = 0, s2 = 0;
  for (let i = 0; i <= n; i++) { const s = i / n, om = 1 - s; let f0 = 0, f1 = 0, f2 = 0;
    if (om > 0) { const r = s / om, w = r * r / (om * om), sr6 = (1 / r) ** 6, V = 4 * (sr6 * sr6 - sr6);
      if (Number.isFinite(V)) { const e = Math.exp(-V / T), t2 = T * T; f0 = (e - 1) * w; f1 = (e * V / t2) * w; f2 = (e * (V * V / (t2 * t2) - 2 * V / (t2 * T))) * w; } else f0 = -w;
      if (!Number.isFinite(f0)) f0 = 0; if (!Number.isFinite(f1)) f1 = 0; if (!Number.isFinite(f2)) f2 = 0; }
    const wt = i === 0 || i === n ? 1 : i % 2 ? 4 : 2; s0 += wt * f0; s1 += wt * f1; s2 += wt * f2; }
  const k = -2 * Math.PI / (3 * n); const b2 = k * s0, db2 = k * s1, d2b2 = k * s2; return neffFrom(b2, db2, d2b2, T); }
let ok = true;
for (const T of [2.0, 5.0]) {
  const s = stock(T, 0).neff, l = lj(T);
  const pass = Math.abs(s - l) < 0.02; ok &&= pass;
  console.log(`mu2=0 T*=${T}: stock neff=${s.toFixed(3)} vs LJ ${l.toFixed(3)} ${pass ? "OK" : "FAIL"}`);
}
for (const T of [3.0]) {
  const a = stock(T, 0).neff, b = stock(T, 2).neff, c = stock(T, 4).neff;
  const pass = a > b && b > c; ok &&= pass;
  console.log(`dipole T*=${T}: neff ${a.toFixed(2)} > ${b.toFixed(2)} > ${c.toFixed(2)} ${pass ? "OK" : "FAIL"}`);
}
console.log(ok ? "E2E PASS" : "E2E FAIL");
process.exit(ok ? 0 : 1);
```

Run: `node /tmp/stockmayer_e2e.mjs`
Expected: prints `E2E PASS` — `(μ*)²=0` Stockmayer `n_eff` matches LJ (≤0.02), and `n_eff`
decreases with `(μ*)²` at T*=3.

- [ ] **Step 2: Full Rust regression**

Run: `cargo test`
Expected: all prior tests plus the 5 new Stockmayer/cubature tests pass. (The B3-cubature /
CO2-QFH tests are slow; allow ~3-4 min.)

- [ ] **Step 3: Commit (if the e2e script is kept) / otherwise note**

The e2e script lives in `/tmp` (transient) — nothing to commit unless you choose to keep it
under `node/`. If keeping it:
```bash
cp /tmp/stockmayer_e2e.mjs node/stockmayer-e2e.mjs
git add node/stockmayer-e2e.mjs
git commit -m "$(cat <<'EOF'
Add node end-to-end check for poc_stockmayer (mu2=0->LJ, dipole order)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```
Otherwise this task adds no commit; it is the green-light gate before integration.

---

## Self-Review (completed against the spec)

**Spec coverage:**
- §2 potential/units → Task 2 (`energy`), Task 6 (reduced + sig³/eps scaling in `computeRow`). ✓
- §3 n_eff (reuse `B2Derivs::neff`) → Tasks 3-4. ✓
- §4.1 `Stockmayer` struct/energy → Task 2. ✓
- §4.2 shared-grid vector cubature `hcubature3` → Task 1. ✓
- §4.3 vector orientational integrand + reduced `b2_and_derivs` (−0.25 norm) → Task 3. ✓
- §5 `poc_stockmayer` export → Task 4. ✓
- §6.1 Stockmayer preset + (μ*)² field + wasm branch → Task 6 (+ `computeRow` in Task 5). ✓
- §6.2 incremental cancellable runner → Task 5. ✓
- §7 validation: (μ*)²=0→LJ (Task 3 + Task 7), n_eff/B₂ dipole ordering (Task 3 + Task 7), node e2e (Task 7). ✓ (High-T→12 is covered indirectly via the LJ match; per the prototype it overshoots at accessible T*, so no brittle tight assertion — noted in spec §7.)
- §8 out of scope (EXP, SW, Worker, Fig-6 family, other molecular derivs, quantum) → not in any task. ✓

**Placeholder scan:** none — every step has complete code and exact commands. The Task 5
`stockmayerReduced` "stub" is actually the real implementation (poc_stockmayer exists from
Task 4), as noted.

**Type consistency:** `hcubature3(dim, &f, a, b, abstol, reltol, maxevals) -> ([f64;3],[f64;3],usize)` consistent across Task 1 (def) and Task 3 (call). `Stockmayer { eps, sig, mu2 }` and `b2_and_derivs(t, reltol) -> (B2Derivs, usize)` consistent across Tasks 2-4. `B2Derivs` import added in Task 3. `stockmayer_b2_derivs(tstar, mu2, reltol) -> B2Derivs` and `poc_stockmayer(tstar, mu2, reltol, out)` consistent (Task 4 def, Task 5/7 callers). Web `computeRow`/`stockmayerReduced`/`mu2el`/`mu2wrap` consistent across Tasks 5-6. Row shape `{T, num, db2, d2b2, neff, ex}` matches the existing `renderTable`/`renderPlotly` consumers.
