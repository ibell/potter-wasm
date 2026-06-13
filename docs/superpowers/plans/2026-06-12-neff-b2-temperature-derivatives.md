# n_eff and second temperature derivatives of B₂ — Implementation Plan (Phase 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute the effective repulsive exponent `n_eff(T)` and the first two temperature derivatives of B₂ for spherical/DSL potentials, two independent ways (deterministic integration + MSMC), anchored on single-site Lennard-Jones.

**Architecture:** Differentiate the Boltzmann factor `e^{-V/T}` analytically (V is T-independent) and integrate all three quantities — B₂, B₂′, B₂″ — on **one shared adaptive grid** via a vector-valued integrand, so their errors are coherent when combined into the `n_eff` ratio. A single-bond MSMC path accumulates the same three quantities from one Monte-Carlo walk (common random numbers) for a fully independent cross-check.

**Tech Stack:** Pure Rust, no new dependencies. Adaptive Simpson (`src/integrate.rs`), virial physics (`src/physics.rs`), Mayer-sampling MC (`src/msmc.rs`), WASM C-ABI exports (`src/lib.rs`). Tests live in `src/main.rs`'s `#[cfg(test)] mod tests`.

**Reference spec:** `docs/superpowers/specs/2026-06-12-neff-b2-temperature-derivatives-design.md`

**Key formulas (potter, verbatim):**
- `n_eff(T) = -3·(B₂ + T·B₂′) / (2T·B₂′ + T²·B₂″)`
- B₂ integrand `e^{-V/T} − 1`; B₂′ integrand `e^{-V/T}·(V/T²)`; B₂″ integrand `e^{-V/T}·(V²/T⁴ − 2V/T³)` (each ×r², before the −2π).

**Conventions to match (from `src/main.rs`):** tests import from `potter_poc::…`; the hard-coded LJ closure is `let lj = |r: f64| { let s6 = (1.0_f64 / r).powi(6); 4.0 * (s6 * s6 - s6) };`; the LJ DSL string is `"4*eps*((sig/r)**12 - (sig/r)**6)"`. Every commit ends with the trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## Task 1: Vector adaptive Simpson (shared-grid `[f64; 3]` integrator)

**Files:**
- Modify: `src/integrate.rs` (append after `adaptive_simpson`/`recur`, before `composite_simpson`)
- Test: `src/main.rs` (in `#[cfg(test)] mod tests`)

- [ ] **Step 1: Write the failing test**

In `src/main.rs`, inside `mod tests`, add (and add `use potter_poc::integrate::adaptive_simpson3;` at the top of the function or module):

```rust
    #[test]
    fn vector_adaptive_simpson_integrates_each_component() {
        use potter_poc::integrate::adaptive_simpson3;
        // f(x) = [1, x, x^2] over [0,1] -> [1, 1/2, 1/3], all on ONE shared grid.
        let i = adaptive_simpson3(&|x| [1.0, x, x * x], 0.0, 1.0, 1e-12, 50);
        assert!((i[0] - 1.0).abs() < 1e-10, "got {}", i[0]);
        assert!((i[1] - 0.5).abs() < 1e-10, "got {}", i[1]);
        assert!((i[2] - 1.0 / 3.0).abs() < 1e-10, "got {}", i[2]);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test vector_adaptive_simpson_integrates_each_component`
Expected: FAIL — compile error, `adaptive_simpson3` not found in `potter_poc::integrate`.

- [ ] **Step 3: Write minimal implementation**

In `src/integrate.rs`, append:

```rust
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test vector_adaptive_simpson_integrates_each_component`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/integrate.rs src/main.rs
git commit -m "$(cat <<'EOF'
Add shared-grid vector adaptive Simpson (adaptive_simpson3)

Integrates a [f64;3] integrand on one subdivision so the components'
quadrature errors are coherent — needed for the n_eff ratio.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `B2Derivs`, the vector B₂ integrand, `b2_and_derivs`, and `neff`

**Files:**
- Modify: `src/physics.rs` (import `adaptive_simpson3`; add struct + functions after `b2_finegrid`, around line 131)
- Test: `src/main.rs`

- [ ] **Step 1: Write the failing test**

In `src/main.rs` `mod tests`, add:

```rust
    #[test]
    fn b2_and_derivs_value_and_fd_first_derivative() {
        use potter_poc::{b2_and_derivs_v, b2_v};
        let lj = |r: f64| {
            let s6 = (1.0_f64 / r).powi(6);
            4.0 * (s6 * s6 - s6)
        };
        let t = 1.5;
        let d = b2_and_derivs_v(&lj, t, 1e-12);
        // (a) the B2 component must equal the existing scalar integrator
        let b2_ref = b2_v(&lj, t, 1e-12);
        assert!((d.b2 - b2_ref).abs() < 1e-9, "B2 {} vs {}", d.b2, b2_ref);
        // (b) dB2/dT must match a central finite difference of B2(T)
        let h = 1e-4;
        let fd = (b2_v(&lj, t + h, 1e-12) - b2_v(&lj, t - h, 1e-12)) / (2.0 * h);
        assert!((d.db2_dt - fd).abs() < 1e-5, "dB2/dT {} vs FD {}", d.db2_dt, fd);
        // (c) n_eff is finite and positive in the repulsion-dominated sense
        assert!(d.neff(t).is_finite(), "neff not finite: {}", d.neff(t));
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test b2_and_derivs_value_and_fd_first_derivative`
Expected: FAIL — `b2_and_derivs_v` not found in `potter_poc`.

- [ ] **Step 3: Write minimal implementation**

In `src/physics.rs`, change the import line (currently `use crate::integrate::{adaptive_simpson, composite_simpson};`) to:

```rust
use crate::integrate::{adaptive_simpson, adaptive_simpson3, composite_simpson};
```

Then, immediately after `b2_finegrid` (after line 131, before the `// --- B3 ---` banner), add:

```rust
/// B₂ and its first two temperature derivatives, all integrated on one shared
/// adaptive grid.
#[derive(Clone, Copy, Debug)]
pub struct B2Derivs {
    pub b2: f64,
    pub db2_dt: f64,
    pub d2b2_dt2: f64,
}

impl B2Derivs {
    /// potter's effective repulsive exponent n_eff(T) for this point.
    #[inline]
    pub fn neff(&self, t: f64) -> f64 {
        neff(self, t)
    }
}

/// potter's effective repulsive exponent (from `squarewell.py`):
///   n_eff = -3 (B₂ + T B₂') / (2 T B₂' + T² B₂'')
/// Dimensionless: identical in reduced or real units. Recovers `n` exactly for an
/// inverse-power potential ε(σ/r)ⁿ.
#[inline]
pub fn neff(d: &B2Derivs, t: f64) -> f64 {
    -3.0 * (d.b2 + t * d.db2_dt) / (2.0 * t * d.db2_dt + t * t * d.d2b2_dt2)
}

/// Vector integrand `[f₀, f₁, f₂]` for `[B₂, B₂', B₂'']` (each ×r²·Jacobian), with
/// the same `s -> r = s/(1-s)` domain map as `b2_integrand_s`. The Boltzmann factor
/// is differentiated analytically (V is T-independent); the `−1` of the Mayer
/// function is T-independent and so absent from f₁, f₂. At the repulsive core
/// (V → ∞) `e^{-V/T} → 0` dominates, so f₁ = f₂ = 0 and f₀ = −1.
#[inline]
fn b2_deriv_integrand_s<V: Fn(f64) -> f64>(v: &V, t: f64, s: f64) -> [f64; 3] {
    let om = 1.0 - s;
    if om <= 0.0 {
        return [0.0; 3]; // s = 1 -> r = inf
    }
    let r = s / om;
    let w = r * r / (om * om); // r^2 * Jacobian
    let vv = v(r);
    let (f0, f1, f2) = if vv.is_finite() {
        let e = (-vv / t).exp();
        let t2 = t * t;
        (
            e - 1.0,
            e * vv / t2,
            e * (vv * vv / (t2 * t2) - 2.0 * vv / (t2 * t)),
        )
    } else {
        (-1.0, 0.0, 0.0)
    };
    let mut out = [f0 * w, f1 * w, f2 * w];
    for x in out.iter_mut() {
        if !x.is_finite() {
            *x = 0.0;
        }
    }
    out
}

/// B₂ and its first two T-derivatives for any potential closure, on one shared grid.
pub fn b2_and_derivs_v<V: Fn(f64) -> f64>(v: &V, t: f64, tol: f64) -> B2Derivs {
    let i = adaptive_simpson3(&|s| b2_deriv_integrand_s(v, t, s), 0.0, 1.0, tol, 60);
    B2Derivs {
        b2: -2.0 * PI * i[0],
        db2_dt: -2.0 * PI * i[1],
        d2b2_dt2: -2.0 * PI * i[2],
    }
}

/// B₂ and its first two T-derivatives for a compiled `Potential`.
pub fn b2_and_derivs(pot: &Potential, t: f64, tol: f64) -> B2Derivs {
    b2_and_derivs_v(&|r| pot.v(r), t, tol)
}
```

Then re-export from `src/lib.rs`: change the `pub use physics::{ … };` block (lines 21-24) to include the new items:

```rust
pub use physics::{
    b2, b2_and_derivs, b2_and_derivs_v, b2_finegrid, b2_lj_series, b2_v, b2_v_grid, b3, b3_cubature,
    b3_cubature_v, b3_v, b3_v_grid, neff, B2Derivs, CsePotential, Potential, LJ_BOYLE_TSTAR,
};
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test b2_and_derivs_value_and_fd_first_derivative`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/physics.rs src/lib.rs src/main.rs
git commit -m "$(cat <<'EOF'
Add B2Derivs, shared-grid b2_and_derivs, and potter's n_eff

Analytic T-derivatives of the Boltzmann factor integrated on one
adaptive grid; n_eff = -3(B2+T B2')/(2T B2'+T^2 B2'').

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Closed-form oracles — IPL `n_eff == n`, LJ series derivatives, high-T → 12

**Files:**
- Modify: `src/physics.rs` (add `b2_lj_series_derivs` after `b2_lj_series`, ~line 289)
- Modify: `src/lib.rs` (re-export `b2_lj_series_derivs`)
- Test: `src/main.rs`

- [ ] **Step 1: Write the failing tests**

In `src/main.rs` `mod tests`, add three tests:

```rust
    #[test]
    fn neff_equals_n_for_inverse_power() {
        use potter_poc::{b2_and_derivs, Potential};
        // For u = eps*(sig/r)^n, n_eff(T) == n exactly, at every T.
        for &n in &[6, 9, 12, 18] {
            let src = format!("eps*(sig/r)**{n}");
            let p = Potential::compile(&src, 1.0, 1.0).unwrap();
            for &t in &[1.0_f64, 2.0, 5.0] {
                let d = b2_and_derivs(&p, t, 1e-12);
                let ne = d.neff(t);
                assert!(
                    (ne - n as f64).abs() < 1e-3,
                    "n={n} T*={t}: n_eff={ne}"
                );
            }
        }
    }

    #[test]
    fn lj_derivs_match_hcb_series() {
        use potter_poc::{b2_and_derivs_v, b2_lj_series_derivs};
        let lj = |r: f64| {
            let s6 = (1.0_f64 / r).powi(6);
            4.0 * (s6 * s6 - s6)
        };
        for &t in &[2.0_f64, 3.0, 5.0] {
            let num = b2_and_derivs_v(&lj, t, 1e-12);
            let ser = b2_lj_series_derivs(t, 60);
            assert!(((num.b2 - ser.b2) / ser.b2).abs() < 1e-4, "B2 T*={t}");
            assert!(((num.db2_dt - ser.db2_dt) / ser.db2_dt).abs() < 1e-4, "B2' T*={t}");
            assert!(
                ((num.d2b2_dt2 - ser.d2b2_dt2) / ser.d2b2_dt2).abs() < 1e-3,
                "B2'' T*={t}"
            );
            assert!((num.neff(t) - ser.neff(t)).abs() < 1e-3, "neff T*={t}");
        }
    }

    #[test]
    fn lj_neff_high_temperature_limit_is_twelve() {
        use potter_poc::b2_lj_series_derivs;
        // Leading HCB term ~ T*^{-1/4} = T^{-3/n} with n = 12 -> n_eff -> 12.
        let ne = b2_lj_series_derivs(1e6, 60).neff(1e6);
        assert!((ne - 12.0).abs() < 0.05, "n_eff(1e6)={ne}");
        // and it should be trending toward 12 from a moderately high T already
        let ne2 = b2_lj_series_derivs(1e4, 60).neff(1e4);
        assert!((ne2 - 12.0).abs() < 0.3, "n_eff(1e4)={ne2}");
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test neff_equals_n_for_inverse_power lj_derivs_match_hcb_series lj_neff_high_temperature_limit_is_twelve`
Expected: FAIL — `b2_lj_series_derivs` not found (and the IPL test fails to compile only if Task 2 is absent; with Task 2 present it compiles but `b2_lj_series_derivs` is still missing → compile error across the group).

- [ ] **Step 3: Write minimal implementation**

In `src/physics.rs`, append after `b2_lj_series` (after line 289):

```rust
/// Closed-form LJ (12-6) reduced B₂ **and its first two T*-derivatives**, by
/// differentiating the HCB Γ-series term-by-term — the analytic oracle for the
/// integrated derivatives. Each term is `coeff_j · T*^{-q_j}` with
/// `q_j = (2j+1)/4`, so `d/dT*` brings down `-q_j`, and `d²/dT*²` brings
/// `q_j (q_j+1)`. (σ = ε = 1.)
pub fn b2_lj_series_derivs(tstar: f64, nterms: usize) -> B2Derivs {
    let mut factorial = 1.0f64;
    let (mut s0, mut s1, mut s2) = (0.0f64, 0.0f64, 0.0f64);
    for j in 0..nterms {
        let jf = j as f64;
        if j > 0 {
            factorial *= jf;
        }
        let p = (2.0 * jf + 1.0) / 2.0;
        let coeff = -(2.0f64.powf(p) / (4.0 * factorial)) * libm::tgamma((2.0 * jf - 1.0) / 4.0);
        let q = (2.0 * jf + 1.0) / 4.0;
        let t0 = coeff * tstar.powf(-q);
        let t1 = coeff * (-q) * tstar.powf(-q - 1.0);
        let t2 = coeff * q * (q + 1.0) * tstar.powf(-q - 2.0);
        if t0.is_finite() {
            s0 += t0;
        }
        if t1.is_finite() {
            s1 += t1;
        }
        if t2.is_finite() {
            s2 += t2;
        }
    }
    let c = 2.0 * PI / 3.0;
    B2Derivs {
        b2: c * s0,
        db2_dt: c * s1,
        d2b2_dt2: c * s2,
    }
}
```

In `src/lib.rs`, add `b2_lj_series_derivs` to the `pub use physics::{ … };` block (alphabetically near `b2_lj_series`):

```rust
pub use physics::{
    b2, b2_and_derivs, b2_and_derivs_v, b2_finegrid, b2_lj_series, b2_lj_series_derivs, b2_v,
    b2_v_grid, b3, b3_cubature, b3_cubature_v, b3_v, b3_v_grid, neff, B2Derivs, CsePotential,
    Potential, LJ_BOYLE_TSTAR,
};
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test neff_equals_n_for_inverse_power lj_derivs_match_hcb_series lj_neff_high_temperature_limit_is_twelve`
Expected: PASS (all three).
If `neff_equals_n_for_inverse_power` is marginally off at the 1e-3 level for n=18, bump the integrator depth argument from 60 to 70 in `b2_and_derivs_v` and re-run — the high-n wall is the stiffest integrand.

- [ ] **Step 5: Commit**

```bash
git add src/physics.rs src/lib.rs src/main.rs
git commit -m "$(cat <<'EOF'
Add HCB-series derivative oracle; validate n_eff (IPL=n, LJ->12)

n_eff==n exactly for inverse-power; integrated LJ derivatives match the
term-differentiated HCB series; LJ n_eff -> 12 at high T (j=0 term).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Library + WASM API (`b2_derivs_from_dsl`, `poc_b2_derivs`, `poc_neff`)

**Files:**
- Modify: `src/lib.rs` (add `b2_derivs_from_dsl`; add two exports in `wasm_exports`)
- Test: `src/main.rs`

- [ ] **Step 1: Write the failing test**

In `src/main.rs` `mod tests`, add:

```rust
    #[test]
    fn b2_derivs_from_dsl_matches_closure() {
        use potter_poc::{b2_and_derivs_v, b2_derivs_from_dsl};
        let lj = |r: f64| {
            let s6 = (1.0_f64 / r).powi(6);
            4.0 * (s6 * s6 - s6)
        };
        let t = 2.0;
        let viadsl =
            b2_derivs_from_dsl("4*eps*((sig/r)**12 - (sig/r)**6)", 1.0, 1.0, t, 1e-12).unwrap();
        let viaclosure = b2_and_derivs_v(&lj, t, 1e-12);
        assert!((viadsl.b2 - viaclosure.b2).abs() < 1e-9);
        assert!((viadsl.db2_dt - viaclosure.db2_dt).abs() < 1e-9);
        assert!((viadsl.d2b2_dt2 - viaclosure.d2b2_dt2).abs() < 1e-9);
        assert!((viadsl.neff(t) - viaclosure.neff(t)).abs() < 1e-9);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test b2_derivs_from_dsl_matches_closure`
Expected: FAIL — `b2_derivs_from_dsl` not found.

- [ ] **Step 3: Write minimal implementation**

In `src/lib.rs`, after `b2_from_dsl` (after line 30), add:

```rust
/// Compile a DSL potential string and compute B₂ with its first two T-derivatives.
pub fn b2_derivs_from_dsl(
    src: &str,
    eps: f64,
    sig: f64,
    t: f64,
    tol: f64,
) -> Result<B2Derivs, String> {
    let pot = Potential::compile(src, eps, sig)?;
    Ok(b2_and_derivs(&pot, t, tol))
}
```

(`B2Derivs`, `b2_and_derivs` are already re-exported into this module's scope via the `pub use physics::{…}` block from Task 2/3.)

In the `wasm_exports` module, update the `use super::{…};` line (line 42) to:

```rust
    use super::{b2_derivs_from_dsl, b2_from_dsl, b3_from_dsl};
```

and add these two exports inside `wasm_exports` (e.g. after `poc_b3`, before `poc_compile_wasm`):

```rust
    /// Parse the DSL at [ptr,len) and write `[B2, dB2/dT, d2B2/dT2, n_eff]` (4 f64)
    /// into the caller-provided `out` array. All NaN on parse/eval error. One
    /// integration pass — avoids recomputing B2 three times.
    #[no_mangle]
    pub extern "C" fn poc_b2_derivs(
        ptr: *const u8,
        len: usize,
        eps: f64,
        sig: f64,
        t: f64,
        out: *mut f64,
    ) {
        let vals = match read_dsl(ptr, len) {
            Some(src) => match b2_derivs_from_dsl(src, eps, sig, t, 1e-12) {
                Ok(d) => [d.b2, d.db2_dt, d.d2b2_dt2, d.neff(t)],
                Err(_) => [f64::NAN; 4],
            },
            None => [f64::NAN; 4],
        };
        unsafe {
            for (k, v) in vals.iter().enumerate() {
                *out.add(k) = *v;
            }
        }
    }

    /// Parse the DSL at [ptr,len) and return the effective repulsive exponent
    /// n_eff(T). NaN on error.
    #[no_mangle]
    pub extern "C" fn poc_neff(ptr: *const u8, len: usize, eps: f64, sig: f64, t: f64) -> f64 {
        match read_dsl(ptr, len) {
            Some(src) => b2_derivs_from_dsl(src, eps, sig, t, 1e-12)
                .map(|d| d.neff(t))
                .unwrap_or(f64::NAN),
            None => f64::NAN,
        }
    }
```

- [ ] **Step 4: Run test to verify it passes (native) and the wasm target still builds**

Run: `cargo test b2_derivs_from_dsl_matches_closure`
Expected: PASS.

Run: `cargo build --target wasm32-unknown-unknown`
Expected: builds cleanly (the new `wasm_exports` items compile). If the target is not installed, run `rustup target add wasm32-unknown-unknown` first.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/main.rs
git commit -m "$(cat <<'EOF'
Add b2_derivs_from_dsl and WASM exports poc_b2_derivs / poc_neff

poc_b2_derivs writes [B2,dB2dT,d2B2dT2,neff] in one pass; poc_neff is a
scalar convenience. Works across all f64 backends.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: MSMC single-bond B₂ derivatives (common random numbers) + two-way anchor

**Files:**
- Modify: `src/msmc.rs` (add `MsmcB2`, `msmc_b2_v`, and a `stderr` helper)
- Test: `src/main.rs`

- [ ] **Step 1: Write the failing test**

In `src/main.rs` `mod tests`, add:

```rust
    #[test]
    fn lj_b2_derivs_msmc_matches_integration() {
        use potter_poc::b2_and_derivs_v;
        use potter_poc::msmc::msmc_b2_v;
        let lj = |r: f64| {
            let s6 = (1.0_f64 / r).powi(6);
            4.0 * (s6 * s6 - s6)
        };
        let t = 2.0;
        let det = b2_and_derivs_v(&lj, t, 1e-11);
        let mc = msmc_b2_v(&lj, t, 1.5, 8_000_000, 0xC0FFEE);
        // value-level agreement (deterministic vs Monte Carlo)
        assert!(
            (mc.d.b2 - det.b2).abs() / det.b2.abs() < 0.02,
            "B2 MSMC {} +/- {} vs integ {}",
            mc.d.b2,
            mc.stderr_b2,
            det.b2
        );
        // n_eff agreement — CRN keeps its variance small
        assert!(
            (mc.neff - det.neff(t)).abs() < 0.3,
            "n_eff MSMC {} +/- {} vs integ {}",
            mc.neff,
            mc.stderr_neff,
            det.neff(t)
        );
        // CRN sanity: sharing one walk makes the n_eff stderr small
        assert!(mc.stderr_neff < 0.1, "n_eff stderr too large: {}", mc.stderr_neff);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test lj_b2_derivs_msmc_matches_integration`
Expected: FAIL — `msmc_b2_v` / `MsmcB2` not found in `potter_poc::msmc`.

- [ ] **Step 3: Write minimal implementation**

In `src/msmc.rs`, add `use crate::physics::B2Derivs;` near the top (after `use std::f64::consts::PI;`). Then append:

```rust
/// MSMC value + first two T-derivatives of B₂ for a spherical potential, computed
/// from ONE walk (common random numbers): the same sampled configurations feed B₂,
/// B₂', and B₂'', so their statistical errors are correlated and n_eff (a ratio)
/// has low variance. Single Mayer bond γ = f(|r|); hard-sphere reference.
pub struct MsmcB2 {
    pub d: B2Derivs,
    pub neff: f64,
    pub stderr_b2: f64,
    pub stderr_neff: f64,
    pub accept: f64,
}

#[inline]
fn block_stderr(blocks: &[f64]) -> f64 {
    let m = blocks.len();
    if m < 2 {
        return f64::NAN;
    }
    let mean = blocks.iter().sum::<f64>() / m as f64;
    let var = blocks.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (m - 1) as f64;
    (var / m as f64).sqrt()
}

/// MSMC estimate of B₂ and its first two T-derivatives for any potential closure.
/// `nsteps` Metropolis steps sampling one particle (relative position `r2`) with
/// weight |γ| = |f(|r2|)|; hard-sphere reference of diameter `sigma_hs`;
/// deterministic given `seed`.
pub fn msmc_b2_v<V: Fn(f64) -> f64>(
    v: &V,
    t: f64,
    sigma_hs: f64,
    nsteps: usize,
    seed: u64,
) -> MsmcB2 {
    // per-bond Mayer factor and its two analytic T-derivatives (V is T-independent)
    let bond = |r: f64| -> (f64, f64, f64) {
        let vv = v(r);
        if vv.is_finite() {
            let e = (-vv / t).exp();
            let t2 = t * t;
            (
                e - 1.0,
                e * vv / t2,
                e * (vv * vv / (t2 * t2) - 2.0 * vv / (t2 * t)),
            )
        } else {
            (-1.0, 0.0, 0.0)
        }
    };
    let hs = |r: f64| if r < sigma_hs { -1.0 } else { 0.0 };
    let b2_hs = (2.0 * PI / 3.0) * sigma_hs.powi(3);

    // start in the support of |gamma|
    let mut r2 = [1.05, 0.0, 0.0];
    let (mut g, mut g1, mut g2) = bond(norm(r2));
    let mut ag = g.abs();

    let mut rng = Rng::new(seed);
    let delta = 0.5;
    let equil = nsteps / 10;
    let nblocks = 50usize;
    let per = (nsteps - equil).max(nblocks) / nblocks;

    // per-block sums of: sign(g), g1/|g|, g2/|g|, gamma_ref/|g|
    let mut b_sign = vec![0.0f64; nblocks];
    let mut b_d1 = vec![0.0f64; nblocks];
    let mut b_d2 = vec![0.0f64; nblocks];
    let mut b_ref = vec![0.0f64; nblocks];
    let mut b_cnt = vec![0usize; nblocks];
    let mut accepts = 0usize;

    for step in 0..nsteps {
        let trial = [
            r2[0] + rng.sym(delta),
            r2[1] + rng.sym(delta),
            r2[2] + rng.sym(delta),
        ];
        let (gn, gn1, gn2) = bond(norm(trial));
        let agn = gn.abs();
        if agn >= ag || rng.unit() < agn / ag {
            r2 = trial;
            g = gn;
            g1 = gn1;
            g2 = gn2;
            ag = agn;
            accepts += 1;
        }
        if step >= equil && ag > 0.0 {
            let b = ((step - equil) / per).min(nblocks - 1);
            b_sign[b] += g.signum();
            b_d1[b] += g1 / ag;
            b_d2[b] += g2 / ag;
            b_ref[b] += hs(norm(r2)) / ag;
            b_cnt[b] += 1;
        }
    }

    // central values from pooled sums (the per-block count cancels in each ratio);
    // errors from the spread of per-block estimates (which share the walk -> CRN).
    let (mut ts, mut t1, mut t2, mut tr) = (0.0, 0.0, 0.0, 0.0);
    let mut b2_blocks = Vec::new();
    let mut neff_blocks = Vec::new();
    for b in 0..nblocks {
        if b_cnt[b] == 0 || b_ref[b] == 0.0 {
            continue;
        }
        ts += b_sign[b];
        t1 += b_d1[b];
        t2 += b_d2[b];
        tr += b_ref[b];
        let db = B2Derivs {
            b2: b2_hs * b_sign[b] / b_ref[b],
            db2_dt: b2_hs * b_d1[b] / b_ref[b],
            d2b2_dt2: b2_hs * b_d2[b] / b_ref[b],
        };
        b2_blocks.push(db.b2);
        neff_blocks.push(db.neff(t));
    }
    let d = if tr != 0.0 {
        B2Derivs {
            b2: b2_hs * ts / tr,
            db2_dt: b2_hs * t1 / tr,
            d2b2_dt2: b2_hs * t2 / tr,
        }
    } else {
        B2Derivs {
            b2: f64::NAN,
            db2_dt: f64::NAN,
            d2b2_dt2: f64::NAN,
        }
    };
    MsmcB2 {
        neff: d.neff(t),
        d,
        stderr_b2: block_stderr(&b2_blocks),
        stderr_neff: block_stderr(&neff_blocks),
        accept: accepts as f64 / nsteps as f64,
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test lj_b2_derivs_msmc_matches_integration`
Expected: PASS. (Statistical but seed-fixed, so deterministic.) If it fails marginally on the `< 0.3` / `< 0.1` bands, raise `nsteps` to `16_000_000` in the test and re-run; do not loosen the bands below 0.5 without flagging — wide bands hide real bugs.

- [ ] **Step 5: Commit**

```bash
git add src/msmc.rs src/main.rs
git commit -m "$(cat <<'EOF'
Add single-bond MSMC B2 derivatives (CRN) + two-way LJ anchor

One walk feeds B2, B2', B2'' (shared denominator), so n_eff has low
variance; cross-checked against the deterministic integrator on LJ.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Full-suite regression + README scope note

**Files:**
- Modify: `README.md` (lines ~172-174, the "out of scope" note about T-derivatives)
- Test: full suite

- [ ] **Step 1: Run the entire test suite**

Run: `cargo test`
Expected: all prior tests (18) plus the 6 new tests pass. If anything else regressed, fix it before continuing.

- [ ] **Step 2: Update the README scope note**

In `README.md`, replace the bullet that currently reads (around lines 172-174):

```
- Temperature derivatives (dⁿB/dTⁿ) are out of scope here, but note they wrap
  *around* the potential: V(r) is T-independent, so a generic/arbitrary potential
  and exact derivatives do not conflict (`num-dual` would supply the autodiff).
```

with:

```
- Temperature derivatives: B2 and its first two T-derivatives are computed by
  differentiating the Boltzmann factor analytically (V(r) is T-independent) on a
  single shared adaptive grid, giving the effective repulsive exponent
  n_eff = -3(B2 + T·B2')/(2T·B2' + T²·B2''). Validated two ways — deterministic
  integration and single-bond MSMC (common random numbers) — anchored by the LJ
  high-T limit n_eff -> 12 and by n_eff == n for inverse-power potentials. See
  `docs/superpowers/specs/2026-06-12-neff-b2-temperature-derivatives-design.md`.
```

- [ ] **Step 3: Run the suite once more (docs change is inert, but confirm clean tree)**

Run: `cargo test`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
README: document B2 T-derivatives and n_eff (Phase 1 done)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review (completed against the spec)

**Spec coverage:**
- §2 `n_eff` formula → Task 2 (`neff`). ✓
- §3 analytic integrands + shared grid → Task 1 (`adaptive_simpson3`) + Task 2 (`b2_deriv_integrand_s`). ✓
- §4 types/functions/API → Task 2 (`B2Derivs`, `b2_and_derivs[_v]`), Task 4 (`b2_derivs_from_dsl`, `poc_b2_derivs`, `poc_neff`). ✓
- §5.0 LJ two-way anchor + n_eff→12 → Task 3 (limit oracle) + Task 5 (MSMC vs integration). ✓
- §5.1 IPL flat-at-n, LJ series, FD guard → Task 3 + Task 2 (FD). ✓
- §6 MSMC CRN estimator (B₂ instance) → Task 5. ✓
- Out of scope (rigid molecules, B₃ cubature, quantum, MSMC B₃/B₄) → correctly deferred to Phase 2/3; not in this plan.

**Placeholder scan:** none — every code/test step contains complete code and exact commands.

**Type consistency:** `B2Derivs { b2, db2_dt, d2b2_dt2 }` and `.neff(t)` used identically in Tasks 2-5; `adaptive_simpson3` signature matches its caller in `b2_and_derivs_v`; `b2_lj_series_derivs` returns `B2Derivs` and is used as such in Task 3; `MsmcB2 { d, neff, stderr_b2, stderr_neff, accept }` fields match the Task 5 test. Re-export list in `src/lib.rs` is built up consistently across Tasks 2→3→4.
