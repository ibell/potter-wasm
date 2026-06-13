# n_eff and second temperature derivatives of B₂ — design

- **Date:** 2026-06-12
- **Status:** Approved (ready for implementation plan)
- **Scope of this round:** classical B₂ for spherical / DSL potentials, all backends. Forward-looking design for the rigid-molecule, B₃ cubature, and MSMC paths is included but phased.

## 1. Goal

Compute the **effective repulsive exponent** `n_eff(T)` — the thing potter was really built to get at. This requires the **first and second temperature derivatives** of the second virial coefficient, which the codebase currently does not provide at all (B₂ returns a bare `f64`; T-derivatives were explicitly out of scope in the README).

## 2. Definition (verbatim from potter)

From `potter/squarewell.py` (`neff_B_att`, SymPy derivation in its docstring):

```
neff = -3*(B2 + T*diff(B2,T)) / (2*T*diff(B2,T) + T**2*diff(B2,T,2))
```

i.e.

```
            -3 ( B₂ + T·B₂' )
n_eff(T) = --------------------
           2T·B₂' + T²·B₂''
```

**Interpretation.** Let 𝔅(T) ≡ B₂ + T·B₂' = d(T·B₂)/dT — the same `B + T dB/dT`
combination that appears in potter's `get_B_plus_TdBdT` (Mie) and `frakBn`
(Hellmann) reduced-transport expressions. Then the numerator is `-3·𝔅` and the
denominator is `T·𝔅'`, so

```
n_eff = -3·𝔅 / (T·𝔅') = -3 / (d ln 𝔅 / d ln T)
```

— the effective inverse-power-law (IPL) exponent *of* 𝔅.

**Sanity check (IPL recovers n exactly).** For `u = ε(σ/r)ⁿ`, `B₂ = C·T^(−3/n)`:
numerator `= -3·C·T^(−3/n)·(1 − 3/n)`, denominator `= C·(−3/n)·T^(−3/n)·(1 − 3/n)`,
so `n_eff = −3 / (−3/n) = n` at **every** temperature. This is both a correctness
property and a built-in validation (flat line at `n`).

`n_eff` is a dimensionless ratio: `T·B₂'` and `T²·B₂''` carry the units of B₂, which
cancel. So it is identical whether computed in reduced or real units.

## 3. Method: analytic differentiation of the Boltzmann factor

`B₂(T) = -2π ∫₀^∞ (e^{-V/T} - 1) r² dr`, and crucially **V(r) is T-independent**.
So the integrand's T-derivatives are exact and need only the scalar `V(r)` that
every backend already provides:

| quantity | integrand (× r², before the −2π factor) |
|---|---|
| B₂   | `e^{-V/T} − 1`                       |
| B₂'  | `e^{-V/T} · (V/T²)`                  |
| B₂'' | `e^{-V/T} · (V²/T⁴ − 2V/T³)`         |

The constant `−1` is T-independent and correctly drops from the derivative
integrands. At the repulsive wall `V → ∞`, `e^{-V/T} → 0` dominates `V`, `V²`, so
all three integrands → 0 — same well-behaved decay as the Mayer function, no new
singularity handling. At large r, `V → 0` and `V/T²`, `V²/T⁴` → 0.

### Why analytic, not autodiff / complex-step

potter computes derivatives via complex-/multicomplex-step (`diff_mcx1`, complex T)
— autodiff wrapped around the T-independent potential. We deliberately **do not**
replicate that, because:

1. **Backend portability.** Our JIT (cranelift), `fasteval`, and AOT-wasm backends
   are f64-specialized. Threading a complex/dual `T` through them would require
   re-plumbing each. The analytic integrands only ever ask the potential for
   `v(r): f64`, so they work **unchanged** across tree-walk, CSE, JIT, fasteval,
   and AOT-wasm.
2. **Exactness.** Analytic derivatives have no step size `h` and no subtractive
   cancellation. (The quantum-correction work already showed FD derivatives
   failing on cancellation where analytic derivatives fixed it.)

### Shared-grid requirement (the correlation point)

The value and its derivatives **must be evaluated on one common quadrature grid**.
What gives complex-/multicomplex-step its consistent error between B₂ and its
derivatives is *node-sharing* (the dual number rides the same adaptive
subdivision), not autodiff per se. If we instead ran three *independent* adaptive
integrations, each would adapt to its own integrand shape, landing on three
different subdivisions with independent truncation errors — and the partial
cancellation we want when forming the `n_eff` ratio would be lost.

**Therefore:** a single adaptive pass with a **vector-valued integrand**
`[f₀, f₁, f₂]`, accumulated on one shared subdivision, with the refinement
criterion driven by **all three** components (the B₂'' integrand peaks harder at
the wall due to the extra `V²`, so it must vote on where to refine — otherwise it
is under-resolved). Same node set → coherent error → correct correlations into
`n_eff`. This captures the autodiff correlation property with none of the autodiff
plumbing, and is strictly better than complex-step (exact, single pass).

## 4. Core implementation (spherical, Phase 1)

### `src/integrate.rs` — vector adaptive Simpson
Add a fixed-width vector variant of the existing adaptive Simpson that carries a
`[f64; 3]` accumulator. The local error test refines a panel while **any**
component exceeds its (relative) tolerance, using a per-component running scale so
the components are controlled independently but on a **shared** subdivision. The
existing scalar integrator stays for B₂-only / B₃ callers (or is expressed as the
N=1 case — implementer's choice; do not regress existing tests).

### `src/physics.rs` — integrands, B₂Derivs, n_eff
- A vector integrand closure returning `[mayer, e^{-V/T}·V/T², e^{-V/T}·(V²/T⁴ − 2V/T³)]`
  with the same `s → r` domain map and Jacobian as `b2_integrand_s`.
- `pub struct B2Derivs { pub b2: f64, pub db2_dt: f64, pub d2b2_dt2: f64 }`
- `pub fn b2_and_derivs_v<V: Fn(f64)->f64>(v: &V, t: f64, tol: f64) -> B2Derivs`
  (one vector adaptive pass, then multiply each component by −2π).
- `pub fn b2_and_derivs(pot: &Potential, t: f64, tol: f64) -> B2Derivs` wrapper.
- `pub fn neff(d: &B2Derivs, t: f64) -> f64`
  = `-3.0*(d.b2 + t*d.db2_dt) / (2.0*t*d.db2_dt + t*t*d.d2b2_dt2)` — potter's
  formula verbatim. (A method `B2Derivs::neff(&self, t)` is equivalent; pick one.)

### `src/lib.rs` + WASM API
- `pub fn b2_derivs_from_dsl(src, eps, sig, t, tol) -> Result<B2Derivs, String>`.
- Batch export `poc_b2_derivs(ptr, len, eps, sig, t, out_ptr)` that writes
  `[B2, dB2dT, d2B2dT2, neff]` into a caller-provided 4×f64 array — avoids
  recomputing B₂ three times. NaN-fills on parse error.
- Convenience scalar `poc_neff(ptr, len, eps, sig, t) -> f64`.
- Browser/node story: `n_eff(T)` overlays naturally on the existing inverse-power
  and LJ presets; for IPL it is a flat line at `n` (live correctness check).

## 5. Validation (uses closed forms already in the tree — no new data)

### 5.0 Anchor case: single-site Lennard-Jones, two independent ways
The centerpiece of Phase 1 is LJ B₂ + `B₂'` + `B₂''` + `n_eff` computed by **two
independent methods that must agree**:

- **(a) Integration** — the deterministic vector adaptive Simpson of §4.
- **(b) MSMC** — Mayer sampling of the single bond `f₁₂` with the CRN derivative
  tallies of §6 (this is the B₂ instance of the MSMC estimator, the simplest
  cluster — one bond — and so the natural place to validate the Monte-Carlo
  derivative machinery against a known answer before B₃/B₄).

They are cross-checked against each other and against the closed forms below, with
the **high-T limit `n_eff → 12`** as the shared physical anchor both must hit.

**Why 12 is exact in the limit:** in the HCB series `B₂ ~ Σ_j c_j T*^{-(2j+1)/4}`,
the `j = 0` term `∝ T*^{-1/4}` dominates as `T* → ∞`. `T*^{-1/4}` is the `T^{-3/n}`
scaling with `n = 12` — the LJ repulsive exponent — so `n_eff → 12`. (Method (b)
must reproduce this to within its sampling error; method (a) to integrator
tolerance.)

### 5.1 Supporting oracles
1. **IPL `ε(σ/r)ⁿ`:** `n_eff(T) == n` to integrator tolerance at several T and
   several n (e.g. n = 6, 9, 12, 18). Strongest deterministic test; the exact
   Γ-function B₂ is already implemented.
2. **LJ analytic oracle:** differentiate `b2_lj_series` (HCB Γ-series) term-by-term.
   With `p_j = (2j+1)/4`:
   `B₂' = Σ c_j·(−p_j)·T*^{−p_j−1}`, `B₂'' = Σ c_j·(−p_j)(−p_j−1)·T*^{−p_j−2}`.
   Assert the integrated `db2_dt`, `d2b2_dt2` match this series (loosely at low
   T*, tightly for T* ≥ 2 where the series converges, mirroring the existing B₂
   series test). MSMC agrees within ~1σ.
3. **Cross-check (oracle only):** central finite difference of `b2()` at T±h
   agrees with the analytic `db2_dt` to FD accuracy — a guard, not a primary check.

## 6. Mapping to Monte Carlo (MSMC) — design

This is where the correlation argument becomes a genuine **variance**-reduction
lever via **common random numbers (CRN)**: estimate the value and both derivatives
from the *same* sample stream. The **B₂ instance (single bond, `γ = f₁₂`) is built
in Phase 1** as the validation vehicle (§5.0); B₃/B₄ (multi-bond clusters) follow
in Phase 3 with the identical estimator.

### Setup
MSMC (Kofke–Singer, `src/msmc.rs`) writes a cluster integral as
`B_n(T) = C · ∫ γ(T; r) dr` over a **T-independent** configuration domain, where
`γ` is the biconnected-graph integrand (for B₃, `γ = f₁₂ f₁₃ f₂₃`, with
`f_ij = e^{-V_ij/T} − 1`). It samples configurations from `π ∝ |γ(T)|` and uses an
overlap/ratio estimator against a known reference (hard sphere):
`B₃ = B₃_HS · ⟨sgn γ⟩ / ⟨γ_ref/|γ|⟩`.

### Key observation
Because the **domain is T-independent**, the temperature derivatives act only on
the integrand:
```
dB_n/dT  = C · ∫ ∂_T γ  dr
d²B_n/dT² = C · ∫ ∂²_T γ dr
```
These are just two more cluster integrals over the *same* configuration space, and
the importance-sampling identity gives, for any integrand `h`:
```
∫ h dr = (∫ γ_ref dr) · ⟨ h/|γ| ⟩_π / ⟨ γ_ref/|γ| ⟩_π
```
(`π ∝ |γ|` being T-dependent is fine — we are not differentiating the estimator;
we importance-sample three T-parameterised integrals, all at the same T, from one
π. The T-dependence of π affects only efficiency, not correctness.)

### Estimator (one walk, four tallies, shared denominator)
On the **same** sample stream that already produces `sgn γ = γ/|γ|` and
`γ_ref/|γ|`, accumulate two additional per-sample tallies:
```
A_ref = ⟨ γ_ref / |γ| ⟩          (denominator; already computed)
A₀    = ⟨ γ      / |γ| ⟩ = ⟨sgn γ⟩
A₁    = ⟨ ∂_T γ  / |γ| ⟩
A₂    = ⟨ ∂²_T γ / |γ| ⟩
```
then
```
B_n      = B_n_ref · A₀/A_ref
dB_n/dT  = B_n_ref · A₁/A_ref
d²B_n/dT² = B_n_ref · A₂/A_ref
```
Because A₀, A₁, A₂ share the walk (and the denominator), their MC errors are
strongly correlated, so the `n_eff` ratio has far lower variance than estimating
the three from independent runs. This is the CRN payoff.

### Computing ∂_T γ, ∂²_T γ (product rule, not log-derivative)
Per bond, reuse the same analytic Boltzmann derivatives as B₂:
`∂_T f_ij = e^{-V_ij/T}·V_ij/T²`, `∂²_T f_ij = e^{-V_ij/T}·(V_ij²/T⁴ − 2V_ij/T³)`.
Accumulate the cluster derivative by the **direct product rule** (for B₃, 3 bonds):
```
∂_T γ  = Σ_k (∂_T f_k) ∏_{j≠k} f_j
∂²_T γ = Σ_k (∂²_T f_k) ∏_{j≠k} f_j  +  Σ_{k<l} 2(∂_T f_k)(∂_T f_l) ∏_{j≠k,l} f_j
```
Use the direct form (not `γ·Σ ∂_T f/f`) to stay robust where a bond `f_ij → 0`
(there `γ → 0` but the log form would divide by zero). `V_ij` is already evaluated
to form `f_ij`, so the derivatives are nearly free. Validation: MSMC `dB₃/dT` must
agree (within ~1σ) with the cubature `dB₃/dT` below, reference-independent.

## 7. Mapping to B₃ cubature — design, Phase 2

The Genz–Malik / nested-adaptive B₃ integrators (`src/cubature.rs`, `physics.rs`)
get the identical treatment as B₂: the region integrand returns
`[γ, ∂_T γ, ∂²_T γ]` (product rule as above) and the adaptive rule accumulates all
three on **shared** regions, so the derivatives inherit the value's subdivision.
Then `B₃Derivs` + the same `neff` formula (with B₃ in place of B₂ if a B₃-based
hardness is wanted; the primary `n_eff` target is B₂).

## 8. Rigid molecules — design, Phase 2

The 4-D orientational cubature (`molecule.rs`, classical `b2_orientational`) uses
the same analytic trick on `e^{-U/T}` (U the orientation-dependent pair energy):
return `[mayer, e^{-U/T}·U/T², e^{-U/T}·(U²/T⁴ − 2U/T³)]` on the shared 4-D grid.
The analytic hard-core contribution `(2π/3)·rmin³` is T-independent, so it adds to
B₂ only (its T-derivatives are 0).

## 9. Out of scope

- **Quantum-corrected B₂ (WK / QFH).** The QFH *effective* potential is itself
  T-dependent (`U + (ħ²/24kT)·bracket`), and the WK correction term carries explicit
  `1/T²`, so the "V is T-independent" shortcut breaks — they would need extra
  derivative terms. Excluded this round; revisit only if quantum `n_eff` is wanted.
- Derivative orders beyond 2 (not needed for `n_eff`).

## 10. Phasing

- **Phase 1 (this round):** single-site LJ as the anchor (§5.0) computed **two
  ways** — (a) deterministic vector adaptive Simpson and (b) MSMC single-bond with
  CRN derivative tallies — plus spherical/DSL B₂ + `n_eff` across all backends,
  WASM exports, and the IPL + LJ-series validation. Both methods must hit
  `n_eff → 12` at high T. This is the browser story and validates the MC derivative
  estimator on a known answer.
- **Phase 2:** rigid-molecule classical B₂ derivatives; B₃ cubature derivatives.
- **Phase 3:** MSMC derivatives for B₃/B₄ (same CRN estimator, multi-bond clusters).

## 11. File-by-file change list (Phase 1)

- `src/integrate.rs` — vector (`[f64; 3]`) adaptive Simpson with shared-subdivision,
  all-component refinement criterion.
- `src/physics.rs` — vector B₂ integrand; `B2Derivs`; `b2_and_derivs_v`,
  `b2_and_derivs`; `neff`; analytic `b2_lj_series` derivative oracles (test helpers).
- `src/msmc.rs` — single-bond (B₂) Mayer-sampling path that accumulates the
  `A_ref, A₀, A₁, A₂` tallies of §6 on one walk; returns `B2Derivs` + `n_eff` with
  per-quantity stderr.
- `src/lib.rs` — `b2_derivs_from_dsl`; WASM exports `poc_b2_derivs`, `poc_neff`.
- Tests — **LJ two-way agreement (integration vs MSMC) + `n_eff → 12` anchor**;
  IPL flat-at-n; LJ-series derivative match; FD guard.
- (Optional now) `web/` — `n_eff(T)` overlay on existing presets.
