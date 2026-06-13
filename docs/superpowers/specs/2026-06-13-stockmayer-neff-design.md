# Stockmayer (12-6-3) n_eff — design

- **Date:** 2026-06-13
- **Status:** Approved (ready for implementation plan)
- **Context:** The browser app is becoming a companion to **Bell, *J. Chem. Phys.* 152, 164508 (2020)** ("Effective hardness of interaction…"), reproducing the effective IPL exponent `n_eff` for that paper's Section-III model potentials. LJ/Mie are done; this spec adds **Stockmayer**, the first orientation-dependent one. EXP and square-well (also in the paper) are easier later add-ons (see §9).

## 1. Goal

Compute `n_eff(T*)` for the Stockmayer 12-6-3 potential as a function of reduced dipole strength `(μ*)²`, and surface it in the web app — reproducing the paper's Fig. 6 behavior (stronger dipole → lower `n_eff` peak; `n_eff → 12` at high T*).

## 2. Potential & units (paper Eqs. 47-48)

Reduced units (ε = σ = 1), single parameter `(μ*)²`:

```
V*(r*, θ1, θ2, φ) = 4[(1/r*)^12 − (1/r*)^6]
                    − (μ*)² (1/r*)^3 [ 2 cosθ1 cosθ2 − sinθ1 sinθ2 cosφ ]
```

`(μ*)² = 0` recovers Lennard-Jones exactly. In real units the dipole term is
`ε(μ*)²(σ/r)³·[…]`; the web's ε/σ scale T* and B₂ via the same chain rule as LJ
(B₂ = σ³·B₂*(T/ε), dB₂/dT = σ³·(dB₂*/dT*)/ε, …), and `(μ*)²` is a dimensionless
input. `n_eff` is scale-invariant, so it is computed directly on the reduced result.

## 3. n_eff definition (unchanged from Phase 1 / paper Eq. 11)

`n_eff = -3·(B₂ + T·B₂') / (2T·B₂' + T²·B₂'')` — already in `physics.rs::neff`
(`B2Derivs::neff`). Stockmayer reuses it; only the B₂/B₂'/B₂'' inputs are new.

## 4. Rust core

### 4.1 `Stockmayer` struct + energy (`src/molecule.rs`)
```rust
pub struct Stockmayer { pub eps: f64, pub sig: f64, pub mu2: f64 } // mu2 = (μ*)²

impl Stockmayer {
    fn energy(&self, r: f64, th1: f64, th2: f64, phi: f64) -> f64 {
        let sr = self.sig / r;
        let sr3 = sr * sr * sr;
        let sr6 = sr3 * sr3;
        let lj = 4.0 * self.eps * (sr6 * sr6 - sr6);
        let ang = 2.0 * th1.cos() * th2.cos() - th1.sin() * th2.sin() * phi.cos();
        lj - self.eps * self.mu2 * sr3 * ang
    }
}
```
No `rmin` cutoff (rmin = 0): the LJ r⁻¹² core bounds the well; the dipole r⁻³ rides
on it, so there is no Coulomb-style catastrophe.

### 4.2 Shared-grid vector cubature (`src/cubature.rs`)
Generalize the Genz-Malik adaptive cubature to a fixed `[f64; 3]` vector integrand so
B₂, B₂', B₂'' are integrated on **one shared region subdivision** (the 4-D analog of
`adaptive_simpson3`; keeps the component errors coherent for the `n_eff` ratio — the
Phase-1 principle).

```rust
pub fn hcubature3<F: Fn(&[f64]) -> [f64; 3]>(
    dim: usize, f: &F, a: &[f64], b: &[f64],
    abstol: f64, reltol: f64, maxevals: usize,
) -> ([f64; 3], [f64; 3], usize) // (value[3], err_estimate[3], nevals)
```
- The degree-7/degree-5 `rule` is generalized to evaluate the `[f64;3]` integrand at
  the same nodes and return per-component value and error.
- A region's heap priority is its **max-component error**; the loop continues while
  **any** component exceeds `abstol + reltol·|value_k|`. Same nodes → coherent errors.
- The existing scalar `hcubature` stays (B₃ cubature et al. use it); `hcubature3` is
  additive. (Implementation may express one in terms of the other or share `rule`
  internals — implementer's choice — but must not regress existing tests.)

### 4.3 Vector orientational integrand + reduced derivatives (`src/molecule.rs`)
A vector analog of `b2_orientational` returning reduced `B2Derivs` (σ³ units). The
integrand differentiates `e^{-U/T}` analytically (U is T-independent), mirroring
`physics.rs::b2_deriv_integrand_s`:

```rust
// per sample x = [s, th1, th2, phi]; r = s/(1-s); w = r^2 * (1/(1-s)^2) * sin θ1 * sin θ2
// f0 = e^{-U/T} - 1
// f1 = e^{-U/T} · U/T²
// f2 = e^{-U/T} · (U²/T⁴ − 2U/T³)
// non-finite U (none for Stockmayer): (f0,f1,f2) = (-1,0,0). Each × w, NaN-scrubbed.
```
Integrate over `[0,1]×[0,π]×[0,π]×[0,2π]` (rmin = 0, so s_lo = 0) with `hcubature3`,
then apply the **same `-0.25` normalization** the scalar `b2_orientational` uses (the
factor validated by the single-site→spherical limit), per component:

```rust
impl Stockmayer {
    /// Reduced B2 and its first two T*-derivatives (σ³ units; ε=σ=1 for the web).
    pub fn b2_and_derivs(&self, t: f64, reltol: f64) -> (B2Derivs, usize) {
        // bounds: lo = [0,0,0,0], hi = [1, π, π, 2π] (rmin = 0 -> s_lo = 0)
        let (i, _e, nev) = hcubature3(4, &integrand, &lo, &hi, ABSTOL, reltol, MAXEVAL);
        (B2Derivs { b2: -0.25*i[0], db2_dt: -0.25*i[1], d2b2_dt2: -0.25*i[2] }, nev)
    }
}
```
Tolerances (reduced units differ from the cm³/mol `b2_orientational`): the integral `i`
is O(1) in σ³, so use `ABSTOL ≈ 1e-3` (σ³) as the absolute floor — applied per component,
so the small B₂' / B₂'' integrands also stop on absolute rather than relative error near
their zeros — `reltol` from the caller (web passes ~1e-3), `MAXEVAL ≈ 5_000_000`. The
abstol is a *per-component* floor in `hcubature3`'s stop test `err_k ≤ abstol + reltol·|val_k|`.
Note: this returns **reduced** B₂ (σ³), *not* the cm³/mol of the existing molecular
`b2()` — it omits `ANG3_TO_CM3MOL`. `B2Derivs::neff(t)` gives `n_eff`.

## 5. WASM export (`src/lib.rs`)
```rust
/// Reduced Stockmayer (ε=σ=1): write [B2*, dB2*/dT*, d2B2*/dT*2, n_eff] into out[4].
#[no_mangle]
pub extern "C" fn poc_stockmayer(tstar: f64, mu2: f64, reltol: f64, out: *mut f64)
```
Builds `Stockmayer { eps:1.0, sig:1.0, mu2 }`, calls `b2_and_derivs(tstar, reltol)`,
writes `[d.b2, d.db2_dt, d.d2b2_dt2, d.neff(tstar)]` (NaN-fill on a non-finite result).

## 6. Web app (`web/index.html.in`)

### 6.1 Stockmayer preset
- New preset **"Stockmayer (12-6-3)"** with a **`(μ*)²`** number field (`#mu2`, default
  `2`, min `0`, reactive like n/m). Show `(μ*)²`; hide n/m. The DSL textarea shows the
  formula read-only (informational — Stockmayer is not a DSL `V(r)`).
- `compute()` branches: for the Stockmayer preset, **do not** use the DSL/AOT/JS path;
  call `poc_stockmayer(T/eps, mu2, reltol, outptr)` per temperature (read 4 f64 back
  from wasm memory via an allocated scratch buffer), then scale: `b2 = σ³·B2*`,
  `db2 = σ³·dB2*/dT*/eps`, `d2b2 = σ³·d2B2*/dT*²/eps²`, `neff = neff*` (scale-invariant).
- No exact-overlay dots (the cubature is the result). Keep the dashed `n = 12` high-T
  reference on the `n_eff` panel.

### 6.2 Incremental, cancellable runner (applies to ALL presets)
Replace synchronous `compute()` with a **time-sliced async** runner so the 4-D sweep
fills in instead of freezing the page:
- Loop temperatures, computing one row at a time; append each table row immediately and
  redraw the plot as points accrue (so it visibly draws).
- **Yield on a time budget** (~16 ms per tick), not per row: cheap presets (LJ/Mie/inv,
  1-D JS integrals) finish in one tick and stay instant; Stockmayer fills progressively.
- **Generation token**: each run captures a counter; a row/plot/finish step aborts if the
  token is stale (so reactive edits mid-sweep cancel the in-flight run instead of piling
  up overlapping 4-D sweeps).
- **Status** shows "computing k/N…" during the sweep, then the normal summary.
- A single temperature's 4-D `poc_stockmayer` call is the only unavoidable block (one
  wasm call can't be interrupted) — bounded, not a full freeze. Kept small via a web
  `reltol ≈ 1e-3` and a maxeval cap. A Web Worker (off-thread cubature) is the clean
  upgrade if a single point still janks — out of scope here (§9).

## 7. Validation

1. **(μ*)² = 0 → Lennard-Jones (exact, primary test).** `Stockmayer { mu2: 0 }`
   `b2_and_derivs(t*)` matches the spherical `physics::b2_and_derivs_v(LJ, t*)` for
   `b2`, `db2_dt`, `d2b2_dt2`, and `neff` to cubature tolerance (~1e-2 relative, matching
   the existing single-site→spherical test). This pins the angular normalization and the
   derivative integrands at once.
2. **High-T limit.** `n_eff → 12` as T* grows, for several `(μ*)²` (the dipole averages
   out; repulsion dominates).
3. **Dipole effect (paper Fig. 6).** At fixed low-to-moderate T*, increasing `(μ*)²`
   makes B₂ more negative (stronger attraction) and **lowers the `n_eff` peak**. A test
   asserts B₂(μ*²=4) < B₂(μ*²=0) at a low T*, and that the `n_eff` peak height decreases
   monotonically across `(μ*)² = 0, 2, 4`.
4. **Web end-to-end (node).** Mirror the existing node harness: instantiate the built
   wasm, call `poc_stockmayer` over a T* grid for `(μ*)² = 0` and `2`, confirm the
   `(μ*)²=0` curve equals the LJ numeric `n_eff` and the `(μ*)²=2` curve sits below it.

## 8. Out of scope (this spec)

- **EXP and square-well** potentials. EXP is a spherical DSL potential already
  (`φ₀·exp(−r/r₀)`) — needs only a preset + Sherwood-Mason exact overlay. Square-well has
  closed-form `n_eff` (paper Eqs. 24-26). Both are web-only, no 4-D; separate small specs.
- **Web Worker** off-thread cubature (the incremental runner is enough for now).
- **Fig. 6 multi-curve family** overlay (single `(μ*)²` at a time, per the chosen UX).
- **Rigid-molecule (Linear/RigidLinear) T-derivatives** generally — `hcubature3` and the
  vector orientational integrand are built here for Stockmayer; wiring them into the other
  molecular constructors is a later step.
- Quantum corrections.

## 9. File-by-file change list

- `src/cubature.rs` — add `hcubature3` (fixed `[f64;3]` vector Genz-Malik cubature,
  shared subdivision, max-component error criterion); generalize/extract `rule` as needed.
- `src/molecule.rs` — `Stockmayer` struct + `energy`; vector orientational integrand;
  `Stockmayer::b2_and_derivs` (reduced σ³ `B2Derivs`); a constructor/default if useful.
- `src/lib.rs` — `poc_stockmayer` wasm export.
- `src/main.rs` (`#[cfg(test)] mod tests`) — tests 1-3 of §7.
- `web/index.html.in` — Stockmayer preset + `(μ*)²` field; `compute()` Stockmayer branch
  (wasm call + scaling); time-sliced incremental cancellable runner; status updates.
- `web/build.sh` regenerates `docs/index.html` (embeds the rebuilt wasm with the new
  export).
- node verification script for §7.4 (under `node/` or `/tmp` during dev).
