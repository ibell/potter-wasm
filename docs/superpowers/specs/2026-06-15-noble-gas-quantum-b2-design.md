# Noble-gas B₂ with Wigner–Kirkwood quantum corrections (Ne–Xe) — design

- **Date:** 2026-06-15
- **Status:** Approved (ready for implementation plan)
- **Reference implementation:** `potter/integrate_potentials.py` (from the SI of Bell, *J. Chem. Phys.* 152, 164508 (2020)). This Rust feature reproduces that script: the Tang–Toennies (TT) potentials for Ne/Ar/Kr/Xe, classical B₂ + the Wigner–Kirkwood (WK) series to 3rd order, and `n_eff` — in real units (cm³/mol), validated against the script's own numbers.
- **He is out of scope** (the ħ² WK expansion diverges for He; it needs the full phase-shift / Beth–Uhlenbeck route — a separate later feature).

## 1. Goal

A self-contained Rust module computing, for the noble gases Ne, Ar, Kr, Xe:
- the classical second virial coefficient B₂(T) in cm³/mol,
- the WK quantum-corrected B₂ to selectable order 0/1/2/3,
- and the effective IPL exponent `n_eff(T)` (paper Eq. 11),

reproducing `integrate_potentials.py`. Spatial potential derivatives (V′, V″, V‴) and the temperature derivatives of B₂ are both obtained by **automatic differentiation** (`num-dual`), not hand-coded.

## 2. The Tang–Toennies potential

Native units: **R in nm, V/k_B in K** (matching the reference, so validation is direct).

```
V_TT(R) = A·exp(a1·R + a2·R² + an1/R + an2/R²)
          − Σ_{n=3}^{nmax} C[2n] · [ 1 − e^{−bR} Σ_{k=0}^{2n} (bR)^k / k! ] / R^{2n}
```

Short-range correction: for `R < Rcutoff·Repsilon`, replace V_TT with the fitted form
```
V(R) = tildeA/R · exp(−tilde_a·R)
```
(This is potter's small-separation fix; `tilde` params are fit to match value+derivative at the cutoff, so V is C¹ there. The C² jump at the cutoff is at a single point — negligible in the integral.)

### 2.1 Parameters (verbatim from `integrate_potentials.py`)

All `A, C[2n]` in K·nmⁿ-consistent units; `a*, b, tilde_a` in 1/nm-consistent units; `mass_rel` in amu; `Repsilon` in nm.

**Neon** (`nmax=8`): `A=0.402915058383e8, a1=-0.428654039586e2, a2=-0.333818674327e1, an1=-0.534644860719e-1, an2=0.501774999419e-2, b=0.492438731676e2`; `C={6:0.440676750157e-1, 8:0.164892507701e-2, 10:0.790473640524e-4, 12:0.485489170103e-5, 14:0.382012334054e-6, 16:0.385106552963e-7}`; `tildeA=2.36770343e6, tilde_a=3.93124973e1, Rcutoff=0.4, Repsilon=0.30894556, mass_rel=20.1797`.

**Argon** (`nmax=8`): `A=4.61330146e7, a1=-2.98337630e1, a2=-9.71208881, an1=2.75206827e-2, an2=-1.01489050e-2, b=4.02517211e1`; `C={6:4.42812017e-1, 8:3.26707684e-2, 10:2.45656537e-3, 12:1.88246247e-4, 14:1.47012192e-5, 16:1.17006343e-6}`; `tildeA=9.36167467e5, tilde_a=2.15969557e1, Rcutoff=0.4, Repsilon=0.376182, mass_rel=39.948`.

**Krypton** (`nmax=8`, `C12/14/16` via `add_recursive`): `A=0.3200711798e8, a1=-0.2430565544e1*10, a2=-0.1435536209*1e2, an1=-0.4532273868/10, an2=0, b=0.2786344368e1*10`; `C={6:0.8992209265e6/1e6, 8:0.7316713603e7/1e8, 10:0.7835488511e8/1e10}` then `add_recursive`; `tildeA=0.8268005465e7/10, tilde_a=0.1682493666e1*10, Rcutoff=0.3, Repsilon=4.015802/10, mass_rel=83.798`.

**Xenon** (`nmax=8`, `C12/14/16` via `add_recursive`): `A=0.579317071e8, a1=-0.208311994e1*10, a2=-0.147746919*1e2, an1=-0.289687722e1/10, an2=0.258976595e1/1e2, b=0.244337880e1*10`; `C={6:0.200298034e7/1e6, 8:0.199130481e8/1e8, 10:0.286841040e9/1e10}` then `add_recursive`; `tildeA=4.18081481e6, tilde_a=2.38954061e1, Rcutoff=0.3, Repsilon=4.37798/10, mass_rel=131.293`.

`add_recursive`: for `n in {6,7,8}` set `C[2n] = C[2n-6]·(C[2n-2]/C[2n-4])³` (fills C12, C14, C16 from C6, C8, C10).

### 2.2 Constructors
`pub fn neon_tt() -> TangToennies`, `argon_tt()`, `krypton_tt()`, `xenon_tt()` (mirror `molecule.rs::n2_hellmann()`), each building the struct with the params above (Kr/Xe applying `add_recursive`).

## 3. Derivatives via `num-dual`

Add `num-dual` to `Cargo.toml`. The potential **value** function is generic over the dual scalar:
```rust
fn v_tt<D: DualNum<f64> + Copy>(&self, r: D) -> D   // V_TT(R), R in nm -> V/k_B in K
```
(The damping `Σ (bR)^k/k!` uses real factorials/constants; `exp`, powers, division come from `DualNum`.) The short-range `tilde` branch is selected on the real value of R.

- **Spatial derivatives**: `num_dual::third_derivative(|r| self.v_full(r), R)` → `(V, V′, V″, V‴)` in K/nmᵏ. (Hand-coded `potprime*` from the reference are NOT ported — they are the validation oracle, §6.)
- These are **T-independent**, so they are computed **once** per grid point (§5).

## 4. WK-series integrand (the reference `get_integrand`)

Work in **SI** for the integrand. Convert the K/nmᵏ derivatives:
`V[J]=V_K·k_B`, `V′[J/m]=V′·k_B·1e9`, `V″[J/m²]=V″·k_B·1e18`, `V‴[J/m³]=V‴·k_B·1e27`, `R[m]=R_nm·1e−9`.

Constants: `k_B=1.380649e-23`, `ħ=1.054571817e-34`, `u=1.66053906660e-27`, `N_A=8.314462618/k_B`.
Per fluid: `m = mass_rel·u` [kg], `β = 1/(k_B·T)` [1/J], `λ = ħ²β/(12 m)` [m²].

Let `p = βV′`, `p2 = βV″`, `p3 = βV‴`, `e = e^{−βV}`. Integrand (R in m), order ≤ requested:
```
g(R) = R² [ −(e − 1)                                                  # order 0 (classical)
          + λ·e·p²                                                    # order 1
          − λ²·e·( 6/5·p2² + 12/(5R²)·p² + 4/(3R)·p³ − 1/6·p⁴ )       # order 2
          + λ³·e·( 36/35·p3² + 216/(35R²)·p2² + 24/21·p2³
                  + 24/(5R)·p·p2² + 288/(315R³)·p³
                  − 6/5·p²·p2² − 2/(15R²)·p⁴ − 2/(5R)·p⁵ + 1/30·p⁶ )  # order 3
        ]
B₂(T) = 2π·N_A·∫ g(R) dR        [m³/mol]   →  ×1e6  [cm³/mol]
```
`order ∈ {0,1,2,3}` selects how many quantum terms are included (0 = classical).

## 5. Integration (fixed log grid)

Mirror the reference for direct comparability and to make the dual-T pass trivial:
- Grid: `R_nm` = `logspace(Rmin, Rmax, N)`, `Rmin = 0.01·Repsilon`, `Rmax = 1e4·Repsilon`, `N ≈ 10000`.
- **Precompute once** (T-independent): `R[m]`, `V[J]`, `V′[J/m]`, `V″[J/m²]`, `V‴[J/m³]` at each grid point via `num-dual` (§3).
- B₂(T) = `2π·N_A·trapz(R[m], g(R; T))·1e6` cm³/mol. (A fixed grid + trapz reproduces the reference; a higher-order rule on the same grid is acceptable if it still matches.)

A fixed grid (rather than the adaptive `physics::b2_v`) is chosen deliberately: it (a) reproduces the reference numbers and (b) lets the dual-T integration be a plain sum of dual values (§7).

## 6. Temperature derivatives & n_eff (dual-T)

The integrand depends on T through β and λ (and p = βV′ …), so analytic T-derivatives are impractical. Instead, thread a dual temperature through the (fixed-grid) integral:
```rust
let (b2, db2_dt, d2b2_dt2) = num_dual::second_derivative(
    |t| self.b2_dual(t, order, &precomputed_grid),  // sums dual integrand over the fixed grid
    T);
let neff = -3.0*(b2 + T*db2_dt) / (2.0*T*db2_dt + T*T*d2b2_dt2);  // paper Eq. 11
```
`b2_dual` is generic over the scalar (`DualNum<f64>`); the precomputed V-derivs are real, only T is dual. `second_derivative` returns B₂ and its first two T-derivatives in one pass.

(Public API: a `b2_and_neff(t, order) -> (B2Derivs_real_units, neff)` style method, plus a plain `b2(t, order) -> f64` in cm³/mol.)

## 7. Public API (sketch)
```rust
pub struct TangToennies { /* params */ }
impl TangToennies {
    pub fn v(&self, r_nm: f64) -> f64;                       // V/k_B [K], incl. tilde
    pub fn v_derivs(&self, r_nm: f64) -> (f64,f64,f64,f64);  // (V,V',V'',V''') K/nm^k via Dual3
    pub fn b2(&self, t: f64, order: u8) -> f64;              // cm^3/mol
    pub fn b2_neff(&self, t: f64, order: u8) -> (f64,f64,f64,f64); // B2, dB2dT, d2B2dT2, neff
}
pub fn neon_tt() -> TangToennies; // + argon_tt, krypton_tt, xenon_tt
```

## 8. Validation

1. **Potential values** vs the reference's `diffassert` anchors (V/k_B in K):
   - Ne: `v(0.16)=26879.940` (≤0.1%), `v(0.56)=−1.631` (≤0.1%)
   - Ar: `v(0.20)=51406.200` (≤0.1%), `v(0.9)=−0.918` (≤0.4%)
   - Kr: `v(0.24)=27872.324` (≤0.4%), `v(0.4)=−200.741` (≤0.4%), `v(1.00)=−0.982` (≤0.4%)
   - Xe: `v(0.26)=37578.501` (≤0.1%), `v(0.9)=−4.343` (≤0.4%)
2. **Dual derivatives** vs the reference's analytic `potprime/potprime2/potprime3`: compute the reference's analytic V′/V″/V‴ at a few R (port just those expressions into the test, or hard-code reference values from running the script) and assert the `num-dual` results match to ~1e-9 relative. (The script's own complex-step `diffassert` checks confirm its analytic derivatives, so matching them validates the autodiff.)
3. **B₂(T)** (classical, order 0) and **WK order-3** vs reference values generated by running the script's `B2(T, potvals, quantum=0|3)` (needs only numpy/scipy — *not* the `multicomplex` dep) at a handful of T per gas (e.g. 50, 100, 300, 1000 K). Assert agreement to the grid/trapz tolerance (~0.1–0.5%).
4. **n_eff** sanity: positive, finite, smooth; classical vs quantum ordering consistent with the paper (quantum correction grows at low T, largest for Ne).

## 9. Out of scope
- **He** (full quantum / phase-shift Beth–Uhlenbeck — separate feature).
- **QFH** effective-potential variant (the reference uses the WK series; QFH would have no reference to validate against here).
- **Any web/WASM surface** — this is a native Rust capability first. (`num-dual` is native-only here; keep it off the wasm hot path. If it ever needs wasm-gating, gate the module.)
- Higher-order WK (>3) and other noble gases (Rn).

## 10. File-by-file
- `Cargo.toml` — add `num-dual` to `[dependencies]`.
- `src/noblegas.rs` (new) — `TangToennies` struct, `v`/`v_tt`/`v_full`, `v_derivs` (Dual3), `b2`/`b2_neff` (fixed-grid + dual-T), `add_recursive`, the four constructors.
- `src/lib.rs` — `pub mod noblegas;` (+ re-exports if useful).
- `src/main.rs` (`#[cfg(test)] mod tests`) — validations §8.
- A short reference-data step: run `integrate_potentials.py`'s `B2()` (numpy/scipy only) to capture B₂(T) reference values for the tests; record them in the test (hard-coded) so the suite is self-contained.
