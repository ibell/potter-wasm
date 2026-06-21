# Full-quantum B₂ via phase shifts — ⁴He, ³He, Ne — design

- **Date:** 2026-06-21
- **Status:** Approved (ready for implementation plan)
- **Context:** The browser app is a companion to Bell, *J. Chem. Phys.* 152, 164508 (2020). Section IV.A of that paper takes ⁴He B₂/n_eff from *tabulated* fully-quantum literature values (Cencek et al., JCP 136, 224303 (2012)) — Fig. 8's dramatic n_eff ≈ 140 peak at ~10 K — because the semiclassical ħ² (WK/QFH) corrections we ship for the other species **diverge** for helium. This feature *computes* that fully-quantum B₂ from first principles (Beth–Uhlenbeck phase shifts) on the Cencek ab initio potential, for **⁴He, ³He, and Ne** (Ne as a cross-check against our existing WK Ne), and surfaces it in the real-fluids mode with n_eff.
- **Reference materials** — kept in `docs/refs/he/` as **local-only working copies (gitignored, NOT committed)**: `potter-wasm` is a public repo and these are AIP-copyrighted JCP supplementary material, so they must not be redistributed. The committed artifacts are the **clean-room Rust port** (functional form + parameters from a published paper are not themselves copyrightable) and a handful of embedded numeric validation anchors. From the Cencek 2012 SI + Ian's extraction:
  - `potentials.f90` — Fortran 90 implementation of the He–He potential (BO + adiabatic + relativistic + QED, with retardation and isotope multipliers). The thing we port.
  - `test.f90`, `readme.txt_readme.txt` — usage + descriptions.
  - `s4_he4prop.txt`, `s5_he3prop.txt` — tabulated B, TB′, T²B″ (+ uncertainties) for ⁴He / ³He, 1 K–10⁴ K.
  - `cencek_he4_neff_data.csv` (Ian's `Cencek.csv`) — ⁴He B₂/TB′/T²B″ + uncertainties; the n_eff formula matches ours exactly.

## 1. Goal

A spherical **Beth–Uhlenbeck** second-virial engine that, for ⁴He / ³He / Ne, computes B₂(T), dB₂/dT, d²B₂/dT², and n_eff in physical units (cm³/mol, K) by summing scattering phase shifts of the ab initio pair potential, with the correct quantum statistics per species. Validated tightly against the Cencek 2012 tabulated values **within their published uncertainties**, reproducing the paper's Fig. 8.

## 2. Potential port (`src/he_potential.rs`)

Port `potentials.f90` verbatim to Rust. **Work in atomic units** (r in Bohr a₀, energy in Hartree E_h) — the potential's native system — converting to K/cm³·mol⁻¹ only at the boundary (`toK = 315774.65` Hartree→K).

The file is all analytic: a Tang–Toennies damping function `damp(n,η,r)` (and `damp_mod`), a Casimir–Polder retardation rational `damp_ret(r)`, and several component modules each of the form `(P-poly)e^{−ar} + (Q-poly)e^{−br(−cr²)} − Σ damp·Cₙ/rⁿ`:
- `Total_Fit::total(ret6,r)` → `V_tot` (the recommended single-fit ⁴He potential).
- `Born_Oppenheimer::BO`, `adiabatic_correction::ad`, `Cowan_Griffin::CG`, `Darwin_1el::a3D1`, `Darwin_2el::a3D2`, `Breit::Br`, `Araki_Sucher::AS`.
- Interface: `V_BO`, `V_ad`, `V_rel = CG + D2 + Br`, `V_QED = a3D1 + a3D2 + AS`, `V_tot`.

**Isotope handling.** Nuclear masses (electron-mass units): `Mnuc4 = 7294.2995365`, `Mnuc3 = 5495.8852765`; reduced masses `μ₄₄ = Mnuc4/2`, `μ₃₃ = Mnuc3/2`. The adiabatic correction scales with the reduced mass: `mult44 = 1`, `mult33 = μ₄₄/μ₃₃`. So:
- **⁴He:** `V(r) = V_tot(r)` (mult44 = 1); equivalently `V_BO + V_ad + V_rel + V_QED`.
- **³He:** `V(r) = V_BO + mult33·V_ad + V_rel + V_QED`.
Include retardation (`ret6 = true`) — it matters at the long range that dominates low-T B₂. A port self-check: ⁴He `V_BO + V_ad + V_rel + V_QED` ≈ `V_tot` to fit accuracy, and the component values reproduce `test.f90`'s printed table at r = 1…12 a₀.

Public API:
```rust
pub enum He { He4, He3 }
pub fn v_he(iso: He, r_bohr: f64, retarded: bool) -> f64; // Hartree
pub fn reduced_mass_me(iso: He) -> f64;                   // electron masses
```

## 3. Phase-shift engine (`src/quantum.rs`)

**Variable-phase (Calogero) method.** For partial wave l and wavenumber k (a.u.), integrate the phase function δ_l(r) outward from a small r₀ to r_max:
```
δ_l′(r) = −(1/k) U(r) [ cos δ_l(r) · ĵ_l(kr) − sin δ_l(r) · ŷ_l(kr) ]²,   δ_l(r₀)=0
```
with `U(r) = 2μ V(r)/ℏ²` (a.u.: ℏ=1, so `U = 2μ V`), and ĵ_l, ŷ_l the Riccati–Bessel functions (computed by upward/downward recurrence). `δ_l(k) = δ_l(r_max)`. RK4, adaptive-enough step (a fraction of the local de Broglie wavelength). No asymptotic matching needed.

**Grids (T-independent → built once, reused across all T):**
- `k` grid: dense near 0 (low-energy scattering dominates low-T B₂), out to k_max set by the highest T of interest (βℏ²k²/2μ ≲ ~30).
- `l`: 0…l_max, growing the sum until `(2l+1)·|contribution|` is negligible at the highest k.
- `r_max`: large enough that δ_l has converged (long de Broglie wavelength at small k → integrate far, ~hundreds of a₀); r₀ small (inside the repulsive wall where δ≈0).
Store `δ_l(k)` and `dδ_l/dk` (finite-difference or from the ODE on the same grid).

## 4. Bound states

⁴He has exactly **one** l=0 dimer (a halo state, binding ≈ −1.1 mK). Find E_b by a shooting/Numerov eigenvalue solve of the radial Schrödinger equation for l=0; cross-check the bound-state *count* via **Levinson's theorem** `δ₀(k→0) = n_bound·π`. ³He has **none** (Levinson check: δ₀(0)=0). The dimer contributes only to ⁴He's `B₂_bound`.

## 5. B₂ assembly — Beth–Uhlenbeck (`src/quantum.rs`)

`B₂(T) = B₂_ideal + B₂_bound + B₂_scatt`, with thermal wavelength `λ = h/√(2π m k_B T)` (m the atomic mass). In a.u. and per the standard formulation:
- **B₂_ideal** — the ideal quantum-gas exchange term (∓λ³ statistics).
- **B₂_bound** — `∝ λ³ Σ_bound (2l+1) (e^{−βE_b} − …)` (⁴He: the one l=0 dimer; ³He: none).
- **B₂_scatt** — `∝ λ³ Σ_l w_l (2l+1) ∫₀^∞ e^{−βℏ²k²/2μ} (dδ_l/dk) dk`.

**Statistics weights `w_l` (the species-specific part):**
- **⁴He** (spin-0 boson): only **even l**.
- **³He** (spin-½ fermion): nuclear-spin combination — even l weighted ¼ (singlet), odd l weighted ¾ (triplet); fermionic ideal sign + spin degeneracy.
- **Ne** (²⁰Ne, spin-0 boson): even l; exchange/ideal term negligible at relevant T (≳ tens of K) — effectively the Boltzmann scattering sum, used as the WK cross-check.

The exact coefficients/signs are pinned by **validating B₂ against the Cencek tables** (§8) — those tabulated numbers are ground truth, so a sign/coefficient error shows up immediately.

Output: convert B₂ from a.u. volume (a₀³ per pair) to **cm³/mol** (`× N_A × (a₀ in cm)³`).

## 6. n_eff and T-derivatives

T enters only through λ³ (∝ T^{−3/2}), the Boltzmann weight `e^{−βℏ²k²/2μ}`, and `e^{−βE_b}` — all analytic. Differentiate under the integral for dB₂/dT and d²B₂/dT² (the δ_l(k) table is T-independent and reused), then `n_eff = −3(B₂ + T·B₂′)/(2T·B₂′ + T²·B₂″)` (`physics::neff` / a local copy). Returns `B2Derivs` + n_eff, matching every other branch. (Cross-check: the Cencek table provides TB′ and T²B″ directly.)

## 7. wasm + web

- `src/lib.rs`: `quantum_b2_neff(species: u32, t: f64) -> [f64; 4]` (0=⁴He, 1=³He, 2=Ne) → `[B₂, dB₂/dT, d²B₂/dT², n_eff]`, cm³/mol. The expensive δ_l(k) table is built per call (one species, one T); if a T-sweep proves slow, expose a build-once/reuse split like `noblegas::grid`/`b2_neff_with_grid` (§10).
- `poc_quantum_b2(species, t, out)` wasm export (4 f64, unaligned writes).
- Web (`web/index.html.in`): a **"full quantum"** optgroup in the real-fluids species `<select>` — ⁴He, ³He, Ne. `quantumRow(species, t)` mirrors `noblegasRow`/`moleculeRow` (cm³/mol → straight into the real-fluids row/plot/table). For **Ne**, optionally overlay our existing **WK Ne** as the live cross-check (a 2nd curve). Default T-range ~1–500 K (log) so the n_eff peak is visible; molecule-style point cap (the per-T cost is modest, but the δ-table build is the cost).

## 8. Validation (tight, against the published tables + their uncertainties)

1. **Potential port** — Rust `V_BO/V_ad/V_rel/V_QED/V_tot` reproduce `test.f90`'s printed table (r = 1…12 a₀) to ~1e-4 K; ⁴He component-sum ≈ `V_tot`.
2. **⁴He B₂/TB′/T²B″** — vs `cencek_he4_neff_data.csv` / `s4_he4prop.txt` at a spread of T (e.g. 2, 4.2, 10, 20, 100, 500 K) **within the tabulated uncertainties** (e.g. 10 K: B=−23.125±0.020, TB′=41.022±0.021, T²B″=−82.478±0.044). n_eff matches Fig. 8 (peak ≈ 140 near 8–12 K).
3. **³He** — vs `s5_he3prop.txt` (validates the Fermi statistics weights).
4. **Dimer** — ⁴He E_b ≈ −1.1 mK; Levinson count = 1 (⁴He), 0 (³He).
5. **Ne full-Q vs WK Ne** — agree at moderate T (≳ 50 K), full-Q "more correct" at low T.
6. **High-T limit** — B₂ → the classical integral as T grows (quantum corrections vanish).
7. **Method check** — the engine on a simple LM2M2 (Aziz–Slaman 1991) potential vs published LM2M2 quantum B₂, isolating engine correctness from the Cencek-potential port.

## 9. Out of scope

- Mixtures / the interaction virial B₃₄ (the SI's `s6`/`s7`).
- The Przybytek 2017 potential update (the 2012 potential matches our validation data; 2017 is a later drop-in).
- B₃ and higher quantum virials.
- H₂ / D₂ (molecule — rotational structure; a separate project).
- Path-integral routes.

## 10. Risks & mitigations

- **³He Fermi spin-statistics weights** — the trickiest physics; mitigated by direct validation against `s5_he3prop.txt`. If it resists, ship ⁴He + Ne first and add ³He once the weights reproduce the table.
- **Cencek potential transcription** — mitigated by the `test.f90` value check (§8.1) and the LM2M2 method check (§8.7), which separate a port bug from an engine bug.
- **Shallow ⁴He dimer / low-T convergence** — needs a fine k-grid near 0 and large r_max; validate E_b and the low-T B₂ rows (1–4 K) where the dimer dominates.
- **Performance** — the δ_l(k) table build is the cost (1-D ODEs, ~l_max×k points); per-T is cheap. T-independent so build once per species; expose a reuse split if web sweeps are slow.

## 11. File-by-file

- `docs/refs/he/` — the Cencek SI reference files (potential Fortran, property tables, csv). **Gitignored / NOT committed** (AIP copyright; public repo). Local working copies for the port + validation only.
- `src/he_potential.rs` (new) — port of `potentials.f90`; `v_he`, `reduced_mass_me`, isotope assembly, retardation.
- `src/quantum.rs` (new) — Riccati–Bessel, variable-phase δ_l(k), bound-state solve, Beth–Uhlenbeck B₂ with statistics, analytic T-derivatives → `B2Derivs`.
- `src/lib.rs` — `quantum_b2_neff` + `poc_quantum_b2` wasm export; `pub mod he_potential; pub mod quantum;`.
- `src/main.rs` — validation tests (§8): potential-port table, ⁴He/³He vs tabulated (with uncertainty bands), dimer, Ne-vs-WK, LM2M2 method check, high-T limit.
- `web/index.html.in` — "full quantum" species group + `quantumRow` + compute/plot wiring; Ne WK overlay.
- A node e2e (`node/quantum-e2e.mjs`) — `poc_quantum_b2` vs the tabulated values through the built wasm.
