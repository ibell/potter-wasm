# Full-quantum B₂ via phase shifts (⁴He, ³He, Ne) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute the fully-quantum second virial coefficient B₂(T)/n_eff for ⁴He, ³He, and Ne by summing scattering phase shifts of the Cencek 2012 ab initio He potential (Beth–Uhlenbeck), and surface it in the web real-fluids mode — reproducing the paper's Fig. 8 (n_eff peak ≈ 140 for ⁴He).

**Architecture:** Port the Cencek `potentials.f90` to Rust (atomic units). A variable-phase (Calogero) engine computes the T-independent phase-shift table δ_l(k) once; the ⁴He halo dimer is found by a Numerov eigenvalue solve. Beth–Uhlenbeck assembles B₂ with per-species quantum statistics; T-derivatives are analytic → B₂Derivs + n_eff. A `poc_quantum_b2` wasm export feeds a "full quantum" group in the real-fluids species dropdown.

**Tech Stack:** Pure Rust (no new deps). New `src/he_potential.rs`, `src/quantum.rs`. Tests in `src/main.rs`. Web `web/index.html.in`.

**Reference spec:** `docs/superpowers/specs/2026-06-21-helium-full-quantum-b2-design.md`

---

## Implementation outcome (what actually shipped — read this first)

The feature is complete for **⁴He and ³He** (the scientific core, reproducing Bell 2020 Fig. 8). A few deliberate deviations from the task-by-task plan below, all validated:

- **Scattering engine = Numerov matching, not RK4 variable-phase.** The He repulsive wall is too stiff for RK4 on the nonlinear Calogero equation (not L-stable — it blows up unless given an impractical step, making runs hours long). Production `quantum_b2_parts` instead integrates the linear radial equation by **Numerov** and matches to free Riccati–Bessel, recovering the continuous Levinson-anchored δ_l(k) by k-unwrapping from a small-k node-count anchor (+ the dimer π for ⁴He). The RK4 `phase_shifts` is kept only for the Task 2 square-well unit test. Full table now runs in ~1 min (release). The Hurly–Mehl 2007 formula (Eqs. 9, 18–24) is still transcribed verbatim, no fitted constants.
- **Validation (vs Cencek 2012, no tuning):** ⁴He within published uncertainty at 4–20 K (T=4: −85.0613 vs −85.061); TB′/T²B″ match (T=10: 41.026/−82.480 vs 41.022/−82.478); **n_eff peaks at 139.8 at 9 K** (Fig. 8 ≈ 140). ³He matches with the sourced I=½ Fermi weights (T=4: −62.312 vs −62.311). High-T → classical.
- **Ne deferred (documented limitation).** Ne₂'s deep well supports multiple bound states across several l; the full Beth–Uhlenbeck B₂ needs their Levinson π's + B_bound terms (He needs only its single shallow dimer), and reaching higher T needs more partial waves than the high-l Bessel/matching is stable for. `Species::Ne` is documented high-T-only; use the existing Wigner–Kirkwood `noblegas` route for Ne. Task 7 became a trend-only guard.
- **Web (Task 9) dropped.** These full-quantum calcs are seconds–minutes per point and wasm is single-threaded (no `thread::spawn`), so they are not interactive-viable. The `poc_quantum_b2` wasm export (Task 8) ships with a cfg-gated serial fallback for wasm and is exercised by `node/quantum-e2e.mjs`; the real-fluids web group was not added.
- **Heavy tests are `#[ignore]`d** (run `cargo test --release -- --ignored`); routine `cargo test` stays fast (44 pass, 6 ignored).
- **Reproduction tooling:** `examples/b2print.rs` (B₂ vs Cencek), `examples/neff_curve.rs` + `scripts/plot_neff.py` → `figures/neff_he_quantum.png` (n_eff(T) for ⁴He/³He overlaid with Cencek).

The task-by-task plan below is preserved as the historical record; where it says "RK4 variable-phase for production B₂", "pin normalization", "Ne cross-check within 2 cm³/mol", or "web group", see the outcome above.

**Local reference materials** (gitignored, present on disk — DO NOT commit; AIP copyright):
- `docs/refs/he/potentials.f90` — the He–He potential Fortran to port.
- `docs/refs/he/test.f90` — usage; its printed output is the port's ground truth (captured below).
- `docs/refs/he/s4_he4prop.txt`, `s5_he3prop.txt`, `cencek_he4_neff_data.csv` — tabulated B, TB′, T²B″ (+ uncertainties).

**Units convention (engine is in atomic units):** length Bohr a₀, energy Hartree E_h, mass electron-mass mₑ, ℏ=1. Constants: `HARTREE_K = 315774.65` (E_h→K), `A0_CM = 0.529177210903e-8` (a₀→cm), `N_A = 6.02214076e23`, `AMU_ME = 1822.888486209` (amu→mₑ). **Scattering & virial use ATOMIC masses** (Cencek/Hurly–Mehl convention): the pair reduced mass is μ = m_atom/2 with atomic masses (amu) ⁴He 4.002602, ³He 3.0160293, Ne 20.1797. The He **nuclear** masses (mₑ) Mnuc4 = 7294.2995365, Mnuc3 = 5495.8852765 are used **only** for the adiabatic-correction multiplier `mult` in the potential (Task 1), not for the Schrödinger equation or λ_T.

**Authoritative B₂ formula (sourced from the original papers — no tuning).** Bell 2020 (the paper we reproduce) does *not* derive a Beth–Uhlenbeck formula: its Fig. 8 ⁴He curve is the *tabulated* fully-quantum B/n_eff of **Cencek et al., J. Chem. Phys. 136, 224303 (2012)** (Bell 2020 ref 56; the `s4_he4prop`/`s5_he3prop` SI tables), and its other noble gases use 3rd-order Wigner–Kirkwood (already implemented). Cencek 2012 computed those values with the phase-shift method of **Hurly & Mehl, "⁴He Thermophysical Properties: New Ab Initio Calculations," J. Res. Natl. Inst. Stand. Technol. 112, 75 (2007)**, Eqs. (9), (18)–(24) — reproduced verbatim in Task 4. The ³He Fermi spin-statistics (even-l ¼ singlet / odd-l ¾ triplet, ideal-term sign flip) follow from the standard identical-fermion result for nuclear spin I=½ (weights w_e = I/(2I+1)=¼, w_o = (I+1)/(2I+1)=¾). The Cencek SI tables (B, TB′, T²B″ with per-column uncertainties) are the a-priori validation anchor: a correct implementation of the published formula matches them within uncertainty with **no fitted constants**.

**Ground-truth potential values** (from compiling `docs/refs/he/{potentials,test}.f90` with gfortran; V in K, r in Bohr):
| r | V_BO | V_ad | V_rel | V_QED | V_tot |
|---|---|---|---|---|---|
| 2.0 | 36142.3480 | 11.8173 | −2.8634 | 0.5100 | 36151.8089 |
| 4.0 | 292.5705 | 0.1077 | 0.0323 | 0.0089 | 292.7203 |
| 5.6 | −11.0006 | −0.0090 | 0.0154 | −0.0014 | −10.9957 |
| 9.0 | −0.9898 | −0.0007 | 0.0019 | −0.0003 | −0.9889 |
(`mult44 = 1.0`, `mult34 = 1.163614611`, `mult33 = 1.327229221`; retardation adds `+0.0043 K` at r=1, negligible beyond.)

**⁴He reference (Cencek 2012; T[K], B, TB′, T²B″ in cm³/mol, with uncertainties U):**
| T | B (±U) | TB′ (±U) | T²B″ (±U) |
|---|---|---|---|
| 2 | −194.38 ±0.13 | 233.39 ±0.17 | −542.92 ±0.46 |
| 4 | −85.061 ±0.054 | 104.263 ±0.063 | −214.00 ±0.15 |
| 10 | −23.125 ±0.020 | 41.022 ±0.021 | −82.478 ±0.044 |
| 20 | −2.7464 ±0.0097 | 20.0766 ±0.0095 | −41.445 ±0.020 |
| 100 | 11.6747 ±0.0023 | 2.0908 ±0.0019 | −6.9989 ±0.0037 |
| 500 | 11.00715 ±0.00062 | −1.87546 ±0.00048 | 0.98256 ±0.00087 |

**³He reference (Cencek 2012):** T=4: B=−62.311, TB′=73.731, T²B″=−137.636; T=10: −16.200, 32.147, −62.159; T=100: 12.0385, 1.6257, −5.9378; T=500: 11.05373, −1.93494, 1.11790.

**n_eff** (matches `Cencek..py`): `n_eff = −3(B + TB′)/(2TB′ + T²B″)`. Check: ⁴He@10K → −3(−23.125+41.022)/(2·41.022−82.478) = 123.7 (the Fig. 8 peak region).

**Conventions:** tests in `src/main.rs` `#[cfg(test)] mod tests`, import `potter_poc::…`. Commit trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. `B2Derivs{b2,db2_dt,d2b2_dt2}` + `.neff(t)` is in `physics.rs`.

---

## Task 1: Port the He potential (`src/he_potential.rs`)

**Files:** Create `src/he_potential.rs`; Modify `src/lib.rs`; Test `src/main.rs`.

The potential is a verbatim port of `docs/refs/he/potentials.f90` (present on disk — read it). All analytic. Translate each Fortran module to Rust functions, parameters to `const`s. Output Hartree, input Bohr.

- [ ] **Step 1: Write the failing test** (in `src/main.rs` `mod tests`)

```rust
    #[test]
    fn he_potential_matches_fortran() {
        use potter_poc::he_potential::{v_components, He};
        const TOK: f64 = 315774.65;
        // (r_bohr, V_BO, V_ad, V_rel, V_QED, V_tot) in K — from the compiled SI Fortran.
        let rows = [
            (2.0, 36142.3480, 11.8173, -2.8634, 0.5100, 36151.8089),
            (4.0, 292.5705, 0.1077, 0.0323, 0.0089, 292.7203),
            (5.6, -11.0006, -0.0090, 0.0154, -0.0014, -10.9957),
            (9.0, -0.9898, -0.0007, 0.0019, -0.0003, -0.9889),
        ];
        for &(r, bo, ad, rel, qed, tot) in &rows {
            let c = v_components(r, false); // (bo, ad, rel, qed, tot) in Hartree
            assert!((c.0 * TOK - bo).abs() < 1e-3, "V_BO r={r}: {}", c.0 * TOK);
            assert!((c.1 * TOK - ad).abs() < 1e-3, "V_ad r={r}: {}", c.1 * TOK);
            assert!((c.2 * TOK - rel).abs() < 1e-3, "V_rel r={r}: {}", c.2 * TOK);
            assert!((c.3 * TOK - qed).abs() < 1e-3, "V_QED r={r}: {}", c.3 * TOK);
            assert!((c.4 * TOK - tot).abs() < 1e-3, "V_tot r={r}: {}", c.4 * TOK);
        }
        // 4He well depth ~ -11 K near r=5.6 a0 via the species potential.
        let v56 = potter_poc::he_potential::v_he(He::He4, 5.6, false) * TOK;
        assert!((v56 - (-10.9957)).abs() < 0.01, "He4 V(5.6)={v56}");
        // 3He differs from 4He only via the adiabatic multiplier (mult33).
        let d4 = potter_poc::he_potential::v_he(He::He4, 5.6, false);
        let d3 = potter_poc::he_potential::v_he(He::He3, 5.6, false);
        assert!((d4 - d3).abs() > 0.0, "3He != 4He potential");
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test he_potential_matches_fortran`
Expected: FAIL — `potter_poc::he_potential` does not exist.

- [ ] **Step 3: Implement** — read `docs/refs/he/potentials.f90` and translate it. Create `src/he_potential.rs` with:
  - `fn damp(n: usize, eta: f64, r: f64) -> f64` and `fn damp_mod(n, eta, r) -> f64` (the two Tang–Toennies dampers; copy the `br>1` / `br<=1` branches exactly).
  - `fn damp_ret(r) -> f64` (the retardation rational from `module retardation`, with its A1..A5/B1..B6 consts).
  - One function per component module returning Hartree: `total(ret6,r)` (Total_Fit), `bo(ret6,r)` (Born_Oppenheimer), `ad(r)`, `cg(r)`, `d1(r)`/`a3d1(r)`, `d2(r)`/`a3d2(r)`, `br(ret6,r)`, `as_(ret6,r)`, `sigma_total(r)`. Each is `(P0+P1 r+P2 r²)e^{−a r} + (Q0+Q1 r)e^{−b r(−c r²)} − Σ damp·Cₙ/rⁿ` — copy the exact constants and term structure from the Fortran.
  - The interface (matching `module potential_interface`):
    ```rust
    pub enum He { He4, He3 }
    const MNUC4: f64 = 7294.2995365;
    const MNUC3: f64 = 5495.8852765;
    pub fn reduced_mass_me(iso: He) -> f64 { match iso { He::He4 => MNUC4/2.0, He::He3 => MNUC3/2.0 } }
    fn mult(iso: He) -> f64 { match iso { He::He4 => 1.0, He::He3 => (MNUC4/2.0)/(MNUC3/2.0) } }
    /// (V_BO, V_ad, V_rel, V_QED, V_tot) in Hartree at r [Bohr].
    pub fn v_components(r: f64, ret6: bool) -> (f64, f64, f64, f64, f64) {
        let v_rel = cg(r) + d2(r) + br(ret6, r);
        let v_qed = a3d1(r) + a3d2(r) + as_(ret6, r);
        (bo(ret6, r), ad(r), v_rel, v_qed, total(ret6, r))
    }
    /// Isotope pair potential V(r) [Hartree], r [Bohr]: V_BO + mult·V_ad + V_rel + V_QED.
    pub fn v_he(iso: He, r: f64, ret6: bool) -> f64 {
        let (bo, ad_, rel, qed, _tot) = v_components(r, ret6);
        bo + mult(iso) * ad_ + rel + qed
    }
    ```

Register in `src/lib.rs`: add `pub mod he_potential;` near the other `pub mod` lines.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test he_potential_matches_fortran`
Expected: PASS (components reproduce the Fortran table to <1e-3 K).

- [ ] **Step 5: Commit**

```bash
git add src/he_potential.rs src/lib.rs src/main.rs
git commit -m "$(cat <<'EOF'
Port the Cencek 2012 He-He ab initio potential to Rust

Clean-room translation of the SI potentials.f90 (BO + adiabatic + rel +
QED, Tang-Toennies damping, Casimir-Polder retardation, isotope mult).
Verified vs the compiled-Fortran value table to <1e-3 K.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Riccati–Bessel functions + variable-phase phase shifts (`src/quantum.rs`)

**Files:** Create `src/quantum.rs`; Modify `src/lib.rs`; Test `src/main.rs`.

The scattering core. Validate δ₀(k) against the **closed-form square-well** result, which needs only ĵ₀/ŷ₀ (sin/−cos) — robust and unambiguous.

- [ ] **Step 1: Write the failing test**

```rust
    #[test]
    fn phase_shift_square_well_s_wave() {
        use potter_poc::quantum::{riccati, s_wave_phase_for_test};
        // Riccati-Bessel sanity: ĵ_0(x)=sin x, ŷ_0(x)=-cos x; recurrence to l=2.
        let (j, y) = riccati(2, 1.3_f64);
        assert!((j[0] - 1.3_f64.sin()).abs() < 1e-12 && (y[0] + 1.3_f64.cos()).abs() < 1e-12);
        // s-wave square well V=-V0 (r<R) else 0: delta0 = -kR + atan((k/k') tan(k' R)),
        // k' = sqrt(k^2 + 2 mu V0). Test the variable-phase engine vs this closed form.
        let (mu, v0, rr) = (1.0_f64, 2.0_f64, 1.5_f64);
        for &k in &[0.4_f64, 1.0, 2.5] {
            let kp = (k * k + 2.0 * mu * v0).sqrt();
            let mut exact = -k * rr + ((k / kp) * (kp * rr).tan()).atan();
            // fold the atan branch to match the engine's continuous accumulation near resonance
            let num = s_wave_phase_for_test(mu, v0, rr, k);
            let mut d = num - exact;
            while d > std::f64::consts::PI / 2.0 { exact += std::f64::consts::PI; d = num - exact; }
            while d < -std::f64::consts::PI / 2.0 { exact -= std::f64::consts::PI; d = num - exact; }
            assert!((num - exact).abs() < 2e-3, "k={k}: engine {num} vs exact {exact}");
        }
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test phase_shift_square_well_s_wave`
Expected: FAIL — `potter_poc::quantum` does not exist.

- [ ] **Step 3: Implement** — create `src/quantum.rs`:

```rust
//! Full-quantum B2 via Beth-Uhlenbeck phase shifts (variable-phase method).
//! Atomic units internally (Bohr, Hartree, electron mass, hbar=1).

/// Riccati-Bessel functions up to order `lmax`: jhat_l(x)=x j_l(x), yhat_l(x)=x y_l(x).
/// jhat_0=sin x, yhat_0=-cos x; jhat_1=sin x/x - cos x, yhat_1=-cos x/x - sin x;
/// both satisfy f_{l+1} = (2l+1)/x f_l - f_{l-1}. Upward recurrence (adequate for the
/// kr range used here; the full B2 vs the Cencek table is the high-l check).
pub fn riccati(lmax: usize, x: f64) -> (Vec<f64>, Vec<f64>) {
    let (s, c) = (x.sin(), x.cos());
    let mut j = vec![0.0; lmax + 1];
    let mut y = vec![0.0; lmax + 1];
    j[0] = s; y[0] = -c;
    if lmax >= 1 { j[1] = s / x - c; y[1] = -c / x - s; }
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
    v: &V, mu: f64, k: f64, lmax: usize, r0: f64, rmax: f64, steps: usize,
) -> Vec<f64> {
    let h = (rmax - r0) / steps as f64;
    let mut d = vec![0.0_f64; lmax + 1];
    let deriv = |r: f64, dl: &[f64]| -> Vec<f64> {
        let u = 2.0 * mu * v(r);
        let (j, y) = riccati(lmax, k * r);
        (0..=lmax).map(|l| {
            let b = d_cos_sin(dl[l], j[l], y[l]);
            -(1.0 / k) * u * b * b
        }).collect()
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
fn d_cos_sin(d: f64, jl: f64, yl: f64) -> f64 { d.cos() * jl - d.sin() * yl }

/// Test helper: s-wave phase shift for a square well V=-v0 (r<rr) else 0.
pub fn s_wave_phase_for_test(mu: f64, v0: f64, rr: f64, k: f64) -> f64 {
    let v = |r: f64| if r < rr { -v0 } else { 0.0 };
    phase_shifts(&v, mu, k, 0, 1e-6, rr + 30.0, 6000)[0]
}
```

Register: add `pub mod quantum;` to `src/lib.rs`.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test phase_shift_square_well_s_wave`
Expected: PASS (engine δ₀ matches the closed form to ~2e-3). If marginal, raise `steps` to 12000 in `s_wave_phase_for_test`.

- [ ] **Step 5: Commit**

```bash
git add src/quantum.rs src/lib.rs src/main.rs
git commit -m "$(cat <<'EOF'
Add Riccati-Bessel + variable-phase (Calogero) phase-shift engine

delta_l(k) by RK4 integration of the Calogero equation; validated vs the
closed-form square-well s-wave phase shift.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: ⁴He dimer bound state (Numerov eigenvalue + Levinson)

**Files:** Modify `src/quantum.rs`; Test `src/main.rs`.

⁴He has one l=0 halo dimer, binding ≈ −1.1 mK (≈ −1.1e-3 K = −3.5e-9 Hartree). Find it by Numerov; ³He has none.

- [ ] **Step 1: Write the failing test**

```rust
    #[test]
    fn he4_dimer_binding_energy() {
        use potter_poc::he_potential::{reduced_mass_me, v_he, He};
        use potter_poc::quantum::s_wave_bound_energy;
        let mu = reduced_mass_me(He::He4);
        let v = |r: f64| v_he(He::He4, r, true); // Hartree
        // returns Some(E_b<0 in Hartree) or None. ~ -1.1 mK = -3.48e-9 Hartree.
        let eb = s_wave_bound_energy(&v, mu).expect("4He has one dimer");
        let eb_mk = eb * 315774.65 * 1e3; // Hartree -> K -> mK
        assert!(eb_mk < 0.0 && (eb_mk - (-1.1)).abs() < 0.6, "E_b = {eb_mk} mK (expect ~ -1.1)");
        // 3He: no bound state
        let mu3 = reduced_mass_me(He::He3);
        let v3 = |r: f64| v_he(He::He3, r, true);
        assert!(s_wave_bound_energy(&v3, mu3).is_none(), "3He has no dimer");
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test he4_dimer_binding_energy`
Expected: FAIL — `s_wave_bound_energy` not found.

- [ ] **Step 3: Implement** — add to `src/quantum.rs`:

```rust
/// Number of l=0 nodes of the zero-energy (E=0) radial solution = number of s-wave
/// bound states (Levinson, l=0). Numerov outward integration of u'' = 2 mu V u.
fn s_wave_node_count<V: Fn(f64) -> f64>(v: &V, mu: f64, rmax: f64, n: usize) -> usize {
    let h = rmax / n as f64;
    let f = |r: f64| 2.0 * mu * v(r); // u'' = f(r) u  (E=0)
    let (mut u0, mut u1) = (0.0_f64, 1e-10_f64);
    let w = |r: f64| 1.0 - h * h / 12.0 * f(r);
    let mut nodes = 0usize;
    let mut r = h;
    for _ in 2..=n {
        let rn = r + h;
        let u2 = ((12.0 - 10.0 * w(r)) * u1 - w(r - h) * u0) / w(rn);
        if u1 * u2 < 0.0 { nodes += 1; }
        u0 = u1; u1 = u2; r = rn;
    }
    nodes
}

/// s-wave bound-state energy (Hartree, <0) of the deepest/only state, or None.
/// Shooting: count nodes of the E<0 solution; bisect E so the wavefunction decays.
pub fn s_wave_bound_energy<V: Fn(f64) -> f64>(v: &V, mu: f64) -> Option<f64> {
    let rmax = 800.0; // a0 — the He dimer is enormous (~50 a0); integrate far
    let n = 200_000;
    if s_wave_node_count(v, mu, rmax, n) == 0 { return None; }
    // bracket E in [-1e-6, -1e-12] Hartree (the dimer is ~ -3.5e-9); the number of
    // nodes of the (E) solution that turn over identifies the bound energy: match the
    // log-derivative at rmax to a decaying exponential via a shooting mismatch sign.
    let mismatch = |e: f64| -> f64 {
        let h = rmax / n as f64;
        let kappa = (-2.0 * mu * e).sqrt(); // decay rate
        let f = |r: f64| 2.0 * mu * (v(r) - e);
        let w = |r: f64| 1.0 - h * h / 12.0 * f(r);
        let (mut u0, mut u1) = (0.0_f64, 1e-30_f64);
        let mut r = h;
        let (mut ulast, mut uprev) = (0.0, 0.0);
        for _ in 2..=n {
            let rn = r + h;
            let u2 = ((12.0 - 10.0 * w(r)) * u1 - w(r - h) * u0) / w(rn);
            uprev = u1; ulast = u2; u0 = u1; u1 = u2; r = rn;
        }
        // log-derivative minus the expected -kappa of a decaying tail
        (ulast - uprev) / (h * ulast) + kappa
    };
    // bisection on log scale for the shallow state
    let (mut lo, mut hi) = (-1e-6_f64, -1e-13_f64);
    let (mut flo, mut fhi) = (mismatch(lo), mismatch(hi));
    if flo * fhi > 0.0 { return None; }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        let fm = mismatch(mid);
        if flo * fm <= 0.0 { hi = mid; fhi = fm; } else { lo = mid; flo = fm; }
        let _ = fhi;
    }
    Some(0.5 * (lo + hi))
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test he4_dimer_binding_energy`
Expected: PASS (E_b within ~0.6 mK of −1.1 mK; ³He None). The halo state is numerically delicate: if the bracket/`rmax`/`n` need tuning to converge E_b, adjust them (rmax up to 1500 a0, n up to 5e5) so the known ~−1.1 mK is recovered — that value is the validation target.

- [ ] **Step 5: Commit**

```bash
git add src/quantum.rs src/main.rs
git commit -m "$(cat <<'EOF'
Add s-wave bound-state solver; 4He halo dimer (~ -1.1 mK), 3He none

Numerov node-count (Levinson) + shooting eigenvalue. Validates the 4He
dimer binding and that 3He is unbound.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Beth–Uhlenbeck B₂ for ⁴He (Bose, even l) + high-T classical limit

**Files:** Modify `src/quantum.rs`; Test `src/main.rs`. **Implements the published Hurly & Mehl 2007 formula verbatim (Eqs. 9, 18–24) — no fitted constants.** The Cencek 2012 table and the high-T classical limit are *validation*, not fitting targets.

- [ ] **Step 1: Write the failing test**

```rust
    #[test]
    fn he4_b2_matches_cencek() {
        use potter_poc::quantum::quantum_b2;
        use potter_poc::quantum::Species;
        // 4He B2 (cm^3/mol) vs Cencek 2012 within the tabulated uncertainties (loosened
        // slightly for grid error). (T, B, U).
        let refs = [(4.0, -85.061, 0.06), (10.0, -23.125, 0.05), (20.0, -2.7464, 0.03),
                    (100.0, 11.6747, 0.02), (500.0, 11.00715, 0.02)];
        for &(t, b, u) in &refs {
            let got = quantum_b2(Species::He4, t);
            assert!((got - b).abs() < u.max(0.1), "4He B2 T={t}: {got} vs {b} (±{u})");
        }
    }

    #[test]
    fn quantum_b2_high_t_to_classical() {
        use potter_poc::quantum::{quantum_b2, classical_b2, Species};
        // At high T the full-quantum B2 -> the classical integral of the same potential.
        for &t in &[2000.0_f64, 5000.0] {
            let q = quantum_b2(Species::He4, t);
            let c = classical_b2(Species::He4, t);
            assert!((q - c).abs() / c.abs() < 0.05, "T={t}: quantum {q} vs classical {c}");
        }
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test he4_b2_matches_cencek`
Expected: FAIL — `quantum_b2` not found.

- [ ] **Step 3: Implement** — add to `src/quantum.rs`. **This is the Hurly & Mehl 2007 formula transcribed verbatim (Eqs. 9, 18–24); every prefactor is sourced, not fitted.** The doc-comment maps each line to its published equation.

```rust
use crate::he_potential::{v_he, He};

const HARTREE_K: f64 = 315774.65;
const A0_CM: f64 = 0.529177210903e-8;
const N_A: f64 = 6.02214076e23;
const AMU_ME: f64 = 1822.888486209;
const PI: f64 = std::f64::consts::PI;
const SQRT2: f64 = std::f64::consts::SQRT_2;

#[derive(Clone, Copy)]
pub enum Species { He4, He3, Ne }

fn iso(sp: Species) -> He { match sp { Species::He3 => He::He3, _ => He::He4 } }
fn mass_amu(sp: Species) -> f64 { match sp { Species::He4 => 4.002602, Species::He3 => 3.0160293, Species::Ne => 20.1797 } }

/// V(r) [Hartree] for the species (He potential for He4/He3; Ne uses the TT potential
/// converted to a.u.: neon_tt v_full is V/kB[K] at r[nm]).
fn potential(sp: Species) -> impl Fn(f64) -> f64 {
    move |r_bohr: f64| match sp {
        Species::Ne => {
            let r_nm = r_bohr * A0_CM * 1e7; // a0->cm->nm
            crate::noblegas::neon_tt().v(r_nm) / HARTREE_K // K -> Hartree
        }
        s => v_he(iso(s), r_bohr, true),
    }
}

/// Pair reduced mass μ (electron masses), ATOMIC masses (Cencek/Hurly–Mehl convention):
/// μ = m_atom/2 for identical atoms. Chosen so E[K] = α κ² exactly (see `quantum_b2_parts`).
fn mu_pair(sp: Species) -> f64 { mass_amu(sp) * AMU_ME / 2.0 }

/// Statistical weight w_l of partial wave l in S(κ) = Σ_l w_l (2l+1) δ_l (Hurly–Mehl Eq. 9,
/// generalized to nuclear spin). spin-0 bosons (⁴He, ²⁰Ne): even l weight 1, odd l 0.
/// spin-½ fermion (³He, I=½): even l (singlet) w_e=I/(2I+1)=¼, odd l (triplet) w_o=(I+1)/(2I+1)=¾.
fn l_weight(sp: Species, l: usize) -> f64 {
    match sp {
        Species::He4 | Species::Ne => if l % 2 == 0 { 1.0 } else { 0.0 },
        Species::He3 => if l % 2 == 0 { 0.25 } else { 0.75 },
    }
}

/// Ideal-gas (exchange) coefficient c so that B_ideal = c · N_A Λ³ (Hurly–Mehl Eq. 20,
/// generalized: B_id = ∓ 2^{-5/2} λ_T³ N_A /(2I+1) = c N_A Λ³ with Λ=√2 λ_T, ∓ = −Bose/+Fermi).
/// ⁴He & ²⁰Ne (boson, I=0): c = −1/16.  ³He (fermion, I=½): c = +1/32.
fn c_ideal(sp: Species) -> f64 { match sp { Species::He3 => 1.0 / 32.0, _ => -1.0 / 16.0 } }

/// Classical B2 (cm^3/mol) of the same potential, for the high-T anchor.
/// B2 = -2 pi N_A int_0^inf (e^{-beta V} - 1) r^2 dr   [a0^3 per pair -> cm^3/mol].
pub fn classical_b2(sp: Species, t: f64) -> f64 {
    let v = potential(sp);
    let beta = HARTREE_K / t; // 1/(kT) in 1/Hartree
    let (n, rmax) = (200_000usize, 60.0_f64);
    let h = rmax / n as f64;
    let mut s = 0.0;
    for i in 0..=n {
        let r = (i as f64) * h + 1e-9; // small floor avoids r=0
        let f = ((-beta * v(r)).exp() - 1.0) * r * r;
        s += if i == 0 || i == n { 0.5 } else { 1.0 } * f;
    }
    -2.0 * PI * N_A * s * h * A0_CM.powi(3)
}

/// Fully-quantum B2 (cm^3/mol) via Beth-Uhlenbeck phase shifts.
pub fn quantum_b2(sp: Species, t: f64) -> f64 { quantum_b2_parts(sp, t).0 }

/// (B2, dB2/dT, d2B2/dT2) in cm^3/mol — the Beth–Uhlenbeck second virial coefficient and
/// its analytic T-derivatives, exactly as published by Hurly & Mehl, J. Res. Natl. Inst.
/// Stand. Technol. 112, 75 (2007), Eqs. (9),(18)-(24) (the method Cencek 2012 used):
///
///   B = B_th + B_ideal + B_bound                                              (18)
///   B_th    = -2 N_A Λ³ α I₀ / (π T),   I₀ = ∫₀^∞ e^{-ακ²/T} S(κ) κ dκ          (19,22)
///   B_ideal = c_ideal · N_A Λ³                                                (20)
///   B_bound = -N_A Λ³ [e^{T_b/T} - 1]    (⁴He dimer: even l=0, (2l+1)=1)        (21)
///   α = (mₑ/m_atom)(E_h/k_B) [K];   Λ = √2 λ_T,   λ_T = h/√(2π m_atom k_B T)     (23,24)
///   S(κ) = Σ_l w_l (2l+1) δ_l(κ)                                               (9)
/// Because μ = m_atom/2, Eq. (6) κ²=(2μ/mₑ)E gives E[K] = α κ² exactly. The T-derivatives
/// reuse the SAME T-independent δ_l(k) table via κ-moments J_p = ∫ e^{-ακ²/T} S(κ) κ (ακ²)^p dκ.
pub fn quantum_b2_parts(sp: Species, t: f64) -> (f64, f64, f64) {
    let v = potential(sp);
    let mu = mu_pair(sp);
    let m = mass_amu(sp) * AMU_ME;          // atomic mass in mₑ (= 2μ)
    let beta = HARTREE_K / t;               // 1/Hartree
    let lambda_t = (2.0 * PI * beta / m).sqrt();             // λ_T in Bohr (h=2π a.u.)
    let na_l3 = N_A * (SQRT2 * lambda_t).powi(3) * A0_CM.powi(3); // N_A Λ³ [cm³/mol], Λ=√2 λ_T
    let alpha = HARTREE_K / m;              // α = (mₑ/m_atom) E_h/k_B  [K]

    // --- thermal (scattering) term: κ-moments of S(κ) ---
    // J_p = ∫ e^{-ακ²/T} S(κ) κ (ακ²)^p dκ, trapezoid in κ; cutoff κ where e^{-ακ²/T}≈e^{-40}.
    let lmax = 40usize;
    let nk = 1200usize;
    let kmax = (40.0 * t / alpha).sqrt();    // adapts with √T (high T needs larger κ)
    let hk = kmax / nk as f64;
    let (mut j0, mut j1, mut j2) = (0.0_f64, 0.0_f64, 0.0_f64);
    for ik in 0..=nk {
        let k = ik as f64 * hk;
        if k == 0.0 { continue; }            // integrand ∝ κ → 0
        let dl = crate::quantum::phase_shifts(&v, mu, k, lmax, 1e-4, 400.0, 4000);
        let mut s = 0.0;
        for l in 0..=lmax { s += l_weight(sp, l) * (2 * l + 1) as f64 * dl[l]; }
        let e = alpha * k * k;               // E in K
        let w = (-e / t).exp() * s * k;      // e^{-ακ²/T} S(κ) κ
        let tw = if ik == nk { 0.5 } else { 1.0 } * hk; // trapezoid weight
        j0 += w * tw;
        j1 += w * e * tw;
        j2 += w * e * e * tw;
    }
    // B_th = A·J0, A = -2 N_A Λ³ α/(π T) ∝ T^{-5/2}; J0'(T)=J1/T², J0''=J2/T⁴-2J1/T³.
    let a = -2.0 * na_l3 * alpha / (PI * t);
    let b_th = a * j0;
    let b_th_d1 = a * (-2.5 / t * j0 + j1 / (t * t));
    let b_th_d2 = a * (8.75 / (t * t) * j0 - 7.0 * j1 / (t * t * t) + j2 / t.powi(4));

    // --- ideal-gas (exchange) term: B_ideal = c N_A Λ³ ∝ T^{-3/2} ---
    let b_id = c_ideal(sp) * na_l3;
    let b_id_d1 = -1.5 / t * b_id;
    let b_id_d2 = 3.75 / (t * t) * b_id;

    // --- bound-state term (⁴He dimer only): B_bound = -N_A Λ³ [e^{T_b/T}-1] ---
    let (mut b_bd, mut b_bd_d1, mut b_bd_d2) = (0.0, 0.0, 0.0);
    if let Species::He4 = sp {
        if let Some(eb) = crate::quantum::s_wave_bound_energy(&v, mu) {
            let tb = -eb * HARTREE_K;        // T_b = |E_b| in K (>0)
            let p = -na_l3;                  // P = -N_A Λ³ ∝ T^{-3/2}
            let ex = (tb / t).exp();
            let g = ex - 1.0;
            let gp = -tb / (t * t) * ex;
            let gpp = (tb * tb / t.powi(4) + 2.0 * tb / (t * t * t)) * ex;
            b_bd = p * g;
            b_bd_d1 = -1.5 / t * p * g + p * gp;
            b_bd_d2 = 3.75 / (t * t) * p * g + 2.0 * (-1.5 / t * p) * gp + p * gpp;
        }
    }

    (b_th + b_id + b_bd, b_th_d1 + b_id_d1 + b_bd_d1, b_th_d2 + b_id_d2 + b_bd_d2)
}
```

**Provenance (no fitting):** every constant above is from Hurly & Mehl 2007 Eqs. (9),(18)–(24): the −2/π and α prefactors of B_th, the −1/16 ideal coefficient (= −2^{−5/2} with Λ=√2 λ_T), the −[e^{T_b/T}−1] bound form, Λ=√2 λ_T, and the even-l S(κ) sum. The ³He pieces (c=+1/32, ¼/¾ weights) are the standard I=½ identical-fermion generalization. **Do not tune these to the table.** If Step 4 disagrees, the bug is numerical (κ/l grid, dimer E_b, units) — fix that, not the published constants. If after honest numerical effort a term is still off by a clean factor, STOP and report the discrepancy with the computed-vs-reference numbers rather than silently rescaling.

- [ ] **Step 3b: Sanity-check `classical_b2`** — verify `classical_b2(Species::He4, 500.0)` is finite and ≈ 11 cm³/mol (it is the high-T limit of `quantum_b2`). No scaffolding to remove; the trapezoid is already clean.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test he4_b2_matches_cencek quantum_b2_high_t_to_classical` (run each separately — cargo takes one filter). 4-D-free but the k×l×r loops are heavy: allow ~1–2 min. Expected: PASS directly from the published formula. Report the computed-vs-reference B2 at each T. If the 500 K point undershoots, the high-l Born tail is truncated — raise `lmax` (the Fig. 8 peak region at ~10 K needs only l≲20, so low-T is robust); do **not** touch the prefactors.

- [ ] **Step 5: Commit**

```bash
git add src/quantum.rs src/main.rs
git commit -m "$(cat <<'EOF'
Add Beth-Uhlenbeck B2 for 4He (Bose, even l) + classical high-T anchor

Thermal (scattering) + ideal (exchange) + bound (dimer) terms, transcribed
verbatim from Hurly & Mehl 2007 (J. Res. NIST 112, 75), Eqs. 9 & 18-24 --
the method Cencek 2012 used. No fitted constants; validated against the
Cencek 2012 table within its uncertainties and the classical high-T limit.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: T-derivatives → n_eff for ⁴He (vs the tabulated TB′, T²B″)

**Files:** Modify `src/quantum.rs` (add `quantum_b2_neff`); Test `src/main.rs`.

- [ ] **Step 1: Write the failing test**

```rust
    #[test]
    fn he4_neff_matches_cencek_and_fig8() {
        use potter_poc::quantum::{quantum_b2_neff, Species};
        // returns (B2, dB2/dT, d2B2/dT2, neff). Compare T*dB2/dT and T^2*d2B2/dT2 to
        // the tabulated TB', T^2B'' (looser band — derivatives are grid-sensitive).
        let refs = [(10.0, 41.022, -82.478), (100.0, 2.0908, -6.9989), (500.0, -1.87546, 0.98256)];
        for &(t, tbp, t2bpp) in &refs {
            let (_b, db, d2b, _ne) = quantum_b2_neff(Species::He4, t);
            assert!((t * db - tbp).abs() < 0.5 + 0.05 * tbp.abs(), "TB' T={t}: {} vs {tbp}", t * db);
            assert!((t * t * d2b - t2bpp).abs() < 1.0 + 0.05 * t2bpp.abs(), "T2B'' T={t}: {} vs {t2bpp}", t * t * d2b);
        }
        // Fig. 8: the 4He n_eff peaks at ~140 near 8-12 K.
        let peak = [6.0, 8.0, 10.0, 12.0, 15.0].iter()
            .map(|&t| quantum_b2_neff(Species::He4, t).3).fold(0.0_f64, f64::max);
        assert!(peak > 100.0, "4He n_eff peak {peak} (expect ~140)");
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test he4_neff_matches_cencek_and_fig8`
Expected: FAIL — `quantum_b2_neff` not found.

- [ ] **Step 3: Implement** — add to `src/quantum.rs`:

```rust
/// (B2 [cm^3/mol], dB2/dT, d2B2/dT2, n_eff) for a species at temperature T [K].
pub fn quantum_b2_neff(sp: Species, t: f64) -> (f64, f64, f64, f64) {
    let (b2, db, d2b) = quantum_b2_parts(sp, t);
    let neff = -3.0 * (b2 + t * db) / (2.0 * t * db + t * t * d2b);
    (b2, db, d2b, neff)
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test he4_neff_matches_cencek_and_fig8`
Expected: PASS — TB′/T²B″ near the tabulated values and the n_eff peak >100 (Fig. 8). If the derivatives are noisy (the k-grid finite-difference of dδ/dk), refine: increase `nk`, or compute the thermal T-derivatives by differentiating the analytic Boltzmann weight (already done) and ensure the dδ/dk grid is smooth. The B₂ value test (Task 4) must still pass.

- [ ] **Step 5: Commit**

```bash
git add src/quantum.rs src/main.rs
git commit -m "$(cat <<'EOF'
Add quantum n_eff (4He): T-derivatives reproduce Cencek TB'/T2B'' + Fig.8 peak

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: ³He (Fermi statistics) vs the tabulated table

**Files:** Test `src/main.rs` (the engine already dispatches `Species::He3`; this validates the Fermi weights). The statistics are **sourced, not fitted**: ³He has nuclear spin I=½, so the standard identical-fermion result gives the even-l (spatially-symmetric, nuclear-singlet) weight w_e = I/(2I+1) = ¼ and the odd-l (spatially-antisymmetric, nuclear-triplet) weight w_o = (I+1)/(2I+1) = ¾ — encoded in `l_weight(Species::He3, …)`. The ideal-gas (exchange) term flips sign and carries the 1/(2I+1)=½ spin-degeneracy factor: B_ideal = +2^{-5/2} λ_T³ N_A/2 = +N_A Λ³/32 — encoded as `c_ideal(Species::He3) = +1/32`. ³He has **no** bound dimer (Task 3), so B_bound = 0. This task confirms those a-priori weights reproduce the Cencek ³He table; it should pass with no code change.

- [ ] **Step 1: Write the test**

```rust
    #[test]
    fn he3_b2_matches_cencek() {
        use potter_poc::quantum::{quantum_b2, Species};
        // 3He B2 (cm^3/mol) vs Cencek 2012 (fermion, spin-1/2).
        let refs = [(4.0, -62.311, 0.1), (10.0, -16.200, 0.06), (100.0, 12.0385, 0.03), (500.0, 11.05373, 0.02)];
        for &(t, b, u) in &refs {
            let got = quantum_b2(Species::He3, t);
            assert!((got - b).abs() < u.max(0.15), "3He B2 T={t}: {got} vs {b}");
        }
    }
```

- [ ] **Step 2: Run**

Run: `cargo test he3_b2_matches_cencek`
Expected: PASS — the sourced Fermi weights (`l_weight`/`c_ideal` for `Species::He3`) reproduce the Cencek ³He table directly. If it FAILS, the bug is numerical (same κ/l grid as ⁴He) or a units slip, **not** the statistics (those are the a-priori I=½ values, already validated by the ⁴He boson case sharing the same code path). If it cannot be made to match within ~0.15 cm³/mol after honest numerical effort, STOP and report the computed-vs-reference numbers — do not loosen the band; we may ship ⁴He+Ne and defer ³He.

- [ ] **Step 3: Commit**

```bash
git add src/main.rs src/quantum.rs
git commit -m "$(cat <<'EOF'
Validate 3He B2 (Fermi statistics) vs the Cencek 2012 table

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Ne full-quantum vs our WK Ne (cross-check)

**Files:** Test `src/main.rs`.

- [ ] **Step 1: Write the test**

```rust
    #[test]
    fn ne_full_quantum_vs_wk() {
        use potter_poc::quantum::{quantum_b2, Species};
        use potter_poc::noblegas::neon_tt;
        // Full-quantum Ne (on the TT potential) should agree with our WK Ne at moderate
        // T (quantum corrections small for Ne), within ~1-2 cm^3/mol.
        for &t in &[100.0_f64, 300.0] {
            let q = quantum_b2(Species::Ne, t);
            let wk = neon_tt().b2(t, 3); // WK order-3 B2, cm^3/mol
            assert!((q - wk).abs() < 2.0, "Ne T={t}: full-Q {q} vs WK {wk}");
        }
    }
```

- [ ] **Step 2: Run**

Run: `cargo test ne_full_quantum_vs_wk`
Expected: PASS — the two independent quantum routes agree for Ne at moderate T (mutual validation). If off by more than ~2 cm³/mol, check the Ne potential a.u. conversion in `potential(Species::Ne)` and the even-l/Boltzmann treatment; small disagreement at the few-% level is acceptable (different methods), but a large gap signals a units bug.

- [ ] **Step 3: Commit**

```bash
git add src/main.rs
git commit -m "$(cat <<'EOF'
Cross-check: full-quantum Ne agrees with WK Ne at moderate T

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Native `quantum_b2_neff` wrapper + `poc_quantum_b2` wasm export

**Files:** Modify `src/lib.rs`; Test `src/main.rs`.

- [ ] **Step 1: Write the failing test**

```rust
    #[test]
    fn quantum_b2_neff_lib_matches_module() {
        use potter_poc::quantum_b2_neff_si;
        use potter_poc::quantum::{quantum_b2_neff, Species};
        let a = quantum_b2_neff_si(0, 10.0); // 0=4He
        let (b, db, d2b, ne) = quantum_b2_neff(Species::He4, 10.0);
        assert!((a[0] - b).abs() < 1e-9 && (a[1] - db).abs() < 1e-9
             && (a[2] - d2b).abs() < 1e-9 && (a[3] - ne).abs() < 1e-9);
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test quantum_b2_neff_lib_matches_module`
Expected: FAIL — `quantum_b2_neff_si` not found.

- [ ] **Step 3: Implement** — in `src/lib.rs`, after `molecule_b2_neff`, add:

```rust
/// Full-quantum B₂ (cm³/mol, K) for a species: 0=⁴He, 1=³He, 2=Ne. Returns
/// [B2, dB2/dT, d2B2/dT2, neff].
pub fn quantum_b2_neff_si(species: u32, t: f64) -> [f64; 4] {
    use crate::quantum::Species;
    let sp = match species { 0 => Species::He4, 1 => Species::He3, _ => Species::Ne };
    let (b, db, d2b, ne) = crate::quantum::quantum_b2_neff(sp, t);
    [b, db, d2b, ne]
}
```

In `mod wasm_exports`, add `quantum_b2_neff_si` to the `use super::{…}` line, and add this export after `poc_molecule`:

```rust
    /// Full-quantum B₂ (cm³/mol, K): writes [B2, dB2/dT, d2B2/dT2, neff] (4 f64) into
    /// `out`. `species`: 0=⁴He, 1=³He, 2=Ne. Unaligned writes.
    #[no_mangle]
    pub extern "C" fn poc_quantum_b2(species: u32, t: f64, out: *mut f64) {
        let vals = quantum_b2_neff_si(species, t);
        unsafe { for (k, v) in vals.iter().enumerate() { out.add(k).write_unaligned(*v); } }
    }
```

- [ ] **Step 4: Verify**

Run: `cargo test quantum_b2_neff_lib_matches_module` → PASS.
Run: `cargo build --lib --target wasm32-unknown-unknown` → MUST compile cleanly (only pre-existing `aot.rs` warning). Report the result.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/main.rs
git commit -m "$(cat <<'EOF'
Add quantum_b2_neff_si + poc_quantum_b2 wasm export (4He/3He/Ne)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Web — "full quantum" species group

**Files:** Modify `web/index.html.in`.

The real-fluids `compute()`/`renderPlotly`/table already handle cm³/mol rows with a `q` (quantum) overlay (from the molecules work). Add the species + a `quantumRow`.

- [ ] **Step 1: Add the species options** — in the `<select id="species">`, after the molecules optgroup, add:
```html
          <optgroup label="Quantum (full, phase-shift)">
            <option value="q:0">⁴He (full quantum)</option>
            <option value="q:1">³He (full quantum)</option>
            <option value="q:2">Ne (full quantum)</option>
          </optgroup>
```

- [ ] **Step 2: Add `quantumRow` + `speciesInfo` handling + plot label.**

(2a) After `moleculeRow`, add:
```javascript
// Full-quantum (phase-shift) row via the poc_quantum_b2 export (cm³/mol, K).
function quantumRow(species, t) {
  const out = ex.poc_alloc(32); // 4 * f64
  ex.poc_quantum_b2(species, t, out);
  const dv = new DataView(ex.memory.buffer);
  const r = Array.from({ length: 4 }, (_, k) => dv.getFloat64(out + 8 * k, true));
  ex.poc_dealloc(out, 32);
  return { T: t, num: r[0], db2: r[1], d2b2: r[2], neff: r[3], q: null, ex: null };
}
```

(2b) In `speciesInfo()`, handle the `q:` kind. The current function returns `{kind, id, name, qLabel}`. Add, before the noble fallback:
```javascript
  if (kind === "q") {
    const names = ["⁴He (full quantum)", "³He (full quantum)", "Ne (full quantum)"];
    return { kind, id, name: names[id], qLabel: null };
  }
```

(2c) In `compute()`'s real branch, the species dispatch currently handles `sp.kind === "mol"` vs noble. Add a `q` arm:
```javascript
    if (sp.kind === "q") {
      makeRow = (T) => quantumRow(sp.id, T);
    } else if (sp.kind === "mol") {
      makeRow = (T) => moleculeRow(sp.id, T);
    } else {
      const order = parseInt($("wkorder").value) || 3;
      makeRow = (T) => noblegasRow(sp.id, T, order);
    }
```

(2d) In `syncSpecies()`, hide the WK-order control for `q:` species too (they have no order), and set a sensible default T-range for the He cases (the n_eff peak is ~10 K). Replace the `mol` check with `mol || quantum`:
```javascript
function syncSpecies() {
  const val = $("species").value || "";
  const heavy = val.startsWith("mol") || val.startsWith("q:");
  $("wkorderwrap").hidden = heavy;
  if (val.startsWith("q:")) {
    if ((parseFloat($("rtmin").value) || 0) > 4) $("rtmin").value = 1;
    if ((parseFloat($("rtmax").value) || 0) > 600) $("rtmax").value = 500;
    const np = parseInt($("rnpts").value) || 24; if (np > 20) $("rnpts").value = 16;
  } else if (val.startsWith("mol")) {
    const np = parseInt($("rnpts").value) || 24; if (np > 14) $("rnpts").value = 12;
    if ((parseFloat($("rtmin").value) || 0) < 120) $("rtmin").value = 150;
  } else if ((parseFloat($("rtmin").value) || 0) >= 120) {
    $("rtmin").value = 50;
  }
}
```

- [ ] **Step 3: Build + syntax + screenshot**

Run: `./web/build.sh`
```bash
python3 - <<'PY'
import re
html=open("web/index.html.in").read()
m=re.search(r'<script>\n(.*)</script>', html, re.S)
open('/tmp/wj.js','w').write(m.group(1))
PY
node --check /tmp/wj.js && echo "JS SYNTAX OK"
grep -c 'q:0\|quantumRow\|poc_quantum_b2\|full quantum' docs/index.html
```
Expected: build OK, `JS SYNTAX OK`, grep ≥ 4. Then a manual/headless check: select ⁴He (full quantum), confirm the B₂(T) curve and the n_eff panel show the dramatic low-T peak (~140 near 10 K).

- [ ] **Step 4: Commit**

```bash
git add web/index.html.in docs/index.html
git commit -m "$(cat <<'EOF'
web: add full-quantum 4He/3He/Ne to real-fluids mode

Phase-shift B2(T)/n_eff via poc_quantum_b2; the 4He n_eff shows the
dramatic ~140 peak at ~10 K (the paper's Fig. 8). WK-order hidden;
He default T-range 1-500 K (log) to frame the peak.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Node e2e + full regression

**Files:** Create `node/quantum-e2e.mjs`; full suite.

- [ ] **Step 1: Rebuild the wasm**

Run: `cargo build --release --target wasm32-unknown-unknown --lib`

- [ ] **Step 2: Create `node/quantum-e2e.mjs`**

```javascript
// e2e: poc_quantum_b2 through the built wasm vs the Cencek 2012 tabulated B2.
import fs from "node:fs";
const bytes = fs.readFileSync("target/wasm32-unknown-unknown/release/potter_poc.wasm");
const { instance } = await WebAssembly.instantiate(bytes, {});
const ex = instance.exports;
function q(species, t) {
  const out = ex.poc_alloc(32); ex.poc_quantum_b2(species, t, out);
  const dv = new DataView(ex.memory.buffer);
  const r = [0,1,2,3].map(k => dv.getFloat64(out + 8*k, true));
  ex.poc_dealloc(out, 32); return r;
}
let ok = true;
// 4He B2 vs Cencek
for (const [t, b] of [[10, -23.125], [100, 11.6747], [500, 11.00715]]) {
  const r = q(0, t), pass = Math.abs(r[0] - b) < 0.15; ok &&= pass;
  console.log(`4He T=${t}: B2=${r[0].toFixed(4)} (ref ${b}) neff=${r[3].toFixed(2)} ${pass?"OK":"FAIL"}`);
}
// 4He n_eff peak near 10 K
const peak = [8,10,12].map(t => q(0,t)[3]).reduce((a,b)=>Math.max(a,b),0);
const pk = peak > 100; ok &&= pk;
console.log(`4He n_eff peak ~10K = ${peak.toFixed(1)} (expect ~140) ${pk?"OK":"FAIL"}`);
console.log(ok ? "E2E PASS" : "E2E FAIL"); process.exit(ok ? 0 : 1);
```

- [ ] **Step 3: Run**

Run: `node node/quantum-e2e.mjs`
Expected: `E2E PASS` — ⁴He B₂ matches Cencek through the wasm ABI and the n_eff peak >100. If FAIL, STOP and report the values.

- [ ] **Step 4: Full Rust regression**

Run: `cargo test`
Expected: all prior tests + the new quantum tests pass. The quantum tests are heavy (phase-shift integrations) — allow ~5–8 min; if the suite is too slow, mark the heaviest quantum tests `#[ignore]` with a note and run them via `cargo test -- --ignored` in CI. Paste the final `test result:` line.

- [ ] **Step 5: Commit**

```bash
git add node/quantum-e2e.mjs
git commit -m "$(cat <<'EOF'
Add node e2e for poc_quantum_b2 (4He vs Cencek, n_eff peak)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review (against the spec)

**Spec coverage:**
- §2 potential port → Task 1 (validated vs compiled-Fortran table). ✓
- §3 variable-phase δ_l(k) + Riccati-Bessel → Task 2 (square-well analytic check). ✓
- §4 bound states (dimer, Levinson) → Task 3. ✓
- §5 Beth–Uhlenbeck B₂ + statistics (⁴He even-l; ³He Fermi; Ne) → Tasks 4, 6, 7. ✓
- §6 n_eff + T-derivatives → Task 5. ✓
- §7 wasm + web → Tasks 8, 9. ✓
- §8 validation (port table, ⁴He/³He vs tabulated within uncertainties, dimer, Ne-vs-WK, high-T classical, Fig. 8) → Tasks 1,3,4,5,6,7,10. ✓ (LM2M2 method check from §8.7 is *subsumed* by the square-well analytic check in Task 2 + the Cencek validation — noted as a deliberate simplification: the square well is a cleaner closed-form engine check than LM2M2.)
- §9 out of scope (mixtures, 2017 potential, B₃, H₂) → not in any task. ✓
- §10 risks → addressed (³He gated in Task 6 with a stop-and-report; Task 4 uses the published Hurly–Mehl formula with no fitted constants — a discrepancy is a numerical bug to fix, not a constant to tune; dimer convergence tuning in Task 3; perf via `#[ignore]` note in Task 10).

**Placeholder scan:** no tuned constants. Task 4's prefactors are transcribed verbatim from Hurly & Mehl 2007 Eqs. (9),(18)–(24) (every line of `quantum_b2_parts` maps to a published equation in its doc-comment); the Cencek table is a *validation* anchor, not a fit target. All steps have complete code + exact commands + concrete numeric anchors.

**Type consistency:** `He{He4,He3}`, `v_he`/`v_components`/`reduced_mass_me` (Task 1); `reduced_mass_me` is used by Task 3's standalone dimer test, while the shipped virial path (Task 4) uses atomic-mass `mu_pair` (= m_atom/2) for consistency with α — both recover the ~−1.1 mK ⁴He dimer within Task 3's tolerance. `riccati`/`phase_shifts`/`s_wave_phase_for_test` (Task 2) used in Task 4. `s_wave_bound_energy` (Task 3) used in Task 4. `Species{He4,He3,Ne}`, `quantum_b2`/`quantum_b2_parts`/`quantum_b2_neff`/`classical_b2` and helpers `l_weight`/`c_ideal`/`mu_pair` (Tasks 4–5) used in Tasks 6,7,8. `quantum_b2_neff_si`/`poc_quantum_b2` (Task 8) used in Tasks 9,10. Web `quantumRow`/`speciesInfo q:`/`syncSpecies` consistent (Task 9). Row shape `{T,num,db2,d2b2,neff,q,ex}` matches the existing real-fluids consumers.
