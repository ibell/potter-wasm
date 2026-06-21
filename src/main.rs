//! Native / WASI demo: B2 and B3 for Lennard-Jones (12-6) from a DSL string,
//! with independent self-verification.

use potter_poc::{
    b2, b2_finegrid, b2_lj_series, b2_v, b3, b3_v, b3_v_grid, CsePotential, FastPotential,
    Potential, LJ_BOYLE_TSTAR,
};

const LJ: &str = "4*eps*((sig/r)**12 - (sig/r)**6)";

/// Hard-sphere "potential" (diameter 1): +inf inside, 0 outside. Used as an exact
/// analytic anchor — B2 = 2*pi/3, and B3 = (5/8) B2^2 exactly.
fn hard_sphere(r: f64) -> f64 {
    if r < 1.0 {
        f64::INFINITY
    } else {
        0.0
    }
}

fn main() {
    println!("potter-in-Rust POC — B2 & B3 for Lennard-Jones (12-6) via a Python-like DSL");
    println!("  potential string:  V(r) = {}", LJ);
    println!("  reduced units:     eps = sig = 1\n");

    let pot = Potential::compile(LJ, 1.0, 1.0).expect("DSL failed to compile");

    println!("{:>8}   {:>13}   {:>13}   {:>13}", "T*", "B2 (DSL)", "B2 (series)", "B3 (DSL)");
    println!("  {}", "-".repeat(56));
    let grid = [1.3, 1.5, 2.0, 2.5, 3.0, LJ_BOYLE_TSTAR, 5.0, 10.0];
    for &t in &grid {
        let b2v = b2(&pot, t, 1e-12);
        let b3v = b3(&pot, t, 1e-6);
        let series = if t >= 1.8 {
            format!("{:>13.6}", b2_lj_series(t, 60))
        } else {
            format!("{:>13}", "(slow conv.)")
        };
        let tag = if (t - LJ_BOYLE_TSTAR).abs() < 1e-6 {
            "  <- Boyle T"
        } else {
            ""
        };
        println!("{:>8.4}   {:>13.6}   {}   {:>13.6}{}", t, b2v, series, b3v, tag);
    }

    // ----------------------------- verification -----------------------------
    println!("\nverification:");

    // (1) B2 adaptive vs an independent 2M-panel grid
    let mut md = 0.0f64;
    for &t in &grid {
        md = md.max((b2(&pot, t, 1e-13) - b2_finegrid(&pot, t, 2_000_000)).abs());
    }
    println!("  (1) B2 adaptive vs 2M grid       : max |diff| = {:.2e}", md);

    // (2) physics anchor: B2 = 0 at the Boyle temperature
    println!(
        "  (2) B2 at Boyle T*={:.5}        : {:+.2e}  (expect ~0)",
        LJ_BOYLE_TSTAR,
        b2(&pot, LJ_BOYLE_TSTAR, 1e-13)
    );

    // (3) B2 vs closed-form series at high T*
    let s5 = b2_lj_series(5.0, 60);
    println!(
        "  (3) B2 DSL vs closed-form @ T*=5 : rel.err = {:.2e}",
        ((b2(&pot, 5.0, 1e-12) - s5) / s5).abs()
    );

    // (4) B3 adaptive vs an independent nested fixed grid
    let b3a = b3(&pot, 2.0, 1e-8);
    let b3g = b3_v_grid(&|r| pot.v(r), 2.0, 256);
    println!(
        "  (4) B3 adaptive vs grid @ T*=2   : adaptive={:.6} grid={:.6} (|diff|={:.1e})",
        b3a,
        b3g,
        (b3a - b3g).abs()
    );

    // (5) hard-sphere exact anchor: B2 = 2*pi/3, B3 = (5/8) B2^2  (validates the
    //     B3 formula + constant against an exact analytic result)
    let b2_hs = b2_v(&hard_sphere, 1.0, 1e-13);
    let b3_hs = b3_v(&hard_sphere, 1.0, 1e-8);
    let b2_hs_exact = 2.0 * std::f64::consts::PI / 3.0;
    let b3_hs_exact = 0.625 * b2_hs_exact * b2_hs_exact;
    println!(
        "  (5) hard sphere: B2={:.6} (exact {:.6}); B3={:.5} (exact {:.5}, 5/8*B2^2)",
        b2_hs, b2_hs_exact, b3_hs, b3_hs_exact
    );

    // (6) fasteval backend agrees with the hand-rolled DSL
    let fast = FastPotential::compile(LJ, 1.0, 1.0).expect("fasteval compile");
    let db2 = (b2(&pot, 2.0, 1e-12) - b2_v(&|r| fast.v(r), 2.0, 1e-12)).abs();
    let db3 = (b3(&pot, 2.0, 1e-7) - b3_v(&|r| fast.v(r), 2.0, 1e-7)).abs();
    println!(
        "  (6) fasteval vs hand DSL @ T*=2  : |dB2|={:.1e}  |dB3|={:.1e}",
        db2, db3
    );

    // (7) CSE/bytecode backend agrees with the tree-walk DSL
    let cse = CsePotential::compile(LJ, 1.0, 1.0).expect("cse compile");
    let cb2 = (b2(&pot, 2.0, 1e-12) - b2_v(&|r| cse.v(r), 2.0, 1e-12)).abs();
    println!(
        "  (7) CSE vs hand DSL @ T*=2       : |dB2|={:.1e}  ({} ops after CSE)",
        cb2,
        cse.ops()
    );
}

#[cfg(test)]
mod tests {
    use potter_poc::{
        b2, b2_finegrid, b2_lj_series, b2_v, b3, b3_v, b3_v_grid, dsl, CsePotential, FastPotential,
        Potential, LJ_BOYLE_TSTAR,
    };

    const LJ: &str = "4*eps*((sig/r)**12 - (sig/r)**6)";
    fn pot() -> Potential {
        Potential::compile(LJ, 1.0, 1.0).unwrap()
    }
    fn hard_sphere(r: f64) -> f64 {
        if r < 1.0 {
            f64::INFINITY
        } else {
            0.0
        }
    }

    #[test]
    fn dsl_matches_hardcoded_lj() {
        let p = pot();
        for &r in &[0.9_f64, 1.0, 1.1, 1.259_921, 1.5, 2.0, 3.0] {
            let hard = 4.0 * ((1.0 / r).powi(12) - (1.0 / r).powi(6));
            assert!((p.v(r) - hard).abs() < 1e-12, "r={r}");
        }
    }

    #[test]
    fn dsl_python_like_precedence() {
        let env: [f64; 0] = [];
        let ev = |s: &str| dsl::eval(&dsl::compile(s, &[]).unwrap(), &env);
        assert_eq!(ev("-2**2"), -4.0);
        assert_eq!(ev("2**-1"), 0.5);
        assert_eq!(ev("2**3**2"), 512.0);
        assert_eq!(ev("-2*3"), -6.0);
    }

    #[test]
    fn b2_adaptive_matches_finegrid() {
        let p = pot();
        for &t in &[0.8, 1.0, 2.0, 3.0, 5.0, 10.0] {
            let a = b2(&p, t, 1e-13);
            let f = b2_finegrid(&p, t, 2_000_000);
            assert!((a - f).abs() < 1e-6, "T*={t}");
        }
    }

    #[test]
    fn b2_boyle_temperature_is_zero() {
        assert!(b2(&pot(), LJ_BOYLE_TSTAR, 1e-13).abs() < 5e-3);
    }

    #[test]
    fn b2_matches_closed_form_series() {
        let p = pot();
        for &t in &[2.0, 3.0, 5.0, 10.0] {
            let num = b2(&p, t, 1e-12);
            let ser = b2_lj_series(t, 60);
            assert!(((num - ser) / ser).abs() < 1e-4, "T*={t}");
        }
    }

    #[test]
    fn co2_qfh_matches_si_tabulated() {
        // Quadratic Feynman-Hibbs B2 vs Hellmann CO2 SI tabulated B_QFH (V_B).
        // mu = reduced mass of the CO2-CO2 pair = M(CO2)/2 = 22.0045 amu (the
        // two-body translational reduced mass; nothing to do with C/O masses).
        // I = 43.202 amu A^2 (CO2 moment of inertia).
        use potter_poc::molecule::co2_hellmann;
        let co2 = co2_hellmann();
        for &(t, b_qfh) in &[(250.0, -184.16), (400.0, -59.87), (700.0, -1.30)] {
            let (qfh, _) = co2.b2_qfh(t, 1e-4, 22.0045, 43.202);
            assert!((qfh - b_qfh).abs() < 0.1, "T={t}: QFH {qfh} vs SI {b_qfh}");
        }
    }

    #[test]
    fn n2_quantum_correction_matches_si() {
        // First-order Wigner-Kirkwood quantum correction vs Hellmann SI column 3
        // (B2 classical + quantum). mu = reduced mass of the N2-N2 pair =
        // M(N2)/2 = 14.0067 amu (two-body translational reduced mass); I = 8.473.
        use potter_poc::molecule::n2_hellmann;
        let n2 = n2_hellmann();
        for &(t, si_clqc) in &[(90.0, -195.57), (200.0, -35.66), (500.0, 16.61)] {
            let (qc, _) = n2.b2_quantum(t, 1e-4, 14.0067, 8.473);
            assert!(
                (qc - si_clqc).abs() < 0.1,
                "T={t}: cl+qc {qc} vs SI {si_clqc}"
            );
        }
    }

    #[test]
    fn molecule_b2_and_derivs_classical_matches_b2() {
        // The vector-cubature classical B2 component must reproduce the scalar b2().
        use potter_poc::molecule::{co2_hellmann, n2_trappe};
        let n2 = n2_trappe(); // Linear (LJ + charge), rmin=0
        for &t in &[150.0_f64, 300.0] {
            let (d, _) = n2.b2_and_derivs(t, 1e-3);
            let (b, _) = n2.b2(t, 1e-3);
            assert!((d.b2 - b).abs() < 0.2, "TraPPE N2 T={t}: {} vs b2 {}", d.b2, b);
        }
        let co2 = co2_hellmann(); // RigidLinear, rmin=2
        for &t in &[250.0_f64, 500.0] {
            let (d, _) = co2.b2_and_derivs(t, 1e-3);
            let (b, _) = co2.b2(t, 1e-3);
            assert!((d.b2 - b).abs() < 0.3, "Hellmann CO2 T={t}: {} vs b2 {}", d.b2, b);
        }
    }

    #[test]
    fn molecule_qfh_derivs_match_tabulated() {
        // QFH B2 component vs the published SI tabulated values (reltol 1e-3, the web
        // tolerance — verified to hold the bands tightly).
        use potter_poc::molecule::{co2_hellmann, n2_hellmann};
        // CO2: SI tabulates QFH directly (B_QFH col5) -> tight band.
        let co2 = co2_hellmann();
        for &(t, b_qfh) in &[(250.0, -184.16), (400.0, -59.87), (700.0, -1.30)] {
            let (d, _) = co2.b2_qfh_and_derivs(t, 1e-3, 22.0045, 43.202);
            assert!((d.b2 - b_qfh).abs() < 0.05, "CO2 QFH T={t}: {} vs SI {b_qfh}", d.b2);
            assert!(d.neff(t).is_finite() && d.neff(t) > 0.0, "CO2 neff {}", d.neff(t));
        }
        // N2: SI tabulates WK; QFH differs by the higher-order resummation (measured
        // ~0.12 cm^3/mol worst, at the coldest 90 K) -> band 0.2.
        let n2 = n2_hellmann();
        for &(t, si_wk) in &[(90.0, -195.57), (200.0, -35.66), (500.0, 16.61)] {
            let (d, _) = n2.b2_qfh_and_derivs(t, 1e-3, 14.0067, 8.473);
            assert!((d.b2 - si_wk).abs() < 0.2, "N2 QFH T={t}: {} vs SI(WK) {si_wk}", d.b2);
        }
    }

    #[test]
    fn molecule_qfh_db2dt_matches_fd() {
        // dB2/dT (analytic chain-rule) vs a central FD of b2_qfh (CO2 @ 300 K).
        // One point, reltol 1e-4 + h=2 K so the FD noise stays below the 3% band.
        use potter_poc::molecule::co2_hellmann;
        let co2 = co2_hellmann();
        let (mu, i) = (22.0045, 43.202);
        let (d, _) = co2.b2_qfh_and_derivs(300.0, 1e-4, mu, i);
        let h = 2.0;
        let (bp, _) = co2.b2_qfh(300.0 + h, 1e-4, mu, i);
        let (bm, _) = co2.b2_qfh(300.0 - h, 1e-4, mu, i);
        let fd = (bp - bm) / (2.0 * h);
        assert!((d.db2_dt - fd).abs() / fd.abs() < 0.03, "CO2 dB2/dT {} vs FD {fd}", d.db2_dt);
    }

    #[test]
    fn hellmann_energy_matches_potter() {
        // Validate the coded potentials against potter's exact reference energies
        // (r in A; angles theta1, theta2, phi; V/kB in K).
        use potter_poc::molecule::{co2_hellmann, n2_hellmann};
        let d = std::f64::consts::PI / 180.0;
        let n2 = n2_hellmann();
        assert!((n2.energy(3.75, 90.0 * d, 90.0 * d, 0.0) - (-113.486)).abs() < 0.05);
        assert!((n2.energy(5.00, 90.0 * d, 90.0 * d, 0.0) - (-27.097)).abs() < 0.05);
        assert!((n2.energy(3.00, 90.0 * d, 90.0 * d, 90.0 * d) - 418.826).abs() < 0.5);
        let co2 = co2_hellmann();
        assert!((co2.energy(4.00, 0.0, 0.0, 0.0) - 35408.788).abs() < 2.0);
        assert!((co2.energy(5.75, 0.0, 0.0, 0.0) - (-24.133)).abs() < 0.05);
        assert!((co2.energy(6.00, 0.0, 0.0, 0.0) - (-12.275)).abs() < 0.05);
    }

    #[test]
    fn molecule_single_site_matches_spherical() {
        use potter_poc::b2_v;
        use potter_poc::molecule::{Linear, Site};
        // A 1-site molecule (no orientation dependence) must reproduce the
        // spherical LJ B2 exactly: the 4-D orientational integral collapses.
        let m = Linear {
            sites: vec![Site { d: 0.0, eps: 1.0, sig: 1.0, q: 0.0 }],
        };
        let lj = |r: f64| {
            let s6 = (1.0_f64 / r).powi(6);
            4.0 * (s6 * s6 - s6)
        };
        for &t in &[1.5_f64, 2.0, 5.0] {
            let (b2_cm3mol, _) = m.b2(t, 1e-7);
            let b2_ang3 = b2_cm3mol / 0.602214; // back to sigma^3 (=Angstrom^3) units
            let spherical = b2_v(&lj, t, 1e-10);
            assert!(
                (b2_ang3 - spherical).abs() / spherical.abs() < 5e-3,
                "T*={t}: molecule {b2_ang3} vs spherical {spherical}"
            );
        }
    }

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
        let v56 = potter_poc::he_potential::v_he(He::He4, 5.6, false) * TOK;
        assert!((v56 - (-10.9957)).abs() < 0.01, "He4 V(5.6)={v56}");
        let d4 = potter_poc::he_potential::v_he(He::He4, 5.6, false);
        let d3 = potter_poc::he_potential::v_he(He::He3, 5.6, false);
        assert!((d4 - d3).abs() > 0.0, "3He != 4He potential");
    }

    #[test]
    fn msmc_b3_matches_cubature() {
        use potter_poc::b3_cubature_v;
        use potter_poc::msmc::msmc_b3_v;
        let p = pot();
        // Mayer-sampling MC reproduces the deterministic B3 within statistical error.
        for &t in &[1.5_f64, 2.0] {
            let (cub, _) = b3_cubature_v(&|r| p.v(r), t, 1e-8);
            let r = msmc_b3_v(&|r| p.v(r), t, 1.5, 4_000_000, 0xC0FFEE);
            assert!(
                (r.b3 - cub).abs() / cub.abs() < 0.02,
                "T*={t}: MSMC {} +/- {} vs cubature {cub}",
                r.b3,
                r.stderr
            );
        }
    }

    #[test]
    fn b3_cubature_matches_nested_and_hardsphere() {
        use potter_poc::{b3_cubature_v, b3_v};
        let p = pot();
        // Genz-Malik hcubature agrees with the (validated) nested-1D integrator.
        for &t in &[1.0, 1.5, 2.0, 5.0] {
            let nested = b3_v(&|r| p.v(r), t, 1e-9);
            let (cub, _n) = b3_cubature_v(&|r| p.v(r), t, 1e-8);
            assert!((nested - cub).abs() < 1e-5, "T*={t} nested={nested} cub={cub}");
        }
        // Hard-sphere exact anchor B3 = (5/8) B2^2 via cubature.
        let (b3hs, _) = b3_cubature_v(&hard_sphere, 1.0, 1e-7);
        let b2x = 2.0 * std::f64::consts::PI / 3.0;
        let b3x = 0.625 * b2x * b2x;
        assert!((b3hs - b3x).abs() / b3x < 1e-2, "B3_HS {b3hs} vs {b3x}");
    }

    #[test]
    fn b3_adaptive_matches_grid() {
        let p = pot();
        for &t in &[1.5, 2.0, 3.0] {
            let a = b3(&p, t, 1e-8);
            let g = b3_v_grid(&|r| p.v(r), t, 256);
            assert!((a - g).abs() < 1e-3, "T*={t} adaptive={a} grid={g}");
        }
    }

    #[test]
    fn hard_sphere_b2_b3_exact() {
        // B2_HS = 2*pi/3 ; B3_HS = (5/8) B2^2 — exact analytic values.
        let b2v = b2_v(&hard_sphere, 1.0, 1e-13);
        let b3v = b3_v(&hard_sphere, 1.0, 1e-8);
        let b2x = 2.0 * std::f64::consts::PI / 3.0;
        let b3x = 0.625 * b2x * b2x;
        assert!((b2v - b2x).abs() < 1e-6, "B2_HS {b2v} vs {b2x}");
        assert!((b3v - b3x).abs() / b3x < 1e-2, "B3_HS {b3v} vs {b3x}");
    }

    #[test]
    fn aot_emits_valid_wasm() {
        use potter_poc::aot::{compile_to_wasm, validate};
        // LJ is pure arithmetic + integer powers -> self-contained module.
        let lj = compile_to_wasm(LJ, &["r", "eps", "sig"]).unwrap();
        validate(&lj).expect("LJ wasm should validate");
        assert!(lj.starts_with(&[0x00, 0x61, 0x73, 0x6d]), "wasm magic header");
        // A transcendental potential validates too (imports env.exp).
        let morse = compile_to_wasm("eps*exp(-(r-sig))", &["r", "eps", "sig"]).unwrap();
        validate(&morse).expect("morse wasm should validate");
    }

    #[test]
    fn jit_matches_tree_dsl() {
        use potter_poc::JitPotential;
        let p = pot();
        let j = JitPotential::compile(LJ, 1.0, 1.0).unwrap();
        for &r in &[0.95_f64, 1.0, 1.2, 1.5, 2.0, 3.0] {
            assert!((p.v(r) - j.v(r)).abs() < 1e-12, "v(r) mismatch at r={r}");
        }
        for &t in &[1.5, 2.0, 5.0] {
            assert!((b3(&p, t, 1e-7) - b3_v(&|r| j.v(r), t, 1e-7)).abs() < 1e-9, "B3 T*={t}");
        }
    }

    #[test]
    fn vector_adaptive_simpson_integrates_each_component() {
        use potter_poc::integrate::adaptive_simpson3;
        // f(x) = [1, x, x^2] over [0,1] -> [1, 1/2, 1/3], all on ONE shared grid.
        let i = adaptive_simpson3(&|x| [1.0, x, x * x], 0.0, 1.0, 1e-12, 50);
        assert!((i[0] - 1.0).abs() < 1e-10, "got {}", i[0]);
        assert!((i[1] - 0.5).abs() < 1e-10, "got {}", i[1]);
        assert!((i[2] - 1.0 / 3.0).abs() < 1e-10, "got {}", i[2]);
    }

    #[test]
    fn jit_handles_transcendentals() {
        // A potential using exp/sqrt exercises the libm-shim call path in the JIT.
        use potter_poc::JitPotential;
        let src = "eps*exp(-r/sig) + sqrt(r)";
        let p = Potential::compile(src, 2.0, 1.5).unwrap();
        let j = JitPotential::compile(src, 2.0, 1.5).unwrap();
        for &r in &[0.5_f64, 1.0, 2.0, 4.0] {
            assert!((p.v(r) - j.v(r)).abs() < 1e-12, "transcendental mismatch at r={r}");
        }
    }

    #[test]
    fn cse_matches_tree_dsl() {
        let p = pot();
        let c = CsePotential::compile(LJ, 1.0, 1.0).unwrap();
        // CSE collapses the shared (sig/r): fewer ops than the tree node count.
        assert!(c.ops() <= 11, "expected CSE to shrink op count, got {}", c.ops());
        for &r in &[0.95_f64, 1.0, 1.2, 1.5, 2.0, 3.0] {
            assert!((p.v(r) - c.v(r)).abs() < 1e-12, "v(r) mismatch at r={r}");
        }
        for &t in &[1.5, 2.0, 5.0] {
            assert!((b2(&p, t, 1e-12) - b2_v(&|r| c.v(r), t, 1e-12)).abs() < 1e-12);
            assert!((b3(&p, t, 1e-7) - b3_v(&|r| c.v(r), t, 1e-7)).abs() < 1e-9, "B3 T*={t}");
        }
    }

    #[test]
    fn fasteval_matches_hand_dsl() {
        let p = pot();
        let f = FastPotential::compile(LJ, 1.0, 1.0).unwrap();
        for &r in &[0.95_f64, 1.0, 1.2, 1.5, 2.0] {
            assert!((p.v(r) - f.v(r)).abs() < 1e-12, "v(r) r={r}");
        }
        for &t in &[1.5, 2.0, 5.0] {
            assert!((b2(&p, t, 1e-12) - b2_v(&|r| f.v(r), t, 1e-12)).abs() < 1e-9);
            assert!((b3(&p, t, 1e-7) - b3_v(&|r| f.v(r), t, 1e-7)).abs() < 1e-6, "B3 T*={t}");
        }
    }

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
        // (c) n_eff is finite
        assert!(d.neff(t).is_finite(), "neff not finite: {}", d.neff(t));
    }

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
                assert!((ne - n as f64).abs() < 1e-3, "n={n} T*={t}: n_eff={ne}");
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
            assert!(((num.d2b2_dt2 - ser.d2b2_dt2) / ser.d2b2_dt2).abs() < 1e-3, "B2'' T*={t}");
            assert!((num.neff(t) - ser.neff(t)).abs() < 1e-3, "neff T*={t}");
        }
    }

    #[test]
    fn lj_neff_high_temperature_limit_is_twelve() {
        use potter_poc::b2_lj_series_derivs;
        // Leading HCB term ~ T*^{-1/4} = T^{-3/n} with n = 12 -> n_eff -> 12.
        let ne = b2_lj_series_derivs(1e6, 60).neff(1e6);
        assert!((ne - 12.0).abs() < 0.05, "n_eff(1e6)={ne}");
        let ne2 = b2_lj_series_derivs(1e4, 60).neff(1e4);
        assert!((ne2 - 12.0).abs() < 0.3, "n_eff(1e4)={ne2}");
    }

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
        assert!((v2 - 4.0).abs() < 1e-12);
        assert!((e1 - 10.0).abs() < 1e-12);
        assert!((e2 - 12.0).abs() < 1e-12);
        // exp/recip on a dual: value AND first derivative (pins the derivative path)
        use num_dual::first_derivative;
        let (g, dg) = first_derivative(|x: num_dual::Dual64| (-x).exp() * x.recip(), 1.0);
        assert!((g - std::f64::consts::E.recip()).abs() < 1e-12); // e^{-1}
        assert!((dg - (-2.0 * std::f64::consts::E.recip())).abs() < 1e-12); // -2/e
    }

    #[test]
    fn noblegas_wk3_b2() {
        use potter_poc::noblegas::{neon_tt, argon_tt, krypton_tt, xenon_tt, TangToennies};
        // WK order-3 B2 [cm^3/mol] vs integrate_potentials.py reference
        let cases: &[(&str, fn() -> TangToennies, [(f64, f64); 4])] = &[
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
            // the quantum correction is a real, nonzero shift off classical at low T
            assert!((g.b2(50.0, 3) - g.b2(50.0, 0)).abs() > 0.0);
        }
    }

    #[test]
    fn noblegas_classical_b2() {
        use potter_poc::noblegas::{neon_tt, argon_tt, krypton_tt, xenon_tt, TangToennies};
        // classical B2 [cm^3/mol] vs integrate_potentials.py reference
        let cases: &[(&str, fn() -> TangToennies, [(f64, f64); 4])] = &[
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

    #[test]
    fn noblegas_grid_reuse_matches_b2_neff() {
        use potter_poc::noblegas::argon_tt;
        let g = argon_tt();
        let pv = g.grid();
        assert_eq!(pv.len(), 10000, "grid size");
        for &(t, order) in &[(120.0_f64, 0u8), (300.0, 3), (800.0, 1)] {
            let a = g.b2_neff_with_grid(t, order, &pv);
            let b = g.b2_neff(t, order);
            assert!((a.0 - b.0).abs() < 1e-12 && (a.1 - b.1).abs() < 1e-12
                 && (a.2 - b.2).abs() < 1e-12 && (a.3 - b.3).abs() < 1e-12,
                 "T={t} order={order}: {a:?} vs {b:?}");
        }
    }

    #[test]
    fn noblegas_b2_neff_array_classical_and_quantum() {
        use potter_poc::{noblegas::argon_tt, noblegas_b2_neff};
        // gas 1 = Argon; order 3 quantum. Returns
        // [b2_cl,db2_cl,d2b2_cl,neff_cl, b2_q,db2_q,d2b2_q,neff_q].
        let a = noblegas_b2_neff(1, 300.0, 3);
        let g = argon_tt();
        let cl = g.b2_neff(300.0, 0);
        let q = g.b2_neff(300.0, 3);
        assert!((a[0] - cl.0).abs() < 1e-12 && (a[3] - cl.3).abs() < 1e-12, "classical slot");
        assert!((a[4] - q.0).abs() < 1e-12 && (a[7] - q.3).abs() < 1e-12, "quantum slot");
        // quantum correction is a real shift at low T
        let lo = noblegas_b2_neff(0, 50.0, 3); // Neon, 50 K
        assert!((lo[0] - lo[4]).abs() > 1e-6, "Ne 50K classical vs WK differ");
    }

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

    // NOTE: the full-quantum B2/n_eff tests evaluate the phase-shift engine at several
    // temperatures and take ~1-3 min each in release (far longer in debug), so they are
    // #[ignore]d to keep routine `cargo test` fast. Run them explicitly with:
    //   cargo test --release -- --ignored
    #[test]
    #[ignore = "heavy phase-shift integration (~1 min release); run with --release -- --ignored"]
    fn he4_b2_matches_cencek() {
        use potter_poc::quantum::quantum_b2;
        use potter_poc::quantum::Species;
        let refs: [(f64, f64, f64); 5] =
            [(4.0, -85.061, 0.06), (10.0, -23.125, 0.05), (20.0, -2.7464, 0.03),
             (100.0, 11.6747, 0.02), (500.0, 11.00715, 0.02)];
        for &(t, b, u) in &refs {
            let got = quantum_b2(Species::He4, t);
            assert!((got - b).abs() < u.max(0.1), "4He B2 T={t}: {got} vs {b} (±{u})");
        }
    }

    #[test]
    #[ignore = "heavy phase-shift integration (~15 s release); run with --release -- --ignored"]
    fn quantum_b2_high_t_to_classical() {
        use potter_poc::quantum::{quantum_b2, classical_b2, Species};
        for &t in &[2000.0_f64, 5000.0] {
            let q = quantum_b2(Species::He4, t);
            let c = classical_b2(Species::He4, t);
            assert!((q - c).abs() / c.abs() < 0.05, "T={t}: quantum {q} vs classical {c}");
        }
    }

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

    #[test]
    #[ignore = "heavy phase-shift integration (~3 min release); run with --release -- --ignored"]
    fn he4_neff_matches_cencek_and_fig8() {
        use potter_poc::quantum::{quantum_b2_neff, Species};
        // T*dB2/dT and T^2*d2B2/dT2 vs the Cencek 2012 tabulated TB', T^2B''.
        let refs = [(10.0, 41.022, -82.478), (100.0, 2.0908, -6.9989), (500.0, -1.87546, 0.98256)];
        for &(t, tbp, t2bpp) in &refs {
            let (_b, db, d2b, _ne) = quantum_b2_neff(Species::He4, t);
            assert!((t * db - tbp).abs() < 0.5 + 0.05 * tbp.abs(), "TB' T={t}: {} vs {tbp}", t * db);
            assert!((t * t * d2b - t2bpp).abs() < 1.0 + 0.05 * t2bpp.abs(), "T2B'' T={t}: {} vs {t2bpp}", t * t * d2b);
        }
        // Fig. 8: the 4He n_eff peaks at ~140 near 8-10 K.
        let peak = [6.0, 8.0, 10.0, 12.0, 15.0].iter()
            .map(|&t| quantum_b2_neff(Species::He4, t).3).fold(0.0_f64, f64::max);
        assert!(peak > 100.0, "4He n_eff peak {peak} (expect ~140)");
    }
}
