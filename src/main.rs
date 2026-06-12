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
}
