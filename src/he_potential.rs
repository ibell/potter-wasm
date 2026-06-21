//! Cencek et al. 2012 ab initio He–He pair potential (clean-room translation of
//! the SI `potentials.f90`). Pure analytic: input r in Bohr (a₀), output Hartree.
//!
//! Components: Born–Oppenheimer (`bo`), adiabatic correction (`ad`), relativistic
//! (`v_rel = cg + d2 + br`) and QED (`v_qed = a3d1 + a3d2 + as_`) corrections, plus
//! the directly-fitted total (`total`). Tang–Toennies damping (`damp`/`damp_mod`)
//! and Casimir–Polder retardation (`damp_ret`) match the Fortran branch-for-branch.

use std::f64::consts::PI;

const FSALPHA: f64 = 1.0 / 137.035_999_679;

/// Tang–Toennies incomplete-gamma damping factor of order `n`:
/// `1 - exp(-br) Σ_{i=0..n} (br)^i/i!`, evaluated stably about `br = 1`.
fn damp(n: usize, eta: f64, r: f64) -> f64 {
    let br = eta * r;
    if br > 1.0 {
        // 1 - exp(-x) sum(x^i/i!, i=0..n)
        let mut term = 1.0;
        let mut suma = 1.0;
        for i in 1..=n {
            term *= br / i as f64;
            suma += term;
        }
        1.0 - (-br).exp() * suma
    } else {
        // exp(-x) sum(x^i/i!, i=n+1..inf)
        let mut term = 1.0;
        for i in 1..=n {
            term *= br / i as f64;
        }
        let mut suma = 0.0;
        for i in (n + 1)..=(n + 40) {
            term *= br / i as f64;
            suma += term;
        }
        suma * (-br).exp()
    }
}

/// Retardation-modified damping: `- exp(-br) Σ_{i=0..n} (br)^i/i!` (the piece that
/// pairs with `damp_ret` in the retarded C₆ term).
fn damp_mod(n: usize, eta: f64, r: f64) -> f64 {
    let br = eta * r;
    let mut term = 1.0;
    let mut suma = 1.0;
    for i in 1..=n {
        term *= br / i as f64;
        suma += term;
    }
    -(-br).exp() * suma
}

// ---- module retardation -----------------------------------------------------

const RET_POLARIZABILITY: f64 = 1.383_192_174_40;
const RET_C6: f64 = 1.460_977_837_725;
// W4 = 0.35322e-4 / fsalpha^2 ; AS3 = 0.577235e-6 / fsalpha^3
// K7 = 23/(4 pi) polarizability^2 / fsalpha ; ratio = fsalpha*K7/C6
const RET_B1: f64 = 8.454_943_177_941_253;
const RET_B2: f64 = 16.006_586_066_260_556;
// trailing digits mirror the Fortran literal; f64 rounds them identically
#[allow(clippy::excessive_precision)]
const RET_B3: f64 = 10.378_373_954_734_820;
const RET_B4: f64 = 3.515_803_817_223_855;
const RET_B5: f64 = 0.591_502_377_533_792;
const RET_B6: f64 = 0.059_455_768_329_599;

/// Casimir–Polder retardation correction factor for the C₆ term.
fn damp_ret(r: f64) -> f64 {
    let w4 = 0.35322e-4 / (FSALPHA * FSALPHA);
    let as3 = 0.577235e-6 / FSALPHA.powi(3);
    let k7 = 23.0 / (4.0 * PI) * RET_POLARIZABILITY * RET_POLARIZABILITY / FSALPHA;
    let ratio = FSALPHA * k7 / RET_C6;

    let a1 = RET_B1;
    let a2 = RET_B2 - w4 / RET_C6;
    let a3 = RET_B3 - RET_B1 * w4 / RET_C6 + as3 / RET_C6;
    let a4 = ratio * RET_B5;
    let a5 = ratio * RET_B6;

    let x = r * FSALPHA;
    let tmp1 = ((((a5 * x + a4) * x + a3) * x + a2) * x + a1) * x + 1.0;
    let tmp2 =
        (((((RET_B6 * x + RET_B5) * x + RET_B4) * x + RET_B3) * x + RET_B2) * x + RET_B1) * x + 1.0;
    tmp1 / tmp2
}

// ---- module Total_Fit -------------------------------------------------------

const TOT_A: f64 = 3.648_903_036_528_30;
const TOT_B: f64 = 2.368_248_717_435_91;
const TOT_ETA: f64 = 4.094_238_051_178_71;
const TOT_P0: f64 = -25.470_166_941_662_1;
const TOT_P1: f64 = 269.244_425_630_616;
const TOT_P2: f64 = -56.387_997_040_207_9;
const TOT_Q0: f64 = 38.795_748_731_007_1;
const TOT_Q1: f64 = -2.765_771_367_727_54;
const TOT_C6BO: f64 = 1.460_977_837_725;
// Cn(3:16)
const TOT_CN: [f64; 14] = [
    0.577235e-06,  // 3
    -0.35322e-04,  // 4
    0.1377841e-05, // 5
    1.46183,       // 6
    0.0,           // 7
    14.1235,       // 8
    0.0,           // 9
    183.7497,      // 10
    -76.74,        // 11
    3372.0,        // 12
    -3806.0,       // 13
    85340.0,       // 14
    -170700.0,     // 15
    2860000.0,     // 16
];

#[inline]
fn cn(n: usize) -> f64 {
    TOT_CN[n - 3]
}

/// Directly-fitted total interaction energy (Hartree). `ret6` toggles retardation.
pub fn total(ret6: bool, r: f64) -> f64 {
    let mut term = (TOT_P0 + TOT_P1 * r + TOT_P2 * r * r) * (-TOT_A * r).exp();
    term += (TOT_Q0 + TOT_Q1 * r) * (-TOT_B * r).exp();
    let mut asy;
    if ret6 {
        asy = -damp_mod(3, TOT_ETA, r) * cn(3) / r.powi(3)
            - damp_mod(4, TOT_ETA, r) * cn(4) / r.powi(4)
            - (damp_ret(r) + damp_mod(6, TOT_ETA, r)) * TOT_C6BO / r.powi(6)
            - damp(6, TOT_ETA, r) * (cn(6) - TOT_C6BO) / r.powi(6);
    } else {
        asy = -damp(3, TOT_ETA, r) * cn(3) / r.powi(3)
            - damp(4, TOT_ETA, r) * cn(4) / r.powi(4)
            - damp(6, TOT_ETA, r) * cn(6) / r.powi(6);
    }
    asy = asy
        - damp(5, TOT_ETA, r) * cn(5) / r.powi(5)
        - damp(8, TOT_ETA, r) * cn(8) / r.powi(8);
    for n in 10..=16 {
        asy -= damp(n, TOT_ETA, r) * cn(n) / r.powi(n as i32);
    }
    term + asy
}

// ---- module Total_Fit_sigma -------------------------------------------------

const SIG_C: [f64; 3] = [0.16702e-3, 0.4524e-5, 0.1843e-7];
const SIG_A: [f64; 3] = [2.456, 1.100, 0.4381];

/// Standard-deviation envelope of the fit (Hartree).
pub fn sigma_total(r: f64) -> f64 {
    let mut term = 0.0;
    for i in 0..3 {
        term += SIG_C[i] * (-SIG_A[i] * r).exp();
    }
    term
}

// ---- module Born_Oppenheimer ------------------------------------------------

const BO_ALPHA: f64 = 3.652_719_493_561_13;
const BO_BETA: f64 = 2.367_208_714_712_73;
const BO_ETA: f64 = 4.097_079_826_512_18;
const BO_P0: f64 = -25.261_631_571_163_8;
const BO_P1: f64 = 272.551_324_570_603;
const BO_P2: f64 = -56.550_658_783_728_7;
const BO_Q0: f64 = 38.633_962_767_547_6;
const BO_Q1: f64 = -2.765_379_300_162_71;
// Cn(6:16)
const BO_CN: [f64; 11] = [
    1.460_977_837_725, // 6
    0.0,               // 7
    14.117_857_37,     // 8
    0.0,               // 9
    183.691_075,       // 10
    -76.74,            // 11
    3372.0,            // 12
    -3806.0,           // 13
    85340.0,           // 14
    -170700.0,         // 15
    2860000.0,         // 16
];

#[inline]
fn bo_cn(n: usize) -> f64 {
    BO_CN[n - 6]
}

/// Born–Oppenheimer interaction energy (Hartree).
pub fn bo(ret6: bool, r: f64) -> f64 {
    let mut term = (BO_P0 + BO_P1 * r + BO_P2 * r * r) * (-BO_ALPHA * r).exp();
    term += (BO_Q0 + BO_Q1 * r) * (-BO_BETA * r).exp();
    let mut asy = if ret6 {
        -(damp_ret(r) + damp_mod(6, BO_ETA, r)) * bo_cn(6) / r.powi(6)
    } else {
        -damp(6, BO_ETA, r) * bo_cn(6) / r.powi(6)
    };
    asy -= damp(8, BO_ETA, r) * bo_cn(8) / r.powi(8);
    for n in 10..=16 {
        asy -= damp(n, BO_ETA, r) * bo_cn(n) / r.powi(n as i32);
    }
    term + asy
}

// ---- module adiabatic_correction --------------------------------------------

const AD_A: f64 = 1.933_034_961_282_77;
const AD_B: f64 = 3.664_944_616_161_41;
const AD_ETA: f64 = 1.436_035_566_839_87;
const AD_P0: f64 = 0.377_267_595_818_441e-02;
const AD_P1: f64 = -0.668_389_341_060_405e-03;
const AD_Q0: f64 = 0.241_186_947_243_731e-01;
const AD_Q1: f64 = -0.219_794_150_747_149e-01;
const AD_A6: f64 = 0.0011445;
const AD_A8: f64 = 0.006519;
const AD_A10: f64 = 0.0668;

/// Adiabatic (diagonal Born–Oppenheimer) correction (Hartree).
pub fn ad(r: f64) -> f64 {
    let mut term = (AD_P0 + AD_P1 * r) * (-AD_A * r).exp();
    term += (AD_Q0 + AD_Q1 * r) * (-AD_B * r).exp();
    let mut asy = -damp(6, AD_ETA, r) * AD_A6 / r.powi(6);
    asy -= damp(8, AD_ETA, r) * AD_A8 / r.powi(8);
    asy -= damp(10, AD_ETA, r) * AD_A10 / r.powi(10);
    term + asy
}

// ---- module Cowan_Griffin ---------------------------------------------------

const CG_A: f64 = 2.900_060_318_787_70;
const CG_B: f64 = 2.242_569_681_917_19;
const CG_C: f64 = 0.425_026_783_288_716e-01;
const CG_ETA: f64 = 2.951_100_608_357_81;
const CG_P0: f64 = -0.225_884_444_742_344e-02;
const CG_P1: f64 = -0.340_134_134_669_477e-02;
const CG_Q0: f64 = 0.369_060_490_502_093e-02;
const CG_Q1: f64 = -0.163_513_409_459_839e-02;
const CG_A6: f64 = -0.000257;
const CG_A8: f64 = -0.00286;
const CG_A10: f64 = -0.0401;

/// Cowan–Griffin relativistic correction (Hartree).
pub fn cg(r: f64) -> f64 {
    let mut term = (CG_P0 + CG_P1 * r) * (-CG_A * r).exp();
    term += (CG_Q0 + CG_Q1 * r) * (-CG_B * r - CG_C * r * r).exp();
    let mut asy = -damp(6, CG_ETA, r) * CG_A6 / r.powi(6);
    asy -= damp(8, CG_ETA, r) * CG_A8 / r.powi(8);
    asy -= damp(10, CG_ETA, r) * CG_A10 / r.powi(10);
    term + asy
}

// ---- module Darwin_1el ------------------------------------------------------

const D1_A: f64 = 2.010_426_478_635_43;
const D1_B: f64 = 2.113_137_905_183_24;
const D1_C: f64 = 0.186_163_623_408_221;
const D1_ETA: f64 = 4.074_212_747_887_35;
const D1_P0: f64 = 0.120_025_637_135_018e-01;
const D1_P1: f64 = -0.136_621_851_503_271e-02;
const D1_Q0: f64 = 0.601_999_646_519_627e-02;
const D1_Q1: f64 = -0.445_269_531_689_116e-02;
const D1_A6: f64 = 0.001426;
const D1_A8: f64 = 0.01753;
const D1_A10: f64 = 0.2735;
const D1_LN_K0: f64 = 4.370_160_222_0;

fn d1(r: f64) -> f64 {
    let mut term = (D1_P0 + D1_P1 * r) * (-D1_A * r).exp();
    term += (D1_Q0 + D1_Q1 * r) * (-D1_B * r - D1_C * r * r).exp();
    let mut asy = -damp(6, D1_ETA, r) * D1_A6 / r.powi(6);
    asy -= damp(8, D1_ETA, r) * D1_A8 / r.powi(8);
    asy -= damp(10, D1_ETA, r) * D1_A10 / r.powi(10);
    term + asy
}

/// One-electron Darwin QED term α³·D1 (Hartree).
pub fn a3d1(r: f64) -> f64 {
    let a2_a3 = FSALPHA * (4.0 / 3.0) * (19.0 / 30.0 - 2.0 * FSALPHA.ln() - D1_LN_K0) / (PI / 2.0);
    a2_a3 * d1(r)
}

// ---- module Darwin_2el ------------------------------------------------------

const D2_A: f64 = 1.852_950_731_266_38;
const D2_B: f64 = 2.527_293_372_964_64;
const D2_C: f64 = 0.351_932_113_973_596;
const D2_ETA: f64 = 4.443_148_231_196_77;
const D2_P0: f64 = 0.544_670_094_065_963e-03;
const D2_P1: f64 = -0.699_315_570_914_574e-04;
const D2_Q0: f64 = 0.248_652_720_153_703e-02;
const D2_Q1: f64 = -0.283_371_150_273_082e-04;
const D2_A6: f64 = 0.000103;
const D2_A8: f64 = 0.00136;
const D2_A10: f64 = 0.0222;

/// Raw two-electron Darwin fit `D2` (Hartree) — the form used by V_rel.
pub fn d2(r: f64) -> f64 {
    let mut term = (D2_P0 + D2_P1 * r) * (-D2_A * r).exp();
    term += (D2_Q0 + D2_Q1 * r) * (-D2_B * r - D2_C * r * r).exp();
    let mut asy = -damp(6, D2_ETA, r) * D2_A6 / r.powi(6);
    asy -= damp(8, D2_ETA, r) * D2_A8 / r.powi(8);
    asy -= damp(10, D2_ETA, r) * D2_A10 / r.powi(10);
    term + asy
}

/// Two-electron Darwin QED term α³·D2 (Hartree) — the form used by V_QED.
pub fn a3d2(r: f64) -> f64 {
    let a2_a3 = FSALPHA * (164.0 / 15.0 + 14.0 / 3.0 * FSALPHA.ln()) / PI;
    a2_a3 * d2(r)
}

// ---- module Breit -----------------------------------------------------------

const BR_A: f64 = 1.745_150_468_393_22;
const BR_B: f64 = 5.000_000_000_000_00;
const BR_C: f64 = 0.263_295_834_449_133;
const BR_ETA: f64 = 3.456_865_781_551_64;
const BR_P0: f64 = 0.203_310_117_594_328e-03;
const BR_P1: f64 = -0.705_614_471_514_519e-04;
const BR_P2: f64 = 0.721_004_448_825_699e-05;
const BR_Q0: f64 = -0.270_357_188_415_393e-02;
const BR_Q1: f64 = 0.516_061_855_783_347e-03;
const BR_A4: f64 = -0.000035322;
const BR_A6: f64 = -0.0001894;

/// Breit interaction correction (Hartree).
pub fn br(ret6: bool, r: f64) -> f64 {
    let mut term = (BR_P0 + BR_P1 * r + BR_P2 * r * r) * (-BR_A * r).exp();
    term += (BR_Q0 + BR_Q1 * r) * (-BR_B * r - BR_C * r * r).exp();
    let mut asy = if ret6 {
        -damp_mod(4, BR_ETA, r) * BR_A4 / r.powi(4)
    } else {
        -damp(4, BR_ETA, r) * BR_A4 / r.powi(4)
    };
    asy -= damp(6, BR_ETA, r) * BR_A6 / r.powi(6);
    term + asy
}

// ---- module Araki_Sucher ----------------------------------------------------

const AS_A: f64 = 0.262_672_224_977_970;
const AS_B: f64 = 2.027_144_040_181_30;
const AS_ETA: f64 = 2.887_355_580_424_63;
const AS_P0: f64 = 0.897_489_244_934_804e-09;
const AS_P1: f64 = -0.186_471_329_166_686e-09;
const AS_Q0: f64 = 0.186_184_585_055_017e-05;
const AS_Q1: f64 = -0.152_618_822_249_290e-05;
const AS_A3: f64 = 0.577235e-06;
const AS_A5: f64 = 0.1377841e-05;

/// Araki–Sucher QED term (Hartree).
pub fn as_(ret6: bool, r: f64) -> f64 {
    let mut term = (AS_P0 + AS_P1 * r) * (-AS_A * r).exp();
    term += (AS_Q0 + AS_Q1 * r) * (-AS_B * r).exp();
    let mut asy = if ret6 {
        -damp_mod(3, AS_ETA, r) * AS_A3 / r.powi(3)
    } else {
        -damp(3, AS_ETA, r) * AS_A3 / r.powi(3)
    };
    asy -= damp(5, AS_ETA, r) * AS_A5 / r.powi(5);
    term + asy
}

// ---- module potential_interface ---------------------------------------------

/// Helium isotope (selects the adiabatic-correction multiplier and reduced mass).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum He {
    He4,
    He3,
}

const MNUC4: f64 = 7294.2995365;
const MNUC3: f64 = 5495.8852765;

/// Two-body reduced mass in electron masses mₑ for a homonuclear pair.
pub fn reduced_mass_me(iso: He) -> f64 {
    match iso {
        He::He4 => MNUC4 / 2.0,
        He::He3 => MNUC3 / 2.0,
    }
}

/// Adiabatic-correction mass scaling: mult44 = 1, mult33 = MU44/MU33 (Fortran
/// `potential_interface`). The adiabatic term scales inversely with nuclear mass.
fn mult(iso: He) -> f64 {
    match iso {
        He::He4 => 1.0,
        He::He3 => (MNUC4 / 2.0) / (MNUC3 / 2.0),
    }
}

/// `(V_BO, V_ad, V_rel, V_QED, V_tot)` in Hartree at separation `r` [Bohr].
///
/// Per the Fortran `potential_interface`:
///   V_rel = CG(r) + D2(r)   + Br(ret6,r)
///   V_QED = a3D1(r) + a3D2(r) + AS(ret6,r)
/// Note V_rel uses the raw `d2`, while V_QED uses the α³-scaled `a3d2`.
pub fn v_components(r: f64, ret6: bool) -> (f64, f64, f64, f64, f64) {
    let v_rel = cg(r) + d2(r) + br(ret6, r);
    let v_qed = a3d1(r) + a3d2(r) + as_(ret6, r);
    (bo(ret6, r), ad(r), v_rel, v_qed, total(ret6, r))
}

/// Isotope pair potential V(r) [Hartree], r [Bohr]:
/// `V_BO + mult·V_ad + V_rel + V_QED`.
pub fn v_he(iso: He, r: f64, ret6: bool) -> f64 {
    let (bo, ad_, rel, qed, _tot) = v_components(r, ret6);
    bo + mult(iso) * ad_ + rel + qed
}
