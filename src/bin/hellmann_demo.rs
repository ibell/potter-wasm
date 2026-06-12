//! High-accuracy ab initio Hellmann potentials for N2 (2013) and CO2 (2014):
//! B2(T) from the site-site PES via 4-D cubature, validated against experiment
//! and (for CO2) against Hellmann's own published B2 correlation.

use potter_poc::molecule::{co2_hellmann, n2_hellmann};

/// Hellmann CO2 classical B2 correlation (cm^3/mol), from the potter repo
/// (Hellmann_CO2_virial.py), as a reference for the PES integration.
fn co2_b2_hellmann_corr(t: f64) -> f64 {
    let powers = [
        1.0, 0.0, -0.5, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0,
    ];
    let c = [
        -0.257176088168, 31.7003916790, 64.5362279298, -119.816302333, 5.86025683115,
        -15.4996935556, 4.83954561747, -1.95578082709, 0.338813992864, -0.0395510810339,
        0.0, 0.0, 0.0, 0.0,
    ];
    let ts = t / 500.0;
    c.iter().zip(powers).map(|(ci, p)| ci * ts.powf(p)).sum()
}

fn main() {
    let tol = 1e-4;

    println!("Hellmann N2 (2013) ab initio PES — B2(T) [cm^3/mol]");
    println!("    T[K]    B2 model    B2 exp(approx)    diff");
    let n2 = n2_hellmann();
    for &(t, exp) in &[
        (100.0, -160.0_f64),
        (150.0, -71.5),
        (200.0, -35.2),
        (250.0, -16.2),
        (300.0, -4.5),
        (400.0, 9.0),
        (500.0, 16.9),
    ] {
        let (b2, _) = n2.b2(t, tol);
        println!("   {t:>6.0}   {b2:9.2}   {exp:11.1}     {:+6.2}", b2 - exp);
    }

    println!("\nHellmann CO2 (2014) ab initio PES — B2(T) [cm^3/mol]");
    println!("    T[K]    B2 model    Hellmann corr    exp(approx)");
    let co2 = co2_hellmann();
    for &(t, exp) in &[
        (250.0, -197.0_f64),
        (300.0, -123.0),
        (350.0, -86.0),
        (400.0, -60.5),
        (500.0, -34.5),
        (700.0, -9.0),
    ] {
        let (b2, _) = co2.b2(t, tol);
        println!("   {t:>6.0}   {b2:9.2}   {:11.2}   {exp:11.1}", co2_b2_hellmann_corr(t));
    }
}
