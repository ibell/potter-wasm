//! B2(T) for rigid linear molecules, validated against experiment.
//! Models: TraPPE nitrogen (2CLJ + quadrupole) and EPM2 carbon dioxide.
//! Experimental B2 (cm^3/mol) are approximate literature values (Dymond & Smith
//! compilation) for a sanity check — verify against the source before quoting.

use potter_poc::molecule::{co2_epm2, n2_trappe, two_center_lj};

fn main() {
    let tol = 1e-4;

    println!("Two-centre LJ (eps/k=100 K, sig=3.5 A) — effect of elongation on B2(300 K):");
    for &bond in &[0.0_f64, 0.3, 0.6, 1.0] {
        let (b2, _) = two_center_lj(100.0, 3.5, bond).b2(300.0, tol);
        println!("   bond L = {bond:.1} A :  B2 = {b2:8.2} cm^3/mol");
    }

    println!("\nNitrogen (TraPPE 2CLJ+Q):   T[K]    B2 model    B2 exp(approx)   diff");
    let n2 = n2_trappe();
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
        println!("   {t:>6.1}   {b2:9.1}   {exp:11.1}      {:+6.1}", b2 - exp);
    }

    println!("\nCarbon dioxide (EPM2):      T[K]    B2 model    B2 exp(approx)   diff");
    let co2 = co2_epm2();
    for &(t, exp) in &[
        (250.0, -197.0_f64),
        (273.15, -150.0),
        (300.0, -124.0),
        (350.0, -86.0),
        (400.0, -60.5),
        (500.0, -34.5),
    ] {
        let (b2, _) = co2.b2(t, tol);
        println!("   {t:>6.1}   {b2:9.1}   {exp:11.1}      {:+6.1}", b2 - exp);
    }
}
