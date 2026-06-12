//! Validate B2 with the quadratic Feynman-Hibbs (QFH) quantum correction against
//! Hellmann's *tabulated* CO2 SI values (CPL 2014, V_B: column 4 = B_cl, column 5
//! = B_QFH) -- the ab initio computed points, not a correlation fit.

use potter_poc::molecule::{co2_hellmann, n2_hellmann};

const CO2: &str = include_str!("co2_b2_hellmann.csv"); // "T,B_cl,B_QFH" (V_B)

fn main() {
    let co2 = co2_hellmann();
    let (mu, inertia) = (22.0045, 43.202); // M(CO2)/2 (pair reduced mass), CO2 moment of inertia

    println!("CO2 vs Hellmann SI tabulated (V_B) [cm^3/mol]:");
    println!("    T[K]   B_cl(this) B_cl(SI)   B_QFH(this) B_QFH(SI)   qfh diff");
    let mut maxd = 0.0f64;
    for (idx, line) in CO2.lines().enumerate() {
        if idx % 6 != 0 {
            continue; // stride: QFH is ~9x the classical cost
        }
        let v: Vec<f64> = line.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        if v.len() != 3 {
            continue;
        }
        let (t, b_cl, b_qfh) = (v[0], v[1], v[2]);
        let (cl, _) = co2.b2(t, 1e-4);
        let (qfh, _) = co2.b2_qfh(t, 1e-4, mu, inertia);
        maxd = maxd.max((qfh - b_qfh).abs());
        println!("   {t:>6.0}   {cl:>9.2} {b_cl:>9.2}   {qfh:>9.2} {b_qfh:>9.2}   {:+7.3}", qfh - b_qfh);
    }
    println!("\n  max |QFH diff| vs SI = {maxd:.3} cm^3/mol");

    let n2 = n2_hellmann();
    println!("\nN2: WK vs QFH [cm^3/mol] (N2 SI is WK; CO2 used QFH):");
    println!("    T[K]    WK        QFH       QFH-WK");
    for &t in &[50.0, 90.0, 150.0, 300.0] {
        let (wk, _) = n2.b2_quantum(t, 1e-4, 14.0067, 8.473);
        let (qfh, _) = n2.b2_qfh(t, 1e-4, 14.0067, 8.473);
        println!("   {t:>6.0}   {wk:>8.3}   {qfh:>8.3}   {:+7.4}", qfh - wk);
    }
}
