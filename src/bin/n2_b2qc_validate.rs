//! Validate the first-order Wigner-Kirkwood quantum correction to N2 B2 against
//! column 3 (B2 classical + quantum) of Hellmann's MP-2012 supplement.

use potter_poc::molecule::n2_hellmann;

const DATA: &str = include_str!("n2_b2qc_hellmann.csv"); // "T,B2_cl+qc"

fn main() {
    let n2 = n2_hellmann();
    let mu = 14.0067; // amu = M(N2)/2, the two-body reduced mass of the N2-N2 pair
    let inertia = 8.473; // amu*A^2, N2 moment of inertia (B0 = 1.9896 cm^-1)

    println!("Hellmann N2 B2 + first-order quantum correction vs SI column 3 [cm^3/mol]");
    println!("    T[K]   B2 classical   B2 cl+qc (this)   SI cl+qc    diff");
    let mut maxd = 0.0f64;
    for (idx, line) in DATA.lines().enumerate() {
        if idx % 4 != 0 {
            continue; // stride: the quantum integrand is ~9x the classical cost
        }
        let v: Vec<f64> = line.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        if v.len() != 2 {
            continue;
        }
        let (t, ref_qc) = (v[0], v[1]);
        let (cl, _) = n2.b2(t, 1e-4);
        let (qc, _) = n2.b2_quantum(t, 1e-4, mu, inertia);
        let d = qc - ref_qc;
        maxd = maxd.max(d.abs());
        println!("   {t:>6.0}   {cl:>11.2}   {qc:>13.2}   {ref_qc:>9.2}   {d:+7.3}");
    }
    println!("\n  max |diff| vs SI (cl+qc) = {maxd:.3} cm^3/mol");
}
