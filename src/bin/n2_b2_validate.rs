//! Validate the computed Hellmann N2 second virial coefficient against the
//! B2(classical) table from Hellmann's Molecular Physics 2012 supplement (the SI).
//! Column 2 of that table is B2(classical) in cm^3/mol, which is exactly what the
//! classical 4-D cubature here computes.

use potter_poc::molecule::n2_hellmann;

const DATA: &str = include_str!("n2_b2_hellmann.csv"); // "T,B2_classical" rows

fn main() {
    let n2 = n2_hellmann();
    println!("Hellmann N2 B2(classical): cubature here vs SI supplement [cm^3/mol]");
    println!("    T[K]     B2 (this)    B2 (Hellmann SI)     diff");
    let mut max_rel = 0.0f64;
    let mut n = 0;
    for (idx, line) in DATA.lines().enumerate() {
        let v: Vec<f64> = line.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        if v.len() != 2 {
            continue;
        }
        // stride to keep runtime modest; the low-T points are the sharp/slow ones
        if idx % 3 != 0 {
            continue;
        }
        let (t, b2ref) = (v[0], v[1]);
        let (b2, _) = n2.b2(t, 1e-4);
        let rel = (b2 - b2ref).abs() / b2ref.abs().max(1.0);
        max_rel = max_rel.max(rel);
        n += 1;
        println!("   {t:>6.0}   {b2:11.2}   {b2ref:13.2}     {:+8.3}", b2 - b2ref);
    }
    println!("\n  {n} points checked, max relative difference = {max_rel:.1e}");
}
