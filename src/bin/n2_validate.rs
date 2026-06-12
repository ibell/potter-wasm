//! Validate the coded Hellmann N2 potential against ALL 408 ab initio reference
//! energies from the potter repo (validdata_nitrogen_potential): each row is
//! r[A], theta1[deg], theta2[deg], phi[deg], V12/kB[K].

use potter_poc::molecule::n2_hellmann;

const DATA: &str = include_str!("n2_validdata.csv");

fn main() {
    let n2 = n2_hellmann();
    let d = std::f64::consts::PI / 180.0;
    let mut n = 0;
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let (mut worst_r, mut worst_ref, mut worst_got) = (0.0, 0.0, 0.0);

    for line in DATA.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: Vec<f64> = line.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        if v.len() != 5 {
            continue;
        }
        let (r, t1, t2, phi, vref) = (v[0], v[1] * d, v[2] * d, v[3] * d, v[4]);
        let got = n2.energy(r, t1, t2, phi);
        let abs = (got - vref).abs();
        let rel = abs / vref.abs().max(1.0);
        if abs > max_abs {
            max_abs = abs;
            worst_r = v[0];
            worst_ref = vref;
            worst_got = got;
        }
        max_rel = max_rel.max(rel);
        n += 1;
    }

    println!("Hellmann N2 potential vs {n} ab initio reference energies (potter validdata):");
    println!("  max |diff|     = {max_abs:.4} K");
    println!("  max rel. diff  = {max_rel:.2e}");
    println!("  worst point: r={worst_r} A  ref={worst_ref:.3} K  computed={worst_got:.3} K");
    // reference energies are printed to 3 decimals, so ~5e-4 K is exact agreement
    if max_abs < 1e-3 {
        println!("  => PASS: reproduces all {n} reference energies to rounding.");
    } else {
        println!("  => MISMATCH");
    }
}
