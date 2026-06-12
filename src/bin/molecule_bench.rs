//! Performance of the linear-molecule B2 (4-D Genz-Malik cubature, Hellmann PES).

use potter_poc::molecule::{co2_hellmann, n2_hellmann, RigidLinear};
use std::time::Instant;

fn time_b2(mol: &RigidLinear, t: f64, reltol: f64) -> (f64, usize, f64) {
    // best of 3
    let mut best = f64::INFINITY;
    let mut out = (0.0, 0);
    for _ in 0..3 {
        let t0 = Instant::now();
        out = mol.b2(t, reltol);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        best = best.min(ms);
    }
    (out.0, out.1, best)
}

fn sweep_ms(mol: &RigidLinear, temps: &[f64], reltol: f64, nthreads: usize) -> f64 {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let next = AtomicUsize::new(0);
    let t0 = Instant::now();
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..nthreads)
            .map(|_| {
                let next = &next;
                s.spawn(move || loop {
                    // dynamic work-stealing: pull the next temperature index, so
                    // the expensive low-T points spread across threads
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= temps.len() {
                        break;
                    }
                    let _ = mol.b2(temps[i], reltol);
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    });
    t0.elapsed().as_secs_f64() * 1e3
}

fn main() {
    let n2 = n2_hellmann();
    let co2 = co2_hellmann();

    println!("Single B2 evaluation (best of 3):");
    println!("  molecule   T[K]   reltol    B2[cm^3/mol]    evals      time      throughput");
    for (name, mol, temps) in [("N2 ", &n2, [100.0, 300.0, 1000.0]), ("CO2", &co2, [250.0, 400.0, 700.0])] {
        for &t in &temps {
            for &rel in &[1e-3, 1e-4] {
                let (b2, nev, ms) = time_b2(mol, t, rel);
                let mev = nev as f64 / (ms / 1e3) / 1e6;
                println!("  {name}      {t:>5.0}   {rel:>6.0e}   {b2:>11.3}   {nev:>9}   {ms:>7.1} ms   {mev:>5.1} M eval/s");
            }
        }
    }

    // Temperature sweep: B2 calculations over a grid are embarrassingly parallel
    // (independent), just like potter threads over temperatures.
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let temps: Vec<f64> = (0..24).map(|i| 100.0 + i as f64 * 50.0).collect(); // 100..1250 K
    println!("\nB2 over a {}-temperature grid (N2, reltol 1e-4), {cores} cores:", temps.len());
    let serial = sweep_ms(&n2, &temps, 1e-4, 1);
    println!("  serial (1 thread) : {serial:7.0} ms");
    for &k in &[2usize, 4, 6, 8] {
        let ms = sweep_ms(&n2, &temps, 1e-4, k);
        println!("  {k:>2} threads        : {ms:7.0} ms   ({:.1}x)", serial / ms);
    }
}
