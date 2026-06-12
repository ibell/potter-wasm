//! Mayer-sampling Monte Carlo for B3, cross-checked against the deterministic
//! Genz-Malik cubature. Shows the MC paradigm reproduces B3 within statistical
//! error — the sampler is dimension-insensitive, so it is what scales to B4+.

use potter_poc::b3_cubature_v;
use potter_poc::msmc::msmc_b3_v;
use std::time::Instant;

fn lj(r: f64) -> f64 {
    let inv = 1.0 / r;
    let s2 = inv * inv;
    let s6 = s2 * s2 * s2;
    4.0 * (s6 * s6 - s6)
}

fn main() {
    let nsteps = 20_000_000usize;
    println!("B3 of Lennard-Jones (12-6): Mayer-sampling MC vs deterministic cubature");
    println!("(MC: {nsteps} steps, hard-sphere reference)\n");

    for &t in &[1.0_f64, 1.5, 2.0, 5.0] {
        let (cub, _n) = b3_cubature_v(&lj, t, 1e-8);
        // a couple of reference diameters to show the estimate is reference-independent
        print!("T* = {t:.1}   cubature B3 = {cub:.5}");
        let t0 = Instant::now();
        let mut line = String::new();
        for &shs in &[1.3_f64, 1.5, 2.0] {
            let r = msmc_b3_v(&lj, t, shs, nsteps, 0x1234_5678 ^ (t.to_bits()));
            let dev = (r.b3 - cub) / r.stderr;
            line.push_str(&format!(
                "\n     MSMC(sig_HS={shs}) = {:.5} +/- {:.5}   ({:+.1} sigma from cubature, acc={:.0}%)",
                r.b3, r.stderr, dev, r.accept * 100.0
            ));
        }
        println!("{line}");
        println!("     [{:.0} ms for 3 MC runs]\n", t0.elapsed().as_secs_f64() * 1e3);
    }
}
