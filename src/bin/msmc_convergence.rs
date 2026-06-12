//! MSMC convergence check: at each temperature, run Mayer-sampling MC for B3 over
//! a geometric ladder of step counts and show (a) the estimate converging to the
//! deterministic cubature value and (b) the statistical error shrinking as 1/sqrt(N)
//! — i.e. err*sqrt(N) stays roughly constant (the Monte Carlo error constant).

use potter_poc::b3_cubature_v;
use potter_poc::msmc::msmc_b3_v;

fn lj(r: f64) -> f64 {
    let inv = 1.0 / r;
    let s2 = inv * inv;
    let s6 = s2 * s2 * s2;
    4.0 * (s6 * s6 - s6)
}

fn main() {
    let sigma_hs = 1.3;
    let ladder: [usize; 5] = [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000];
    println!("MSMC B3 convergence (LJ 12-6, hard-sphere reference sig_HS={sigma_hs})");
    println!("expect: B3 -> cubature, dev within a few sigma, err*sqrt(N) ~ constant\n");

    for &t in &[1.0_f64, 1.5, 2.0, 5.0] {
        let (cub, _) = b3_cubature_v(&lj, t, 1e-9);
        println!("T* = {t:.1}   cubature B3 = {cub:.6}");
        println!("        N          B3          stderr     |B3-cub|    dev      err*sqrt(N)");
        for &n in &ladder {
            let seed = 0xA5A5_0000u64 ^ t.to_bits() ^ (n as u64).wrapping_mul(0x9E3779B1);
            let r = msmc_b3_v(&lj, t, sigma_hs, n, seed);
            let dev = (r.b3 - cub) / r.stderr;
            println!(
                "  {n:>11}   {:>9.5}   {:>9.5}   {:>8.1e}   {:>+5.1}s   {:>8.2}",
                r.b3,
                r.stderr,
                (r.b3 - cub).abs(),
                dev,
                r.stderr * (n as f64).sqrt()
            );
        }
        println!();
    }
}
