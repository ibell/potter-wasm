//! Two MSMC improvements: (A) overlap sampling (a better reference treatment) for
//! variance reduction, and (B) thread parallelism. B3 of LJ (12-6).

use potter_poc::b3_cubature_v;
use potter_poc::msmc::{msmc_b3_overlap_parallel, msmc_b3_overlap_v, msmc_b3_v};
use std::time::Instant;

fn lj(r: f64) -> f64 {
    let inv = 1.0 / r;
    let s2 = inv * inv;
    let s6 = s2 * s2 * s2;
    4.0 * (s6 * s6 - s6)
}

fn main() {
    let sigma_hs = 1.5;
    let n = 20_000_000usize;

    println!("(A) plain MSMC vs overlap sampling  [N={n}, sig_HS={sigma_hs}, err*sqrt(N) = MC noise constant]\n");
    println!("  T*    cubature     plain: B3 (err*rtN)         overlap: B3 (err*rtN)      variance drop");
    for &t in &[1.0_f64, 1.5, 2.0, 5.0] {
        let (cub, _) = b3_cubature_v(&lj, t, 1e-8);
        let seed = 0xBEEF ^ t.to_bits();
        let p = msmc_b3_v(&lj, t, sigma_hs, n, seed);
        let o = msmc_b3_overlap_v(&lj, t, sigma_hs, n, seed);
        let kp = p.stderr * (n as f64).sqrt();
        let ko = o.stderr * (n as f64).sqrt();
        println!(
            "  {t:.1}   {cub:>8.5}    {:>8.5} ({:>7.1})        {:>8.5} ({:>6.2})        {:>4.1}x less",
            p.b3, kp, o.b3, ko, kp / ko
        );
    }

    // (B) parallelism: same total work split across threads
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let t = 1.5_f64;
    let (cub, _) = b3_cubature_v(&lj, t, 1e-8);
    println!("\n(B) parallel scaling  [overlap MSMC, T*={t}, total N={n}, {cores} cores available]");
    println!("    cubature B3 = {cub:.5}");
    let t1 = {
        let t0 = Instant::now();
        let r = msmc_b3_overlap_parallel(&lj, t, sigma_hs, n, 0xC0FFEE, 1);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        println!("    threads= 1 : B3={:.5} +/- {:.5}  [{ms:7.1} ms]  (1.0x)", r.b3, r.stderr);
        ms
    };
    for &k in &[2usize, 4, 8] {
        let t0 = Instant::now();
        let r = msmc_b3_overlap_parallel(&lj, t, sigma_hs, n, 0xC0FFEE, k);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        println!(
            "    threads={k:>2} : B3={:.5} +/- {:.5}  [{ms:7.1} ms]  ({:.1}x speedup)",
            r.b3, r.stderr, t1 / ms
        );
    }
}
