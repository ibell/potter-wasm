//! Mayer-sampling Monte Carlo (Kofke & Singer, 2004) for the third virial
//! coefficient — the importance-sampling paradigm that takes over from
//! deterministic cubature as dimension grows.
//!
//! B3 = -(1/3) integral f12 f13 f23 over the positions of particles 2 and 3
//! (particle 1 at the origin) — a 6-D integral. We Metropolis-sample
//! configurations with probability proportional to |gamma|, gamma = f12 f13 f23,
//! and reference to a hard-sphere system whose B3 is known exactly:
//!
//!   B3 = B3_HS * <sign(gamma)> / <gamma0 / |gamma|>
//!
//! where gamma0 is the hard-sphere Mayer product and B3_HS = (5/8)(2*pi/3 *
//! sigma_HS^3)^2. The estimator is exact for any reference; sigma_HS only affects
//! the variance. MC error ~ 1/sqrt(N), independent of dimension — so the same
//! sampler does B4 (9-D) where cubature struggles.

use std::f64::consts::PI;

/// SplitMix64 — a tiny, dependency-free, deterministic PRNG (fine for MC).
struct Rng {
    s: u64,
}
impl Rng {
    fn new(seed: u64) -> Self {
        Rng { s: seed }
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.s = self.s.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    #[inline]
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    #[inline]
    fn sym(&mut self, d: f64) -> f64 {
        (self.unit() * 2.0 - 1.0) * d
    }
}

pub struct Msmc {
    pub b3: f64,
    pub stderr: f64,
    pub accept: f64,
}

#[inline]
fn norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}
#[inline]
fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    let (x, y, z) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    (x * x + y * y + z * z).sqrt()
}

/// MSMC estimate of B3 for any potential closure. `nsteps` Metropolis steps,
/// hard-sphere reference of diameter `sigma_hs`, deterministic given `seed`.
pub fn msmc_b3_v<V: Fn(f64) -> f64>(
    v: &V,
    t: f64,
    sigma_hs: f64,
    nsteps: usize,
    seed: u64,
) -> Msmc {
    let mayer = |r: f64| {
        let vv = v(r);
        if vv.is_finite() {
            (-vv / t).exp() - 1.0
        } else {
            -1.0
        }
    };
    let hs = |r: f64| if r < sigma_hs { -1.0 } else { 0.0 };
    let b3_hs = 0.625 * ((2.0 * PI / 3.0) * sigma_hs.powi(3)).powi(2);

    let gamma = |r2: [f64; 3], r3: [f64; 3]| mayer(norm(r2)) * mayer(norm(r3)) * mayer(dist(r2, r3));
    let gamma0 = |r2: [f64; 3], r3: [f64; 3]| hs(norm(r2)) * hs(norm(r3)) * hs(dist(r2, r3));

    // start in the support of |gamma| (all pairs interacting)
    let mut r2 = [1.05, 0.0, 0.0];
    let mut r3 = [0.0, 1.05, 0.0];
    let mut g = gamma(r2, r3);
    let mut ag = g.abs();

    let mut rng = Rng::new(seed);
    let delta = 0.5; // step size (~40-50% acceptance)
    let equil = nsteps / 10;
    let nblocks = 50usize;
    let per = (nsteps - equil).max(nblocks) / nblocks;

    let mut bsign = vec![0.0f64; nblocks];
    let mut bref = vec![0.0f64; nblocks];
    let mut bcnt = vec![0usize; nblocks];
    let mut accepts = 0usize;

    for step in 0..nsteps {
        let move2 = (rng.next_u64() & 1) == 0;
        let trial = if move2 {
            [r2[0] + rng.sym(delta), r2[1] + rng.sym(delta), r2[2] + rng.sym(delta)]
        } else {
            [r3[0] + rng.sym(delta), r3[1] + rng.sym(delta), r3[2] + rng.sym(delta)]
        };
        let gnew = if move2 { gamma(trial, r3) } else { gamma(r2, trial) };
        let agnew = gnew.abs();
        if agnew >= ag || rng.unit() < agnew / ag {
            if move2 {
                r2 = trial;
            } else {
                r3 = trial;
            }
            g = gnew;
            ag = agnew;
            accepts += 1;
        }
        if step >= equil {
            let b = ((step - equil) / per).min(nblocks - 1);
            bsign[b] += g.signum();
            bref[b] += gamma0(r2, r3) / ag;
            bcnt[b] += 1;
        }
    }

    // central value from pooled sums; error from the spread of per-block estimates
    let mut tot_s = 0.0;
    let mut tot_r = 0.0;
    let mut blocks = Vec::new();
    for b in 0..nblocks {
        if bcnt[b] == 0 || bref[b] == 0.0 {
            continue;
        }
        tot_s += bsign[b];
        tot_r += bref[b];
        blocks.push(b3_hs * bsign[b] / bref[b]);
    }
    let b3 = if tot_r != 0.0 {
        b3_hs * tot_s / tot_r
    } else {
        f64::NAN
    };
    let m = blocks.len().max(1);
    let mean = blocks.iter().sum::<f64>() / m as f64;
    let var = blocks.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (m.max(2) - 1) as f64;
    Msmc {
        b3,
        stderr: (var / m as f64).sqrt(),
        accept: accepts as f64 / nsteps as f64,
    }
}

/// Run `nthreads` independent overlap-sampling chains across threads (Monte Carlo
/// is embarrassingly parallel — no shared state), splitting `nsteps` between them
/// and combining the per-chain estimates. Pure std `thread::scope`, no deps.
pub fn msmc_b3_overlap_parallel<V: Fn(f64) -> f64 + Sync>(
    v: &V,
    t: f64,
    sigma_hs: f64,
    nsteps: usize,
    seed: u64,
    nthreads: usize,
) -> Msmc {
    let per = (nsteps / nthreads.max(1)).max(1);
    let chains: Vec<Msmc> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..nthreads)
            .map(|k| {
                let sk = seed ^ (k as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15);
                s.spawn(move || msmc_b3_overlap_v(v, t, sigma_hs, per, sk))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    // combine independent chains: mean of per-chain estimates, error from spread
    let bs: Vec<f64> = chains.iter().map(|c| c.b3).filter(|x| x.is_finite()).collect();
    let k = bs.len().max(1);
    let mean = bs.iter().sum::<f64>() / k as f64;
    let var = bs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (k.max(2) - 1) as f64;
    let acc = chains.iter().map(|c| c.accept).sum::<f64>() / chains.len().max(1) as f64;
    Msmc {
        b3: mean,
        stderr: (var / k as f64).sqrt(),
        accept: acc,
    }
}

/// Overlap-sampling MSMC (Bennett / Kofke): sample configurations with weight
/// w = |gamma_T| + |gamma_R| and estimate
///   B3 = B3_HS * <gamma_T / w> / <gamma_R / w>.
/// Both ratios are bounded by 1 (no blow-up where a Mayer product vanishes) and
/// the walk visits the reference's support, so this has far lower variance than
/// the plain estimator — especially at low T where the sign problem bites.
pub fn msmc_b3_overlap_v<V: Fn(f64) -> f64>(
    v: &V,
    t: f64,
    sigma_hs: f64,
    nsteps: usize,
    seed: u64,
) -> Msmc {
    let mayer = |r: f64| {
        let vv = v(r);
        if vv.is_finite() {
            (-vv / t).exp() - 1.0
        } else {
            -1.0
        }
    };
    let hs = |r: f64| -> f64 {
        if r < sigma_hs {
            -1.0
        } else {
            0.0
        }
    };
    let b3_hs = 0.625 * ((2.0 * PI / 3.0) * sigma_hs.powi(3)).powi(2);

    let gt = |r2: [f64; 3], r3: [f64; 3]| mayer(norm(r2)) * mayer(norm(r3)) * mayer(dist(r2, r3));
    let gr = |r2: [f64; 3], r3: [f64; 3]| hs(norm(r2)) * hs(norm(r3)) * hs(dist(r2, r3));
    let weight = |r2: [f64; 3], r3: [f64; 3]| gt(r2, r3).abs() + gr(r2, r3).abs();

    let mut r2 = [1.05, 0.0, 0.0];
    let mut r3 = [0.0, 1.05, 0.0];
    let mut w = weight(r2, r3);

    let mut rng = Rng::new(seed);
    let delta = 0.5;
    let equil = nsteps / 10;
    let nblocks = 50usize;
    let per = (nsteps - equil).max(nblocks) / nblocks;

    let mut bnum = vec![0.0f64; nblocks];
    let mut bden = vec![0.0f64; nblocks];
    let mut bcnt = vec![0usize; nblocks];
    let mut accepts = 0usize;

    for step in 0..nsteps {
        let move2 = (rng.next_u64() & 1) == 0;
        let trial = if move2 {
            [r2[0] + rng.sym(delta), r2[1] + rng.sym(delta), r2[2] + rng.sym(delta)]
        } else {
            [r3[0] + rng.sym(delta), r3[1] + rng.sym(delta), r3[2] + rng.sym(delta)]
        };
        let wnew = if move2 { weight(trial, r3) } else { weight(r2, trial) };
        if wnew >= w || (w > 0.0 && rng.unit() < wnew / w) {
            if move2 {
                r2 = trial;
            } else {
                r3 = trial;
            }
            w = wnew;
            accepts += 1;
        }
        if step >= equil && w > 0.0 {
            let b = ((step - equil) / per).min(nblocks - 1);
            bnum[b] += gt(r2, r3) / w;
            bden[b] += gr(r2, r3) / w;
            bcnt[b] += 1;
        }
    }

    let mut tot_n = 0.0;
    let mut tot_d = 0.0;
    let mut blocks = Vec::new();
    for b in 0..nblocks {
        if bcnt[b] == 0 || bden[b] == 0.0 {
            continue;
        }
        tot_n += bnum[b];
        tot_d += bden[b];
        blocks.push(b3_hs * bnum[b] / bden[b]);
    }
    let b3 = if tot_d != 0.0 {
        b3_hs * tot_n / tot_d
    } else {
        f64::NAN
    };
    let m = blocks.len().max(1);
    let mean = blocks.iter().sum::<f64>() / m as f64;
    let var = blocks.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (m.max(2) - 1) as f64;
    Msmc {
        b3,
        stderr: (var / m as f64).sqrt(),
        accept: accepts as f64 / nsteps as f64,
    }
}
