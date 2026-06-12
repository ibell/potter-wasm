//! Rust side of the B3 timing side-by-side. Times the SAME nested-adaptive B3 at
//! T*=1.2 with three potential backends, so we can compare:
//!   - hard-coded closure  vs  C++ hard-coded  -> the language question
//!   - hard-coded vs hand-DSL vs fasteval      -> the cost of flexibility

use potter_poc::{b3_v, CsePotential, FastPotential, JitPotential, Potential};
use std::time::Instant;

const LJ: &str = "4*eps*((sig/r)**12 - (sig/r)**6)";

fn lj_hard(r: f64) -> f64 {
    // explicit multiplication to match the C++ baseline op-for-op.
    let inv = 1.0 / r;
    let s2 = inv * inv;
    let s6 = s2 * s2 * s2;
    4.0 * (s6 * s6 - s6)
}

fn time_b3<F: Fn(f64) -> f64>(v: &F, t: f64, tol: f64, reps: usize) -> (f64, f64) {
    let mut best = f64::INFINITY;
    let mut val = 0.0;
    for _ in 0..reps {
        let t0 = Instant::now();
        val = b3_v(v, t, tol);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        if ms < best {
            best = ms;
        }
    }
    (val, best)
}

fn main() {
    let t = 1.5;
    let tol = 1e-7;
    let hand_raw = Potential::compile_unoptimized(LJ, 1.0, 1.0).unwrap();
    let hand_opt = Potential::compile(LJ, 1.0, 1.0).unwrap();
    let cse = CsePotential::compile(LJ, 1.0, 1.0).unwrap();
    let jit = JitPotential::compile(LJ, 1.0, 1.0).unwrap();
    let fast = FastPotential::compile(LJ, 1.0, 1.0).unwrap();

    println!("Rust B3 timing @ T*={t}, tol={tol:e} (min of 5 reps)");
    let (v0, m0) = time_b3(&lj_hard, t, tol, 5);
    println!("  hard-coded closure        : B3={v0:.8}  [{m0:.1} ms]");
    let (v1, m1) = time_b3(&|r| hand_raw.v(r), t, tol, 5);
    println!("  hand DSL (powf)           : B3={v1:.8}  [{m1:.1} ms]");
    let (v2, m2) = time_b3(&|r| hand_opt.v(r), t, tol, 5);
    println!("  hand DSL (int-power)      : B3={v2:.8}  [{m2:.1} ms]");
    let (v3, m3) = time_b3(&|r| cse.v(r), t, tol, 5);
    println!("  hand DSL (int-power+CSE)  : B3={v3:.8}  [{m3:.1} ms]  ({} ops)", cse.ops());
    let (v5, m5) = time_b3(&|r| jit.v(r), t, tol, 5);
    println!("  cranelift JIT (native)    : B3={v5:.8}  [{m5:.1} ms]");
    let (v4, m4) = time_b3(&|r| fast.v(r), t, tol, 5);
    println!("  fasteval (powf/^)         : B3={v4:.8}  [{m4:.1} ms]");
}
