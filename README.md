# potter-poc — DSL-defined virial coefficients, integrated in Rust, run as WASM

A proof of concept for the idea behind a possible [potter](https://github.com/usnistgov/potter)
rewrite: instead of hard-coding each pair potential as a C++ template, the user
supplies the potential as a **Python-like expression string**, and a
self-contained **WASM module** parses it, integrates it, and returns virial
coefficients — with no recompilation and no C++ toolchain.

It computes the second and third virial coefficients of the Lennard-Jones (12-6)
fluid from the DSL string `4*eps*((sig/r)**12 - (sig/r)**6)`:

```
B2(T) = -2*pi * \int_0^inf ( e^{-V/T} - 1 ) r^2 dr
B3(T) = -(8*pi^2/3) \int\int\int  r1 r2 r3 f1 f2 f3   (triangle coordinates)
```

## What it demonstrates

- **DSL grammar in the plugin** (`src/dsl.rs`): a hand-rolled tokenizer + Pratt
  parser with Python semantics (`**` right-assoc, `exp()`, variables, …) →
  expression tree → evaluator.
- **Four DSL backends** behind the same `.v(r)` interface: hand-rolled tree-walk
  (`Potential`), CSE'd bytecode (`CsePotential`), **cranelift JIT to native code**
  (`JitPotential`, `src/jit.rs`, native-only), and **`fasteval`** (`src/fastpot.rs`).
- **"Cubature" in Rust** (`src/integrate.rs`): adaptive Simpson with local error
  control. B2 is 1-D; B3 is a 3-D nested-adaptive integral with the
  triangle-inequality constraint.
- **Generic over the potential**: integrators take any `Fn(f64)->f64`, so the DSL,
  fasteval, a hard-coded closure, or a hard-sphere step all plug in. The `[0,inf)`
  domain is mapped to `[0,1)` so there are no potential-specific tail hacks.
- **Pure Rust, one dependency on the hot path** (`fasteval`; `libm` only for the
  reference `tgamma`): the same code compiles to native, `wasm32-wasip1`, and
  `wasm32-unknown-unknown`.

## Correctness

`cargo test` (8 tests), all passing:

- B2 adaptive vs an independent 2M-panel grid (~3e-10).
- B2 = 0 at the LJ Boyle temperature T\* = 3.41793 (~9e-9).
- B2 vs the closed-form Hirschfelder-Curtiss-Bird Gamma series (~3e-8 for T\*≥2).
- **B3 hard-sphere anchor**: B2 = 2π/3 and B3 = (5/8)·B2² *exactly* — validates
  the B3 formula + constant against an exact analytic result.
- B3 adaptive vs an independent nested fixed grid.
- fasteval backend == hand-rolled DSL (values and B2/B3).
- All four implementations (Rust hard-coded / hand-DSL / fasteval / C++) agree on
  B3(T\*=1.5) = **2.38348616** to 8 digits.

## Rust vs C++ side-by-side (`./run_sxs.sh`)

Same algorithm, same potential, B3 at T\*=1.5, tol 1e-7 (Apple Silicon, min of 5
reps; absolute ms are machine-dependent, the *ratios* are the point):

| implementation                  | B3          | time     | vs hard-coded | note |
|---------------------------------|-------------|----------|---------------|------|
| C++ (clang -O3), hard-coded     | 2.38348616  | ~224 ms  | 0.9×          | **parity with Rust** |
| Rust, hard-coded closure        | 2.38348616  | ~245 ms  | 1.0×          | the language baseline |
| **Rust, cranelift JIT (native)**| 2.38348616  | **~275 ms** | **~1.1×**  | **native code from the DSL → hard-coded parity** |
| Rust, hand DSL (int-power+CSE)  | 2.38348616  | ~1155 ms | ~4.7×         | value-numbered DAG, 10 ops |
| Rust, hand DSL (int-power)      | 2.38348616  | ~1583 ms | ~6.5×         | `**12`/`**6` folded to `.powi` |
| Rust, hand DSL (`powf`)         | 2.38348616  | ~2567 ms | ~10.5×        | tree-walk + two transcendental `powf` |
| Rust, fasteval (`^`)            | 2.38348616  | ~3656 ms | ~14.9×        | generality overhead; still `powf` |

Takeaways:
- **Rust ≈ C++** on the identical hard-coded core. A straight port buys no raw
  speed (both are LLVM, transcendental-bound). *(Watch the fairness traps: an
  early run had C++ 2.4× slower purely because it used `std::pow(x,6)` instead of
  multiplication — see `cpp/b2b3.cpp`.)*
- **A runtime DSL spans the whole range — flexibility costs nothing once you JIT.**
  Starting from the naive interpreter and closing the gap in stages:
  - *Integer-power recognition* (fold `**12`/`**6` to `.powi`): 10.5× → 6.5×.
  - *CSE + bytecode flatten* (`src/dsl.rs`: value-numbered DAG, no recursion): → 4.7×.
  - **Cranelift JIT** (`src/jit.rs`: emit native machine code from the expression):
    → **~1.1×, i.e. hard-coded parity.**
- So the "user supplies a potential string at runtime" ergonomics can be had at
  full native speed. The JIT's residual ~10% over the hand-coded closure is the
  function-call boundary (the JIT'd `V(r)` isn't inlined into the integrator) plus
  codegen differences; it would vanish if the whole integrand were JIT-compiled.
- The JIT is native-only (cranelift emits host machine code); the WASM build keeps
  using the CSE interpreter. **For the browser, the AOT backend below does the
  equivalent — emit wasm bytecode the engine JITs to native.**
- *Gotcha found en route:* a naive bytecode evaluator that zero-inits an oversized
  scratch array each call was slower than the tree — fixed with a reused buffer.

## AOT: compile the DSL to a WebAssembly module (the browser JIT story)

`src/aot.rs` emits a standalone `.wasm` module from a DSL string (via
`wasm-encoder`), exporting `potential(r, eps, sig) -> f64`. The expression tree
maps onto wasm's stack machine; arithmetic / integer powers / `sqrt` / `abs` are
native wasm instructions, transcendentals are imported from `env`. The JS engine
(or any wasm runtime) then JITs that module to native code — so a user-supplied
potential becomes native-speed *in the browser*, with no toolchain.

```sh
cargo run --release --bin aot_demo   # writes target/lj_potential.wasm (+ morse)
node node/run-aot.mjs                # instantiate & run the generated modules
```

- Lennard-Jones compiles to a **136-byte, self-contained** module (no imports).
- A B2 computed by calling the generated native `potential` from JS round-trips
  to the correct value (≈ −5.316 at T\*=1). Morse compiles to 109 bytes importing
  `env.exp` (host supplies `Math.exp`).
- Same idea as the cranelift JIT, but the target is portable wasm bytecode instead
  of host machine code — `compile_to_wasm` is pure Rust and also runs *inside* the
  wasm module (wasm generating wasm).

## Browser page — [live demo](https://ibell.github.io/potter-wasm/)

A standalone single-file web app to compute B₂(T) for a potential you type — the
whole pipeline running in the browser. **[Try it live.](https://ibell.github.io/potter-wasm/)**

```sh
./web/build.sh            # builds the wasm and embeds it (base64) into docs/index.html
open docs/index.html      # opens via file:// — no server needed
```

The published page is `docs/index.html` (served by GitHub Pages), generated from
`web/index.html.in` by `web/build.sh`.

- **Presets with exact solutions:** Lennard-Jones (12-6), overlaid with the
  Hirschfelder-Curtiss-Bird Gamma series; and **inverse-power** ε(σ/r)ⁿ, overlaid
  with the exact B₂ = (2π/3)σ³·Γ(1−3/n)·(T/ε)^(−3/n) (n>3).
- **Custom potentials:** type any DSL expression; B₂ is computed numerically.
- The potential string is **AOT-compiled to a wasm module in the browser**
  (`poc_compile_wasm`), instantiated, and called natively to integrate B₂; the
  exact overlays reuse the Rust `b2_lj_series` / `tgamma` exposed as wasm exports.
- Results show a table (numeric vs exact vs |Δ|) and a B₂-vs-T plot. The compute
  core is verified against the exact solutions in `node/run-aot-browser.mjs`
  (numeric vs exact agree to ~1e-8 for LJ, ~1e-13 for inverse-power).

## Build & run

```sh
cargo run --release        # native B2/B3 demo + verification
cargo test --release       # the 8 checks
./run_sxs.sh               # Rust vs C++ B3 timing

# WASM for the node function-call host
cargo build --release --target wasm32-unknown-unknown --lib
node node/run.mjs          # node passes a DSL string into WASM, gets B2 & B3 back

# full program under node's built-in WASI
cargo build --release --target wasm32-wasip1 --bin potter_poc
node node/run-wasi.mjs

# AOT: DSL -> .wasm module, run by the JS engine
cargo run --release --bin aot_demo && node node/run-aot.mjs
```

## Notes / next steps

- **B3 has two integrators** (`src/physics.rs`): a simple nested-1-D adaptive
  Simpson, and a genuine 3-D **Genz-Malik adaptive cubature** (`src/cubature.rs`,
  pure Rust — degree-7 rule + embedded degree-5 error estimate + global region
  heap, the `hcubature` scheme). They agree with each other and with the
  hard-sphere exact anchor. `cargo run --release --bin b3bench` compares them:
  on a *smooth* integrand the rule reaches 3e-13 in ~30k evals; on LJ B3 it is
  ~3× faster than nesting at low T (where the repulsive wall makes the integrand
  sharp) and comparable at high T. The residual difficulty is that thin
  repulsive-wall layer — which is exactly why the field reaches for
  importance-sampling / Mayer-sampling Monte Carlo for B4 and beyond, rather than
  any uniform deterministic rule. That (and potter's Cuba/VEGAS path) is the real
  lever for the high-dimensional cases.
- Temperature derivatives (dⁿB/dTⁿ) are out of scope here, but note they wrap
  *around* the potential: V(r) is T-independent, so a generic/arbitrary potential
  and exact derivatives do not conflict (`num-dual` would supply the autodiff).
