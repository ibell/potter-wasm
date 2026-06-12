// node.js test of the AOT path: load the .wasm modules generated from DSL strings
// (by `cargo run --release --bin aot_demo`), instantiate them — the JS engine JITs
// them to native — and call/integrate the potential. This mirrors the browser
// path: user-defined potentials become native-speed wasm with no toolchain.
//
//   cargo run --release --bin aot_demo      # writes target/*.wasm
//   node node/run-aot.mjs

import { readFileSync } from "node:fs";

function load(path, imports = {}) {
  const url = new URL("../" + path, import.meta.url);
  const mod = new WebAssembly.Module(readFileSync(url));
  return new WebAssembly.Instance(mod, imports).exports;
}

// ---- Lennard-Jones: self-contained module, no imports ----
const lj = load("target/lj_potential.wasm").potential;
console.log("AOT wasm generated from DSL, run natively by the JS engine:");
console.log("  lennard-jones V(r) = 4*eps*((sig/r)**12 - (sig/r)**6)");
for (const r of [0.95, 1.0, 1.122462, 1.5, 2.0]) {
  const hard = 4 * ((1 / r) ** 12 - (1 / r) ** 6);
  console.log(
    `    r=${r.toFixed(4)}  wasm V=${lj(r, 1, 1).toFixed(8)}  (JS check ${hard.toFixed(8)})`,
  );
}

// Compute B2 in JS by calling the native wasm potential in an integration loop.
function b2(potential, T, n = 200000) {
  // B2 = -2*pi * \int_0^1 (exp(-V(r)/T)-1) r^2 (dr/ds) ds, r = s/(1-s)
  let sum = 0;
  for (let i = 0; i <= n; i++) {
    const s = i / n;
    const om = 1 - s;
    let val = 0;
    if (om > 0) {
      const r = s / om;
      const jac = 1 / (om * om);
      const V = potential(r, 1, 1);
      const m = Number.isFinite(V) ? Math.exp(-V / T) - 1 : -1;
      val = m * r * r * jac;
      if (!Number.isFinite(val)) val = 0;
    }
    const w = i === 0 || i === n ? 1 : i % 2 === 1 ? 4 : 2; // Simpson
    sum += w * val;
  }
  return (-2 * Math.PI * (sum * (1 / n)) / 3) * 1; // h/3 with h=1/n
}
console.log(
  `\n  B2(T*=1.0) via native wasm potential = ${b2(lj, 1.0).toFixed(5)} (expect ~ -5.316)`,
);
console.log(`  B2(T*=2.0) via native wasm potential = ${b2(lj, 2.0).toFixed(5)} (expect ~ -1.314)`);

// ---- Morse: needs env.exp supplied by the host ----
const morse = load("target/morse_potential.wasm", { env: { exp: Math.exp } }).potential;
console.log("\n  morse V(r) = eps*(exp(-2*(r-sig)) - 2*exp(-(r-sig)))  (imports env.exp)");
for (const r of [0.8, 1.0, 1.5, 2.0]) {
  console.log(`    r=${r.toFixed(2)}  wasm V=${morse(r, 1, 1).toFixed(6)}`);
}
