// node.js test: load the pure-Rust WASM module (wasm32-unknown-unknown), pass a
// Python-like DSL potential string *into* WASM, and get B2 back. This is the
// "host supplies a potential, the plugin computes the virial coefficient" flow.
//
//   build:  cargo build --release --target wasm32-unknown-unknown --lib
//   run:    node node/run.mjs

import { readFileSync } from "node:fs";

const wasmPath = new URL(
  "../target/wasm32-unknown-unknown/release/potter_poc.wasm",
  import.meta.url,
);
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const { memory, poc_alloc, poc_dealloc, poc_b2, poc_b3 } = instance.exports;

// Marshal a JS string into WASM linear memory, call `fn`, free it.
function call(fn, dsl, eps, sig, t) {
  const bytes = new TextEncoder().encode(dsl);
  const ptr = poc_alloc(bytes.length);
  new Uint8Array(memory.buffer, ptr, bytes.length).set(bytes);
  const v = fn(ptr, bytes.length, eps, sig, t);
  poc_dealloc(ptr, bytes.length);
  return v;
}
const b2 = (dsl, eps, sig, t) => call(poc_b2, dsl, eps, sig, t);
const b3 = (dsl, eps, sig, t) => call(poc_b3, dsl, eps, sig, t);

const LJ = "4*eps*((sig/r)**12 - (sig/r)**6)";

console.log("node.js -> WASM (wasm32-unknown-unknown): DSL parsed & integrated inside WASM");
console.log("  V(r) =", LJ, "\n");
console.log("      T*        B2 (WASM)       B3 (WASM)");
console.log("  --------------------------------------");
for (const t of [1.0, 1.5, 2.0, 2.5, 3.0, 3.417928, 5.0, 10.0]) {
  console.log(
    "  " +
      t.toFixed(4).padStart(7) +
      "   " +
      b2(LJ, 1.0, 1.0, t).toFixed(6).padStart(13) +
      "   " +
      b3(LJ, 1.0, 1.0, t).toFixed(6).padStart(13),
  );
}
console.log(
  "\n  Boyle check: B2(3.417928) =",
  b2(LJ, 1.0, 1.0, 3.417928).toExponential(3),
  "(expect ~0)",
);

// Show genericity: a *different* potential (soft-sphere r^-12) with no rebuild.
const SOFT = "4*eps*(sig/r)**12";
console.log("\n  swap potential to V(r) =", SOFT, "(no rebuild):");
console.log("  B2(T*=2) =", b2(SOFT, 1.0, 1.0, 2.0).toFixed(6));
