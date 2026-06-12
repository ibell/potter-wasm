// De-risk the web page's compute path in node: load potter_poc.wasm, AOT-compile
// a DSL potential to a child wasm module (poc_compile_wasm), instantiate it, and
// integrate B2 by calling the native child `potential`. The browser page uses the
// exact same APIs (WebAssembly, Math).

import { readFileSync } from "node:fs";

const wasm = readFileSync(
  new URL("../target/wasm32-unknown-unknown/release/potter_poc.wasm", import.meta.url),
);
const { instance } = await WebAssembly.instantiate(wasm, {});
const ex = instance.exports;

function compileToWasm(dsl) {
  const enc = new TextEncoder().encode(dsl);
  const sp = ex.poc_alloc(enc.length);
  new Uint8Array(ex.memory.buffer, sp, enc.length).set(enc);
  const ret = ex.poc_compile_wasm(sp, enc.length); // BigInt (i64)
  ex.poc_dealloc(sp, enc.length);
  if (ret === 0n) throw new Error("parse error");
  const ptr = Number(ret >> 32n);
  const len = Number(ret & 0xffffffffn);
  const bytes = new Uint8Array(ex.memory.buffer, ptr, len).slice();
  ex.poc_free(ptr, len);
  return bytes;
}

const env = {
  exp: Math.exp, ln: Math.log, log10: Math.log10,
  sin: Math.sin, cos: Math.cos, tan: Math.tan, powf: Math.pow,
};

async function makePotential(dsl) {
  const bytes = compileToWasm(dsl);
  const { instance } = await WebAssembly.instantiate(bytes, { env });
  return { fn: instance.exports.potential, size: bytes.length };
}

function b2(pot, eps, sig, T, n = 20000) {
  let sum = 0;
  for (let i = 0; i <= n; i++) {
    const s = i / n, om = 1 - s;
    let val = 0;
    if (om > 0) {
      const r = s / om, jac = 1 / (om * om);
      const V = pot(r, eps, sig);
      const m = Number.isFinite(V) ? Math.exp(-V / T) - 1 : -1;
      val = m * r * r * jac;
      if (!Number.isFinite(val)) val = 0;
    }
    const w = i === 0 || i === n ? 1 : i % 2 === 1 ? 4 : 2;
    sum += w * val;
  }
  return (-2 * Math.PI * sum) / (3 * n);
}

// exact solutions (call the Rust implementations exposed by the wasm module)
const b2LjExact = (T, eps, sig) => sig ** 3 * ex.poc_b2_lj_series(T / eps, 80);
const b2InvExact = (T, eps, sig, n) =>
  n > 3 ? ((2 * Math.PI) / 3) * sig ** 3 * ex.poc_tgamma(1 - 3 / n) * (T / eps) ** (-3 / n) : NaN;

console.log("Lennard-Jones (12-6): numeric (AOT wasm) vs exact (HCB series)");
const lj = await makePotential("4*eps*((sig/r)**12 - (sig/r)**6)");
console.log("  module:", lj.size, "bytes");
for (const T of [1.5, 2.0, 3.417928, 5.0, 10.0]) {
  const num = b2(lj.fn, 1, 1, T), exa = b2LjExact(T, 1, 1);
  console.log(`  T*=${T.toFixed(4)}  B2=${num.toFixed(5)}  exact=${exa.toFixed(5)}  |d|=${Math.abs(num - exa).toExponential(1)}`);
}

console.log("\nInverse power V=eps*(sig/r)^12: numeric vs exact (2pi/3)Gamma(1-3/n)(T/eps)^(-3/n)");
const inv = await makePotential("eps*(sig/r)**12");
console.log("  module:", inv.size, "bytes");
for (const T of [1.0, 2.0, 5.0]) {
  const num = b2(inv.fn, 1, 1, T), exa = b2InvExact(T, 1, 1, 12);
  console.log(`  T*=${T.toFixed(4)}  B2=${num.toFixed(5)}  exact=${exa.toFixed(5)}  |d|=${Math.abs(num - exa).toExponential(1)}`);
}
