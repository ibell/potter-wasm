// e2e: poc_quantum_b2 through the built wasm vs the Cencek 2012 tabulated B2.
// NOTE: the full-quantum phase-shift engine runs SINGLE-THREADED in wasm (no threads), so
// this exercises only the fastest high-T points (smallest matching radius) to keep the ABI
// check to a few minutes. The physics is validated comprehensively by the native test suite.
import fs from "node:fs";
const bytes = fs.readFileSync("target/wasm32-unknown-unknown/release/potter_poc.wasm");
const { instance } = await WebAssembly.instantiate(bytes, {});
const ex = instance.exports;

function q(species, t) {
  const out = ex.poc_alloc(32);
  ex.poc_quantum_b2(species, t, out);
  const dv = new DataView(ex.memory.buffer);
  const r = [0, 1, 2, 3].map((k) => dv.getFloat64(out + 8 * k, true));
  ex.poc_dealloc(out, 32);
  return r;
}

let ok = true;
// 4He and 3He at T=500 K (fastest: smallest matching radius) vs Cencek 2012 B2.
for (const [sp, t, ref, nm] of [[0, 500, 11.00715, "4He"], [1, 500, 11.05373, "3He"]]) {
  const r = q(sp, t);
  const pass = Number.isFinite(r[0]) && Math.abs(r[0] - ref) < 0.1 && Number.isFinite(r[3]);
  ok &&= pass;
  console.log(
    `${nm} T=${t}: B2=${r[0].toFixed(4)} (ref ${ref}) neff=${r[3].toFixed(2)} ${pass ? "OK" : "FAIL"}`
  );
}
console.log(ok ? "E2E PASS" : "E2E FAIL");
process.exit(ok ? 0 : 1);
