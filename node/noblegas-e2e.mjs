// End-to-end check of poc_noblegas through the built wasm: the classical slot
// matches order-0 and the quantum slot the WK-order-3, and they differ at low T.
import fs from "node:fs";
const bytes = fs.readFileSync("target/wasm32-unknown-unknown/release/potter_poc.wasm");
const { instance } = await WebAssembly.instantiate(bytes, {});
const ex = instance.exports;
function noble(gas, t, order) {
  const out = ex.poc_alloc(64);
  ex.poc_noblegas(gas, t, order, out);
  const dv = new DataView(ex.memory.buffer);
  const r = Array.from({ length: 8 }, (_, k) => dv.getFloat64(out + 8 * k, true));
  ex.poc_dealloc(out, 64);
  return r;
}
// reference classical B2 (cm^3/mol) from the noble-gas tests (integrate_potentials.py)
const REF = {
  // gas: [[T, B2_classical], ...]
  0: [[50, -38.5466], [300, 11.4662]],   // Ne
  1: [[50, -774.1828], [300, -15.2992]], // Ar
};
let ok = true;
for (const gas of [0, 1]) {
  for (const [t, b2cl] of REF[gas]) {
    const r = noble(gas, t, 3);
    const pass = Math.abs(r[0] - b2cl) / Math.abs(b2cl) < 2e-3;
    ok &&= pass;
    console.log(`gas=${gas} T=${t}: classical B2=${r[0].toFixed(4)} (ref ${b2cl}) ${pass ? "OK" : "FAIL"}`);
  }
}
// quantum differs from classical at 50 K, largest for Ne
const ne = noble(0, 50, 3), ar = noble(1, 50, 3);
const shiftNe = Math.abs(ne[0] - ne[4]), shiftAr = Math.abs(ar[0] - ar[4]);
const q = shiftNe > 0 && shiftAr > 0;
ok &&= q;
console.log(`WK shift @50K: Ne=${shiftNe.toFixed(3)} Ar=${shiftAr.toFixed(3)} ${q ? "OK" : "FAIL"}`);
console.log(ok ? "E2E PASS" : "E2E FAIL");
process.exit(ok ? 0 : 1);
