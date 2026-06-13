// End-to-end check of the poc_stockmayer wasm export: (mu*)^2=0 must reproduce
// the LJ n_eff, and n_eff must decrease with (mu*)^2 at fixed T*.
import fs from "node:fs";
const bytes = fs.readFileSync("target/wasm32-unknown-unknown/release/potter_poc.wasm");
const { instance } = await WebAssembly.instantiate(bytes, {});
const ex = instance.exports;
function stock(tstar, mu2, reltol = 1e-4) {
  const out = ex.poc_alloc(32);
  ex.poc_stockmayer(tstar, mu2, reltol, out);
  const dv = new DataView(ex.memory.buffer);
  const r = [0, 1, 2, 3].map(k => dv.getFloat64(out + 8 * k, true));
  ex.poc_dealloc(out, 32);
  return { b2: r[0], db2: r[1], d2b2: r[2], neff: r[3] };
}
const neffFrom = (b2, db2, d2b2, T) => (-3 * (b2 + T * db2)) / (2 * T * db2 + T * T * d2b2);
function lj(T, n = 40000) { let s0 = 0, s1 = 0, s2 = 0;
  for (let i = 0; i <= n; i++) { const s = i / n, om = 1 - s; let f0 = 0, f1 = 0, f2 = 0;
    if (om > 0) { const r = s / om, w = r * r / (om * om), sr6 = (1 / r) ** 6, V = 4 * (sr6 * sr6 - sr6);
      if (Number.isFinite(V)) { const e = Math.exp(-V / T), t2 = T * T; f0 = (e - 1) * w; f1 = (e * V / t2) * w; f2 = (e * (V * V / (t2 * t2) - 2 * V / (t2 * T))) * w; } else f0 = -w;
      if (!Number.isFinite(f0)) f0 = 0; if (!Number.isFinite(f1)) f1 = 0; if (!Number.isFinite(f2)) f2 = 0; }
    const wt = i === 0 || i === n ? 1 : i % 2 ? 4 : 2; s0 += wt * f0; s1 += wt * f1; s2 += wt * f2; }
  const k = -2 * Math.PI / (3 * n); const b2 = k * s0, db2 = k * s1, d2b2 = k * s2; return neffFrom(b2, db2, d2b2, T); }
let ok = true;
for (const T of [2.0, 5.0]) {
  const s = stock(T, 0).neff, l = lj(T);
  const pass = Math.abs(s - l) < 0.02; ok &&= pass;
  console.log(`mu2=0 T*=${T}: stock neff=${s.toFixed(3)} vs LJ ${l.toFixed(3)} ${pass ? "OK" : "FAIL"}`);
}
for (const T of [3.0]) {
  const a = stock(T, 0).neff, b = stock(T, 2).neff, c = stock(T, 4).neff;
  const pass = a > b && b > c; ok &&= pass;
  console.log(`dipole T*=${T}: neff ${a.toFixed(2)} > ${b.toFixed(2)} > ${c.toFixed(2)} ${pass ? "OK" : "FAIL"}`);
}
console.log(ok ? "E2E PASS" : "E2E FAIL");
process.exit(ok ? 0 : 1);
