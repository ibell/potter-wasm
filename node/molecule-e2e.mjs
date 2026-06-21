import fs from "node:fs";
const bytes = fs.readFileSync("target/wasm32-unknown-unknown/release/potter_poc.wasm");
const { instance } = await WebAssembly.instantiate(bytes, {});
const ex = instance.exports;
function mol(id, t) {
  const out = ex.poc_alloc(64); ex.poc_molecule(id, t, 1e-3, out);
  const dv = new DataView(ex.memory.buffer);
  const r = [...Array(8)].map((_,k)=>dv.getFloat64(out+8*k,true));
  ex.poc_dealloc(out,64); return r;
}
// Hellmann CO2 (id 3) QFH vs SI B_QFH col5; Hellmann N2 (id2) vs SI(WK) col3
let ok=true;
for (const [id,t,ref,band,nm] of [[3,250,-184.16,0.05,"CO2 QFH"],[3,400,-59.87,0.05,"CO2 QFH"],
    [2,90,-195.57,0.2,"N2 QFH/WK"],[2,500,16.61,0.2,"N2 QFH/WK"]]) {
  const r=mol(id,t), pass=Math.abs(r[4]-ref)<band; ok&&=pass;
  console.log(`${nm} id${id} T=${t}: q.b2=${r[4].toFixed(3)} (ref ${ref}) ${pass?"OK":"FAIL"}`);
}
// empirical models: quantum slots NaN
const tr=mol(0,300);
const nanq = Number.isNaN(tr[4]); ok&&=nanq;
console.log(`TraPPE N2 quantum NaN: ${nanq?"OK":"FAIL"} (cl b2=${tr[0].toFixed(3)})`);
console.log(ok?"E2E PASS":"E2E FAIL"); process.exit(ok?0:1);
