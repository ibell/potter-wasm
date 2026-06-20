# Web "Real fluids" mode (noble-gas B₂ with quantum corrections) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a top-level mode toggle to the browser app — *Model potentials (reduced)* ⇄ *Real fluids (SI)* — where real-fluids mode plots noble-gas (Ne/Ar/Kr/Xe) classical and Wigner–Kirkwood-corrected B₂(T)/derivatives/n_eff in cm³/mol vs T (K).

**Architecture:** Tiny grid-reuse helpers in `src/noblegas.rs` let one grid build serve both the classical and WK curves; a per-temperature `poc_noblegas` wasm export returns both (8 f64); the web reuses the existing time-sliced `compute()` loop (so lin/log spacing comes free) with a new noble branch, a parameterized `renderPlotly` (second curve + physical labels), and a physical table/copy/definitions fork.

**Tech Stack:** Rust + `num-dual` (native, already a dep), wasm C-ABI exports, `web/index.html.in` (Plotly). Tests in `src/main.rs`, a node e2e script, and build/syntax checks.

**Reference spec:** `docs/superpowers/specs/2026-06-20-web-real-fluids-noble-gas-design.md`

**Conventions:** tests in `src/main.rs` `#[cfg(test)] mod tests` import from `potter_poc::…`. Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. The web template is `web/index.html.in`; `./web/build.sh` regenerates `docs/index.html` (embeds the wasm). Measured: `b2_neff` ≈ 2.6 ms/call native incl. the 10000-pt grid; a 24-T classical+quantum sweep ≈ 125 ms native (~0.3–0.5 s wasm), so the existing time-sliced loop never janks.

**Existing facts the plan relies on (verified):**
- `src/noblegas.rs`: `TangToennies` with private `grid_potvals(&self) -> Vec<[f64;5]>`, `b2_generic<D: DualNum<f64>+Copy>(&self, t: D, order: u8, pv: &[[f64;5]]) -> D`, and `pub fn b2_neff(&self, t: f64, order: u8) -> (f64,f64,f64,f64)` (currently builds the grid then `second_derivative`). Constructors `neon_tt/argon_tt/krypton_tt/xenon_tt`.
- `src/lib.rs`: `pub mod noblegas; pub use noblegas::{argon_tt, krypton_tt, neon_tt, xenon_tt, TangToennies};` (lines 12-13). Native `pub fn stockmayer_b2_derivs(...)` (~line 49). Inside `#[cfg(target_arch="wasm32")] mod wasm_exports`: `use super::{b2_derivs_from_dsl, b2_from_dsl, b3_from_dsl, stockmayer_b2_derivs};` and `pub extern "C" fn poc_stockmayer(tstar, mu2, reltol, out: *mut f64)` writing 4 f64 via `out.add(k).write_unaligned(*v)`.
- `web/index.html.in`: `stockmayerReduced` (alloc/DataView/dealloc pattern), `computeRow(preset, pot, T, eps, sig)`, async `compute()` (time-sliced, `runGen` cancel), `renderPlotly(rows, hasExact, logX)`, `renderTable(rows, hasExact)`, `copyTable()`, `tGrid(tmin,tmax,np,log)`, `repExponent()`, element-ref block (`const presetSel = $("preset")…`), `syncPreset()`, the boot IIFE wiring `$("go").onclick`/`$("tspace").onchange`/`tmin,tmax,npts`.

---

## Task 1: Grid-reuse helpers in `noblegas.rs`

**Files:** Modify `src/noblegas.rs`; Test `src/main.rs`.

- [ ] **Step 1: Write the failing test** (in `src/main.rs` `mod tests`)

```rust
    #[test]
    fn noblegas_grid_reuse_matches_b2_neff() {
        use potter_poc::noblegas::argon_tt;
        let g = argon_tt();
        let pv = g.grid();
        assert_eq!(pv.len(), 10000, "grid size");
        for &(t, order) in &[(120.0_f64, 0u8), (300.0, 3), (800.0, 1)] {
            let a = g.b2_neff_with_grid(t, order, &pv);
            let b = g.b2_neff(t, order);
            assert!((a.0 - b.0).abs() < 1e-12 && (a.1 - b.1).abs() < 1e-12
                 && (a.2 - b.2).abs() < 1e-12 && (a.3 - b.3).abs() < 1e-12,
                 "T={t} order={order}: {a:?} vs {b:?}");
        }
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test noblegas_grid_reuse_matches_b2_neff`
Expected: FAIL — no methods `grid` / `b2_neff_with_grid`.

- [ ] **Step 3: Implement** — in `src/noblegas.rs`, find the current `b2_neff` method:

```rust
    pub fn b2_neff(&self, t: f64, order: u8) -> (f64, f64, f64, f64) {
        let pv = self.grid_potvals();
        let (b2, db2_dt, d2b2_dt2) =
            num_dual::second_derivative(|tt| self.b2_generic(tt, order, &pv), t);
        let neff = -3.0 * (b2 + t * db2_dt) / (2.0 * t * db2_dt + t * t * d2b2_dt2);
        (b2, db2_dt, d2b2_dt2, neff)
    }
```

Replace it with these three items (the public grid accessor, the grid-reusing core, and `b2_neff` as a thin wrapper):

```rust
    /// The T-independent precomputed grid (per point `[R_m, V_J, V'_{J/m},
    /// V''_{J/m²}, V'''_{J/m³}]`). Build once and pass to `b2_neff_with_grid` to
    /// avoid rebuilding it for every temperature / each WK order.
    pub fn grid(&self) -> Vec<[f64; 5]> {
        self.grid_potvals()
    }

    /// B₂ [cm³/mol], dB₂/dT, d²B₂/dT², n_eff at temperature `t` [K] and WK
    /// truncation `order`, reusing a prebuilt grid from `grid()`.
    pub fn b2_neff_with_grid(&self, t: f64, order: u8, pv: &[[f64; 5]]) -> (f64, f64, f64, f64) {
        let (b2, db2_dt, d2b2_dt2) =
            num_dual::second_derivative(|tt| self.b2_generic(tt, order, pv), t);
        let neff = -3.0 * (b2 + t * db2_dt) / (2.0 * t * db2_dt + t * t * d2b2_dt2);
        (b2, db2_dt, d2b2_dt2, neff)
    }

    /// B₂ [cm³/mol], dB₂/dT, d²B₂/dT², n_eff at temperature `t` [K], WK truncation
    /// `order`. Builds the grid then delegates to `b2_neff_with_grid`.
    pub fn b2_neff(&self, t: f64, order: u8) -> (f64, f64, f64, f64) {
        let pv = self.grid_potvals();
        self.b2_neff_with_grid(t, order, &pv)
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test noblegas_grid_reuse_matches_b2_neff`
Expected: PASS. Also run `cargo test noblegas_` to confirm the existing noble-gas tests still pass (behavior-preserving refactor).

- [ ] **Step 5: Commit**

```bash
git add src/noblegas.rs src/main.rs
git commit -m "$(cat <<'EOF'
Add grid-reuse helpers to TangToennies (grid, b2_neff_with_grid)

b2_neff is now a thin wrapper; the grid (10000-pt Dual3 build) can be
built once and reused across temperatures / WK orders. Equivalence
verified vs b2_neff.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: native `noblegas_b2_neff` + `poc_noblegas` wasm export

**Files:** Modify `src/lib.rs`; Test `src/main.rs`.

- [ ] **Step 1: Write the failing test** (in `src/main.rs` `mod tests`)

```rust
    #[test]
    fn noblegas_b2_neff_array_classical_and_quantum() {
        use potter_poc::{noblegas::argon_tt, noblegas_b2_neff};
        // gas 1 = Argon; order 3 quantum. Returns
        // [b2_cl,db2_cl,d2b2_cl,neff_cl, b2_q,db2_q,d2b2_q,neff_q].
        let a = noblegas_b2_neff(1, 300.0, 3);
        let g = argon_tt();
        let cl = g.b2_neff(300.0, 0);
        let q = g.b2_neff(300.0, 3);
        assert!((a[0] - cl.0).abs() < 1e-12 && (a[3] - cl.3).abs() < 1e-12, "classical slot");
        assert!((a[4] - q.0).abs() < 1e-12 && (a[7] - q.3).abs() < 1e-12, "quantum slot");
        // quantum correction is a real shift at low T
        let lo = noblegas_b2_neff(0, 50.0, 3); // Neon, 50 K
        assert!((lo[0] - lo[4]).abs() > 1e-6, "Ne 50K classical vs WK differ");
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test noblegas_b2_neff_array_classical_and_quantum`
Expected: FAIL — `noblegas_b2_neff` not found.

- [ ] **Step 3: Implement**

(3a) In `src/lib.rs`, after the `stockmayer_b2_derivs` function (a native `pub fn` in the crate root), add:

```rust
/// Noble-gas B₂ in physical units (cm³/mol, K): classical (order 0) and WK-order-N.
/// `gas`: 0=Ne 1=Ar 2=Kr 3=Xe. Returns
/// `[b2_cl, db2dT_cl, d2b2dT2_cl, neff_cl,  b2_q, db2dT_q, d2b2dT2_q, neff_q]`.
/// One grid build serves both curves.
pub fn noblegas_b2_neff(gas: u32, t: f64, order: u8) -> [f64; 8] {
    use crate::noblegas::{argon_tt, krypton_tt, neon_tt, xenon_tt};
    let g = match gas {
        0 => neon_tt(),
        1 => argon_tt(),
        2 => krypton_tt(),
        _ => xenon_tt(),
    };
    let pv = g.grid();
    let (b0, d0, e0, n0) = g.b2_neff_with_grid(t, 0, &pv);
    let (bq, dq, eq, nq) = g.b2_neff_with_grid(t, order, &pv);
    [b0, d0, e0, n0, bq, dq, eq, nq]
}
```

(3b) In `mod wasm_exports`, update the `use super::{...};` line. The current line is:
```rust
    use super::{b2_derivs_from_dsl, b2_from_dsl, b3_from_dsl, stockmayer_b2_derivs};
```
Change it to:
```rust
    use super::{b2_derivs_from_dsl, b2_from_dsl, b3_from_dsl, noblegas_b2_neff, stockmayer_b2_derivs};
```

(3c) In `mod wasm_exports`, add this export right after `poc_stockmayer`:

```rust
    /// Noble-gas classical + WK-order-N B₂ in physical units (cm³/mol, K). Writes
    /// 8 f64 into `out`: [b2_cl,db2dT_cl,d2b2dT2_cl,neff_cl, b2_q,db2dT_q,d2b2dT2_q,neff_q].
    /// `gas`: 0=Ne 1=Ar 2=Kr 3=Xe. `order`: WK truncation for the quantum curve.
    /// Unaligned writes so `out` need not be 8-byte aligned.
    #[no_mangle]
    pub extern "C" fn poc_noblegas(gas: u32, t: f64, order: u32, out: *mut f64) {
        let vals = noblegas_b2_neff(gas, t, order as u8);
        unsafe {
            for (k, v) in vals.iter().enumerate() {
                out.add(k).write_unaligned(*v);
            }
        }
    }
```

- [ ] **Step 4: Verify**

Run: `cargo test noblegas_b2_neff_array_classical_and_quantum`  → PASS.
Run: `cargo build`  → clean.
Run: `cargo build --lib --target wasm32-unknown-unknown`  → MUST compile (this is what actually compiles `poc_noblegas`; num-dual is pure Rust). Only pre-existing warnings (`aot.rs` unused_mut) allowed. Report the exact result.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/main.rs
git commit -m "$(cat <<'EOF'
Add noblegas_b2_neff + poc_noblegas wasm export (classical + WK)

Per-temperature export writing 8 f64 (classical order-0 and WK order-N
B2/derivs/neff, cm^3/mol & K); one grid build serves both curves.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Parameterize `renderPlotly` (default = current reduced behavior)

**Files:** Modify `web/index.html.in`.

This is a pure refactor: add an `opts` parameter that defaults to exactly the current reduced behavior, so the existing callers are unaffected. No new feature yet.

- [ ] **Step 1: Replace `renderPlotly`** — find the current `function renderPlotly(rows, hasExact, logX) { ... }` (the whole function, from its signature through the closing `}` before the copy-table comment) and replace it with:

```javascript
// Four vertically-stacked subplots (B₂, dB₂/dT, d²B₂/dT², n_eff) sharing the T
// axis. opts (defaults = reduced model-potential mode):
//   xlabel, panels:[B₂,dB₂,d²B₂] titles (n_eff is fixed), hover (x label),
//   series2: null | {key, name, color, dash} for a second line per panel,
//   ref: bool (draw the n=repExponent() dashed high-T reference on n_eff).
function renderPlotly(rows, hasExact, logX, opts = {}) {
  const host = $("plots");
  if (!host) return;
  if (typeof Plotly === "undefined") {
    host.innerHTML = "<div class=note>Plots are rendered with Plotly (loaded from a CDN). " +
      "Offline file:// has no network, so the plots are unavailable here — the table and " +
      "“Copy table” still work.</div>";
    return;
  }
  const o = {
    xlabel: "k_BT/ε",
    panels: ["B₂/σ³", "dB₂*/dT*", "d²B₂*/dT*²"],
    hover: "k_BT/ε",
    series2: null,
    ref: true,
    ...opts,
  };
  const Ts = rows.map(r => r.T);
  const BLUE = "#2a6df4", RED = "#e0533d";
  const line = (y, axis, color, dash) => ({
    x: Ts, y, type: "scatter", mode: "lines",
    line: { color: color || BLUE, width: 2, dash: dash || "solid" },
    yaxis: axis, hovertemplate: `${o.hover}=%{x:.3g}<br>%{y:.5g}<extra></extra>`,
  });
  const dots = (y, axis) => ({
    x: Ts, y, type: "scatter", mode: "markers", marker: { color: RED, size: 6 },
    yaxis: axis, hovertemplate: "exact %{y:.5g}<extra></extra>",
  });

  const traces = [
    line(rows.map(r => r.num), "y4"),
    line(rows.map(r => r.db2), "y3"),
    line(rows.map(r => r.d2b2), "y2"),
    line(rows.map(r => r.neff), "y"),
  ];
  if (hasExact) {
    traces.push(dots(rows.map(r => (r.ex ? r.ex.b2 : null)), "y4"));
    traces.push(dots(rows.map(r => (r.ex ? r.ex.db2 : null)), "y3"));
    traces.push(dots(rows.map(r => (r.ex ? r.ex.d2b2 : null)), "y2"));
    traces.push(dots(rows.map(r => (r.ex ? r.ex.neff : null)), "y"));
  }
  if (o.series2) {
    const k = o.series2.key, c = o.series2.color, d = o.series2.dash || "dot";
    traces[0].name = "classical"; // blue series gets a legend name
    const s = (sel, axis) => { const t = line(rows.map(r => (r[k] ? r[k][sel] : null)), axis, c, d);
      t.name = o.series2.name; return t; };
    traces.push(s("b2", "y4"), s("db2", "y3"), s("d2b2", "y2"), s("neff", "y"));
  }

  const band = (lo, hi, title) => ({
    domain: [lo, hi], title: { text: title, font: { size: 12 } }, zeroline: true,
    automargin: true, ticks: "outside", tickfont: { size: 10 },
    showline: true, linecolor: "#ddd",
  });
  const layout = {
    height: 660, margin: { l: 72, r: 16, t: 20, b: 44 }, showlegend: !!o.series2,
    legend: { orientation: "h", y: 1.02, x: 0 },
    font: { size: 11 }, paper_bgcolor: "#fff", plot_bgcolor: "#fff",
    xaxis: {
      domain: [0, 1], anchor: "y", title: { text: o.xlabel },
      type: logX ? "log" : "linear", ticks: "outside", showline: true, linecolor: "#ddd",
    },
    yaxis: band(0.0, 0.21, "n_eff"),
    yaxis2: band(0.27, 0.47, o.panels[2]),
    yaxis3: band(0.53, 0.73, o.panels[1]),
    yaxis4: band(0.79, 1.0, o.panels[0]),
  };
  if (o.ref) {
    // dashed reference at the repulsive exponent n on the n_eff (bottom) panel.
    const ref = repExponent();
    if (Number.isFinite(ref)) {
      layout.shapes = [{ type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: ref, y1: ref,
        line: { color: "#bbb", width: 1, dash: "dash" } }];
      layout.annotations = [{ xref: "paper", x: 0.012, yref: "y", y: ref,
        text: "n = " + ref + " (high-T limit)", showarrow: false,
        font: { size: 10, color: "#999" }, xanchor: "left", yanchor: "bottom" }];
    }
  }
  Plotly.react(host, traces, layout, { displayModeBar: false, responsive: true });
}
```

- [ ] **Step 2: Build + syntax check + confirm reduced behavior unchanged**

Run: `./web/build.sh`
```bash
python3 - <<'PY'
import re
html=open("web/index.html.in").read()
m=re.search(r'<script>\n(.*)</script>', html, re.S)
open('/tmp/webjs_t3.js','w').write(m.group(1))
PY
node --check /tmp/webjs_t3.js && echo "JS SYNTAX OK"
```
Expected: build OK, `JS SYNTAX OK`. The existing reduced callers pass only `(rows, hasExact, log)`, so `opts={}` → defaults reproduce the prior labels/behavior exactly (starred titles, red-dot overlay, dashed n reference, no legend). (Manual smoke, not automated: the reduced app looks unchanged.)

- [ ] **Step 3: Commit**

```bash
git add web/index.html.in docs/index.html
git commit -m "$(cat <<'EOF'
web: parameterize renderPlotly with an opts arg (labels, 2nd curve, ref)

Pure refactor — defaults reproduce the reduced model-potential behavior
exactly. Adds an optional second line series and configurable axis
labels for the upcoming real-fluids mode.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Mode toggle + real-fluids controls + compute path

**Files:** Modify `web/index.html.in`.

- [ ] **Step 1: Add the mode toggle + real-fluids controls (HTML)**

(1a) Immediately after the `<h1>…</h1>` line, add the mode toggle:
```html
  <fieldset>
    <legend>mode</legend>
    <label>view
      <select id="mode">
        <option value="model">Model potentials (reduced)</option>
        <option value="real">Real fluids (SI)</option>
      </select>
    </label>
  </fieldset>
```

(1b) After the existing reduced `temperature range` fieldset (the one ending with `<button id="go" disabled>Compute B₂(T)</button>` then `</fieldset>`), add the real-fluids controls fieldset (hidden by default):
```html
  <fieldset id="realctl" hidden>
    <legend>real fluid (noble gas, ab initio Tang–Toennies)</legend>
    <div class="row">
      <label>gas
        <select id="gas">
          <option value="0">Neon</option>
          <option value="1" selected>Argon</option>
          <option value="2">Krypton</option>
          <option value="3">Xenon</option>
        </select>
      </label>
      <label>WK order
        <select id="wkorder">
          <option value="1">1</option>
          <option value="2">2</option>
          <option value="3" selected>3</option>
        </select>
      </label>
    </div>
    <div class="row">
      <label>Tmin (K) <input id="rtmin" type="number" value="50" min="1" step="10" /></label>
      <label>Tmax (K) <input id="rtmax" type="number" value="1000" step="50" /></label>
      <label>points <input id="rnpts" type="number" value="24" min="2" step="1" /></label>
      <label>spacing
        <select id="rtspace"><option value="log">log</option><option value="lin">linear</option></select>
      </label>
    </div>
    <div class="note">B₂ in cm³/mol vs T in K; <b>classical</b> vs
      <b>Wigner–Kirkwood</b> order-N quantum correction. Potentials: Ne (Bich, MP 2008),
      Ar (Vogel, MP 2010), Kr (Jäger, JCP 2016), Xe (Hellmann, JCP 2017).</div>
  </fieldset>
```

- [ ] **Step 2: Add element refs + `syncMode` + `noblegasRow` + the compute branch (JS)**

(2a) After the existing ref line `const lamwrap = $("lamwrap"), lamel = $("lam");`, add:
```javascript
const modeSel = $("mode");
const GAS_NAMES = ["Neon", "Argon", "Krypton", "Xenon"];
```

(2b) After `function syncPreset() { … }` (its closing `}`), add `syncMode` (shows/hides the two control blocks; the reduced blocks are the `potential` fieldset, the reduced `temperature range` fieldset, and the reduced `defs`):
```javascript
// Show the controls + definitions for the active mode; hide the other surface.
function syncMode() {
  const real = modeSel.value === "real";
  document.getElementById("modelctl").hidden = real;     // reduced potential fieldset
  document.getElementById("modeltrange").hidden = real;  // reduced T* range fieldset
  document.getElementById("modeldefs").hidden = real;    // reduced-unit defs
  $("realctl").hidden = !real;
  $("realdefs").hidden = !real;
}
```

(2c) Add `noblegasRow` (the wasm call) just before `function computeRow`:
```javascript
// Noble-gas classical + WK row via the poc_noblegas wasm export (cm³/mol, K).
function noblegasRow(gas, t, order) {
  const out = ex.poc_alloc(64); // 8 * f64
  ex.poc_noblegas(gas, t, order, out);
  const dv = new DataView(ex.memory.buffer);
  const r = Array.from({ length: 8 }, (_, k) => dv.getFloat64(out + 8 * k, true));
  ex.poc_dealloc(out, 64);
  // classical (num/db2/d2b2/neff) + WK curve carried in a parallel `q` field set.
  return { T: t, num: r[0], db2: r[1], d2b2: r[2], neff: r[3],
           q: { b2: r[4], db2: r[5], d2b2: r[6], neff: r[7] }, ex: null };
}
```

(2d) Rework `compute()` to branch on mode. Replace the current `async function compute() { … }` (whole function) with:
```javascript
async function compute() {
  const gen = ++runGen;
  const real = modeSel.value === "real";

  // T grid + per-row producer differ by mode; the slicing/cancel loop is shared.
  let Ts, makeRow, doneMsg;
  if (real) {
    const gas = parseInt($("gas").value) || 0, order = parseInt($("wkorder").value) || 3;
    let tmin = parseFloat($("rtmin").value), tmax = parseFloat($("rtmax").value);
    const np = Math.max(2, parseInt($("rnpts").value) || 2);
    const log = $("rtspace").value === "log";
    if (log && tmin <= 0) tmin = 1;
    Ts = tGrid(tmin, tmax, np, log);
    compute._logX = log;
    makeRow = (T) => noblegasRow(gas, T, order);
    doneMsg = `✓ ${GAS_NAMES[gas]} — ${np} temperatures (${log ? "log" : "linear"} spacing): ` +
      `classical and Wigner–Kirkwood order-${order} B₂(T), cm³/mol.`;
  } else {
    const preset = presetSel.value, eps = 1, sig = 1;
    let tmin = parseFloat($("tmin").value), tmax = parseFloat($("tmax").value);
    const np = Math.max(2, parseInt($("npts").value) || 2);
    const log = $("tspace").value === "log";
    if (log && tmin <= 0) tmin = 1e-3;
    let pot = null;
    if (preset !== "stockmayer" && preset !== "sw") {
      try { pot = makePotential(dslEl.value); }
      catch (e) { $("status").innerHTML = `<span class="err">✗ ${e.message}</span>`; return; }
    }
    Ts = tGrid(tmin, tmax, np, log);
    compute._logX = log;
    makeRow = (T) => computeRow(preset, pot, T, eps, sig);
    doneMsg = `✓ ${preset === "stockmayer" ? "Stockmayer 4-D cubature" : "compiled <code>" +
      dslEl.value.replace(/</g, "&lt;") + "</code>"} — ${np} temperatures ` +
      `(${log ? "log" : "linear"} spacing): B₂, its first two T-derivatives, and n_eff.`;
  }

  const rows = [];
  let i = 0;
  while (i < Ts.length) {
    if (gen !== runGen) return;
    const sliceStart = performance.now();
    try {
      do { rows.push(makeRow(Ts[i])); i++; }
      while (i < Ts.length && performance.now() - sliceStart < 16);
    } catch (e) {
      $("status").innerHTML = `<span class="err">✗ ${e.message}</span>`; return;
    }
    const hasExact = rows.some(r => r.ex);
    lastRows = rows.slice();
    lastHasExact = hasExact;
    lastReal = real;
    renderTable(rows, hasExact, real);
    renderPlotly(rows, hasExact, compute._logX, plotOpts(real));
    $("status").innerHTML = i < Ts.length
      ? `computing ${i}/${Ts.length}…`
      : doneMsg + (!real && !hasExact ? " <span class=note>(no closed-form overlay)</span>" : "");
    if (i < Ts.length) await new Promise(r => setTimeout(r, 0));
  }
}

// Plotly opts per mode: real = physical labels + a 2nd (WK) curve, no IPL ref line.
function plotOpts(real) {
  if (!real) return {}; // reduced defaults
  return {
    xlabel: "T (K)", hover: "T",
    panels: ["B₂ (cm³/mol)", "dB₂/dT (cm³/mol/K)", "d²B₂/dT² (cm³/mol/K²)"],
    series2: { key: "q", name: "Wigner–Kirkwood", color: "#1a9850", dash: "dot" },
    ref: false,
  };
}
```

(2e) Add a `lastReal` module var. Find the line `let lastRows = [], lastHasExact = false; // retained for "Copy table"` and change it to:
```javascript
let lastRows = [], lastHasExact = false, lastReal = false; // retained for "Copy table"
```

- [ ] **Step 3: Tag the reduced control blocks with ids + wire boot/handlers**

(3a) The three reduced blocks need ids so `syncMode` can hide them. In the HTML:
- the `potential` `<fieldset>` (the one containing `<select id="preset">`): add `id="modelctl"` → `<fieldset id="modelctl">`.
- the reduced `temperature range` `<fieldset>` (with `id="tmin"` etc.): add `id="modeltrange"`.
- the `<details class="defs">` reduced-unit definitions block: add `id="modeldefs"` → `<details class="defs" id="modeldefs">`.

(3b) Add a real-mode definitions block. Immediately after the reduced `<details class="defs" id="modeldefs">…</details>`, add:
```html
  <details class="defs" id="realdefs" hidden>
    <summary>real-fluid definitions</summary>
    <table class="defs">
      <tr><th>quantity</th><th>meaning</th></tr>
      <tr><td>B₂ (cm³/mol)</td><td>second virial coefficient in physical units</td></tr>
      <tr><td>T (K)</td><td>temperature</td></tr>
      <tr><td>classical</td><td>B₂ = −2π N_A ∫ (e<sup>−V/k_BT</sup>−1) r² dr</td></tr>
      <tr><td>Wigner–Kirkwood</td><td>+ ħ²-series quantum correction to order N (grows at low T; largest for Ne)</td></tr>
      <tr><td>n_eff</td><td>−3(B₂+T·B₂′)/(2T·B₂′+T²·B₂″)</td></tr>
    </table>
  </details>
```

(3c) In the boot IIFE, after the existing `for (const id of ["tmin", "tmax", "npts"]) $(id).onchange = compute;` line, add the real-mode wiring and the mode toggle:
```javascript
    $("mode").onchange = () => { syncMode(); compute(); };
    $("gas").onchange = () => { if (modeSel.value === "real") compute(); };
    $("wkorder").onchange = () => { if (modeSel.value === "real") compute(); };
    for (const id of ["rtmin", "rtmax", "rnpts"]) $(id).onchange = () => { if (modeSel.value === "real") compute(); };
    $("rtspace").onchange = () => { if (modeSel.value === "real") compute(); };
```
And after the existing `syncPreset();` call in boot, add `syncMode();`.

- [ ] **Step 4: Build + syntax check**

Run: `./web/build.sh`
```bash
python3 - <<'PY'
import re
html=open("web/index.html.in").read()
m=re.search(r'<script>\n(.*)</script>', html, re.S)
open('/tmp/webjs_t4.js','w').write(m.group(1))
PY
node --check /tmp/webjs_t4.js && echo "JS SYNTAX OK"
grep -c 'id="mode"\|id="realctl"\|noblegasRow\|poc_noblegas\|syncMode\|plotOpts' docs/index.html
```
Expected: build OK, `JS SYNTAX OK`, grep count ≥ 6.

`renderTable(rows, hasExact, real)` is called with a 3rd arg that the current `renderTable` ignores (Task 5 makes it mode-aware) — harmless; the table renders with reduced headers in real mode until Task 5. The plot is fully correct in real mode now.

- [ ] **Step 5: Commit**

```bash
git add web/index.html.in docs/index.html
git commit -m "$(cat <<'EOF'
web: add Real-fluids mode (noble-gas classical vs WK overlay)

Mode toggle (model reduced <-> real SI); gas/WK-order/T-range(K, lin/log)
controls; noblegasRow calls poc_noblegas; compute() branches on mode and
reuses the time-sliced loop; real mode plots classical + Wigner-Kirkwood
curves (cm^3/mol vs T(K), no IPL reference line) via the parameterized
renderPlotly. Table physical-fork follows.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Physical table + copy-table fork

**Files:** Modify `web/index.html.in`.

- [ ] **Step 1: Make `renderTable` mode-aware** — replace the current `function renderTable(rows, hasExact) { … }` with:

```javascript
function renderTable(rows, hasExact, real = false) {
  if (real) {
    let h = "<table><tr><th>T (K)</th><th>B₂ cl</th><th>B₂ WK</th>" +
            "<th>n_eff cl</th><th>n_eff WK</th></tr>";
    const fmt = (v) => (Number.isFinite(v) ? v.toPrecision(6) : "—");
    const f3 = (v) => (Number.isFinite(v) ? v.toFixed(3) : "—");
    for (const r of rows) {
      h += `<tr><td>${r.T.toFixed(1)}</td><td>${fmt(r.num)}</td><td>${fmt(r.q ? r.q.b2 : NaN)}</td>` +
           `<td>${f3(r.neff)}</td><td>${f3(r.q ? r.q.neff : NaN)}</td></tr>`;
    }
    $("tablewrap").innerHTML = h + "</table>" +
      "<div class=note>cl = classical, WK = Wigner–Kirkwood; B₂ in cm³/mol.</div>";
    return;
  }
  let h = "<table><tr><th>k_BT/ε</th><th>B₂/σ³</th><th>dB₂*/dT*</th><th>d²B₂*/dT*²</th>" +
          "<th>n_eff</th>" + (hasExact ? "<th>n_eff*</th>" : "") + "</tr>";
  for (const r of rows) {
    const ne = r.ex ? r.ex.neff : NaN;
    h += `<tr><td>${r.T.toFixed(3)}</td><td>${r.num.toFixed(4)}</td>` +
         `<td>${r.db2.toFixed(4)}</td><td>${r.d2b2.toFixed(4)}</td>` +
         `<td>${Number.isFinite(r.neff) ? r.neff.toFixed(3) : "—"}</td>`;
    if (hasExact)
      h += `<td>${Number.isFinite(ne) ? ne.toFixed(3) : "—"}</td>`;
    h += "</tr>";
  }
  $("tablewrap").innerHTML = h + "</table>" +
    (hasExact
      ? "<div class=note>n_eff* = exact. High-T limit is the repulsive exponent n; " +
        "for LJ/Mie n_eff → 0 as T → 0 (not m). Inverse-power is flat at n.</div>"
      : "");
}
```

- [ ] **Step 2: Make `copyTable` mode-aware** — replace the current `function copyTable() { … }` with:

```javascript
function copyTable() {
  if (!lastRows.length) return;
  let cols, rowf;
  if (lastReal) {
    cols = ["T_K", "B2_cl_cm3mol", "B2_WK_cm3mol", "dB2dT_cl", "dB2dT_WK", "neff_cl", "neff_WK"];
    rowf = (r) => [r.T, r.num, r.q ? r.q.b2 : "", r.db2, r.q ? r.q.db2 : "", r.neff, r.q ? r.q.neff : ""];
  } else {
    cols = ["kBT/eps", "B2/sig^3", "dB2*/dT*", "d2B2*/dT*^2", "n_eff"];
    if (lastHasExact) cols.push("n_eff_exact");
    rowf = (r) => {
      const row = [r.T, r.num, r.db2, r.d2b2, r.neff];
      if (lastHasExact) row.push(r.ex ? r.ex.neff : "");
      return row;
    };
  }
  const lines = [cols.join("\t"), ...lastRows.map(r => rowf(r).join("\t"))];
  const text = lines.join("\n");
  const btn = $("copy"), label = btn.textContent;
  const done = () => { btn.textContent = "copied ✓"; setTimeout(() => (btn.textContent = label), 1200); };
  if (navigator.clipboard && navigator.clipboard.writeText)
    navigator.clipboard.writeText(text).then(done, () => fallbackCopy(text, done));
  else fallbackCopy(text, done);
}
```

- [ ] **Step 3: Build + syntax check**

Run: `./web/build.sh`
```bash
python3 - <<'PY'
import re
html=open("web/index.html.in").read()
m=re.search(r'<script>\n(.*)</script>', html, re.S)
open('/tmp/webjs_t5.js','w').write(m.group(1))
PY
node --check /tmp/webjs_t5.js && echo "JS SYNTAX OK"
```
Expected: build OK, `JS SYNTAX OK`.

- [ ] **Step 4: Commit**

```bash
git add web/index.html.in docs/index.html
git commit -m "$(cat <<'EOF'
web: physical-units table + TSV copy for Real-fluids mode

renderTable/copyTable are mode-aware: real mode shows T(K), B2 cl/WK
(cm^3/mol, toPrecision), n_eff cl/WK; reduced mode unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Node end-to-end validation + full regression

**Files:** Create `node/noblegas-e2e.mjs`; full suite + builds.

- [ ] **Step 1: Confirm the wasm is current**

Run: `cargo build --release --target wasm32-unknown-unknown --lib`
(`./web/build.sh` from earlier tasks already did this; rebuild to be safe.)

- [ ] **Step 2: Create `node/noblegas-e2e.mjs`**

```javascript
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
```

- [ ] **Step 3: Run the e2e**

Run: `node node/noblegas-e2e.mjs`
Expected: each line `OK`, final `E2E PASS` (exit 0). This proves `poc_noblegas` round-trips through the wasm ABI: classical slot matches the `integrate_potentials.py` reference B₂, and the WK quantum slot differs at low T. If `E2E FAIL`, STOP and report the printed values.

- [ ] **Step 4: Full Rust regression**

Run: `cargo test`
Expected: all prior tests + the 2 new ones (`noblegas_grid_reuse_matches_b2_neff`, `noblegas_b2_neff_array_classical_and_quantum`) pass. (B3/CO2-QFH tests are slow — allow ~4 min.) Paste the final `test result:` line. If anything fails, STOP and report BLOCKED.

- [ ] **Step 5: Commit**

```bash
git add node/noblegas-e2e.mjs
git commit -m "$(cat <<'EOF'
Add node e2e for poc_noblegas (classical==ref, WK shift at low T)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review (completed against the spec)

**Spec coverage:**
- §2 Rust grid-reuse (`grid`, `b2_neff_with_grid`, `b2_neff` wrapper) → Task 1. ✓
- §3 `poc_noblegas` export (8 f64 classical+quantum) → Task 2. ✓
- §4.1 mode toggle + `syncMode` → Task 4 (Steps 1, 2b, 3). ✓
- §4.2 real controls (gas/order/T-range K, lin/log via `tGrid`) → Task 4 Step 1b. ✓
- §4.3 reactive handlers → Task 4 Step 3c. ✓
- §5 compute path (`noblegasRow`, `computeRow`/`compute` real branch, time-sliced reuse) → Task 4 Step 2. ✓
- §6 parameterized `renderPlotly` (second curve, physical labels, no ref) → Task 3 (param) + Task 4 `plotOpts`. ✓
- §7 physical table/copy/defs fork → Task 5 + Task 4 Step 3b (real defs block). ✓
- §8 validation (Rust grid-reuse equiv, node e2e, build/syntax) → Task 1, Task 6, Tasks 3-5 build checks. ✓
- §9 out of scope (reference-data overlay, He, QFH, no unit mixing) → not in any task. ✓

**Placeholder scan:** none — every step has complete code and exact commands; the e2e reference numbers are the validated noble-gas test values.

**Type consistency:** `grid()->Vec<[f64;5]>` and `b2_neff_with_grid(t,order,&pv)->(f64,f64,f64,f64)` consistent across Tasks 1-2. `noblegas_b2_neff(gas:u32,t:f64,order:u8)->[f64;8]` (Task 2) and `poc_noblegas(gas:u32,t:f64,order:u32,out)` consistent. Web: the row shape `{T, num, db2, d2b2, neff, q:{b2,db2,d2b2,neff}, ex}` is produced by `noblegasRow` (Task 4) and consumed by `renderPlotly` `series2.key="q"` (Task 3/4), `renderTable` real path (Task 5: `r.q.b2`, `r.q.neff`), and `copyTable` (Task 5: `r.q.*`) — all use `.q.b2/.db2/.d2b2/.neff`. `plotOpts(real)`/`renderTable(rows,hasExact,real)`/`lastReal` consistent across Tasks 4-5. `modeSel`/`syncMode`/ids (`modelctl`/`modeltrange`/`modeldefs`/`realctl`/`realdefs`) consistent.
