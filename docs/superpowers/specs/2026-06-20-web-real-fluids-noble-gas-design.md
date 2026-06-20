# Web "Real fluids" mode — noble-gas B₂(T) with quantum corrections — design

- **Date:** 2026-06-20
- **Status:** Approved (ready for implementation plan)
- **Context:** The browser app (`web/index.html.in`) is a companion to Bell, *J. Chem. Phys.* 152, 164508 (2020). It currently plots `n_eff` and B₂ for **reduced-unit model potentials** (LJ/Mie/Stockmayer/EXP/square-well/inverse-power) on starred dimensionless axes (`B₂/σ³` vs `k_BT/ε`). This feature adds a second, **physical-units** surface: the **noble gases Ne/Ar/Kr/Xe** computed from the Rust `noblegas` module (Tang–Toennies ab initio potentials + Wigner–Kirkwood quantum corrections), in **cm³/mol vs T (K)**, shown as a **classical-vs-quantum overlay**. Reduced model potentials and real fluids are kept as **deliberately separate surfaces** (units never mixed).

## 1. Goal

A top-level **mode toggle** — *Model potentials (reduced)* ⇄ *Real fluids (SI)* — where the real-fluids mode lets the user pick a noble gas, a Wigner–Kirkwood order, and a T range, and plots **classical** and **WK-order-N** B₂(T), its first two T-derivatives, and `n_eff` in physical units, with the quantum correction visible as the gap between the two curves (widening at low T, largest for Ne).

## 2. Rust additions (`src/noblegas.rs`)

The existing `TangToennies` has `b2(t,order)` and `b2_neff(t,order)`, each of which rebuilds the (T-independent) 10000-point derivative grid internally. To compute **both** the classical (order 0) and WK-order-N curves at one temperature off **one** grid build, expose the grid-reuse the final review flagged:

```rust
impl TangToennies {
    /// The T-independent precomputed grid (R, V, V', V'', V''' in SI) — was private.
    pub fn grid(&self) -> Vec<[f64; 5]> { self.grid_potvals() }

    /// B₂ [cm³/mol], dB₂/dT, d²B₂/dT², n_eff at temperature `t` [K], WK truncation
    /// `order`, reusing a prebuilt grid (avoids rebuilding per call).
    pub fn b2_neff_with_grid(&self, t: f64, order: u8, pv: &[[f64; 5]]) -> (f64, f64, f64, f64) {
        let (b2, db2, d2b2) =
            num_dual::second_derivative(|tt| self.b2_generic(tt, order, pv), t);
        let neff = -3.0 * (b2 + t * db2) / (2.0 * t * db2 + t * t * d2b2);
        (b2, db2, d2b2, neff)
    }
}
```
`b2_neff` becomes a thin wrapper: `let pv = self.grid_potvals(); self.b2_neff_with_grid(t, order, &pv)`. (`grid_potvals`, `b2_generic` stay private; `grid` is the public accessor.) No behavior change — existing tests still pass.

Performance note (measured, native release): one `b2_neff` ≈ 2.6 ms incl. the 10000-pt grid build; a full 24-T × (classical+quantum) sweep ≈ 125 ms native, so ~0.3–0.5 s in wasm — fast enough that the existing time-sliced compute loop never janks (no batch export needed).

## 3. WASM export (`src/lib.rs`)

One **per-temperature** export, in the `wasm_exports` module, following the `poc_stockmayer` pattern (out-pointer + `write_unaligned`):

```rust
/// Noble-gas B₂ in physical units; writes 8 f64 into `out`:
///   [B2_cl, dB2dT_cl, d2B2dT2_cl, neff_cl,  B2_q, dB2dT_q, d2B2dT2_q, neff_q]
/// (cm³/mol and K). `gas`: 0=Ne 1=Ar 2=Kr 3=Xe. `order`: WK truncation for the
/// quantum curve (1..3); the classical curve is always order 0.
#[no_mangle]
pub extern "C" fn poc_noblegas(gas: u32, t: f64, order: u32, out: *mut f64) {
    let g = match gas { 0 => neon_tt(), 1 => argon_tt(), 2 => krypton_tt(), _ => xenon_tt() };
    let pv = g.grid();
    let (b0, d0, e0, n0) = g.b2_neff_with_grid(t, 0, &pv);
    let (bq, dq, eq, nq) = g.b2_neff_with_grid(t, order as u8, &pv);
    let vals = [b0, d0, e0, n0, bq, dq, eq, nq];
    unsafe { for (k, v) in vals.iter().enumerate() { out.add(k).write_unaligned(*v); } }
}
```
(Add `neon_tt, argon_tt, krypton_tt, xenon_tt` to the `wasm_exports` `use super::{…}` line. The grid is built once per call and serves both curves.)

## 4. Web — mode toggle & controls (`web/index.html.in`)

### 4.1 Mode toggle
A `<select id="mode">` (or radio) at the very top: **`model`** (default) | **`real`**. A `syncMode()` function shows/hides two control blocks and the two intro/definitions blocks, and points `compute()` at the right path. The plot host (`#plots`), the time-sliced `compute()` driver, `tGrid`, `renderPlotly`, `renderTable`, `copyTable`, and the boot wiring are **shared**; only labels, the per-row compute, and the table/defs text differ by mode.

### 4.2 Real-fluids controls (a fieldset shown only in `real` mode)
- Gas `<select id="gas">`: Ne / Ar / Kr / Xe (value = the `gas` id 0..3).
- WK order `<select id="wkorder">`: 1 / 2 / 3 (default 3).
- T-range fieldset for real mode: `Tmin (K)` (default 50), `Tmax (K)` (default 1000), `points` (default 24), `spacing` (lin/log, **default log**). **Reuse `tGrid(tmin,tmax,np,log)`** — so log/linear spacing is supported identically to the reduced app.

The existing reduced controls (preset dropdown, n/m/μ²/λ inputs, the `T*` range fieldset) live in a block shown only in `model` mode.

### 4.3 Reactive handlers
`gas`, `wkorder`, and the real T-range inputs each `onchange = () => { if (mode==="real") compute(); }`; the mode toggle calls `syncMode()` then `compute()`.

## 5. Web — compute path

`compute()` is unchanged except `computeRow` gains a `real`/noble branch. A new `noblegasRow(gas, t, order)` follows the `stockmayerReduced` wasm pattern:

```js
function noblegasRow(gas, t, order) {
  const out = ex.poc_alloc(64); // 8 * f64
  ex.poc_noblegas(gas, t, order, out);
  const dv = new DataView(ex.memory.buffer);
  const r = Array.from({length: 8}, (_, k) => dv.getFloat64(out + 8 * k, true));
  ex.poc_dealloc(out, 64);
  // classical (num/db2/d2b2/neff) + quantum carried in a parallel field set
  return { T: t,
           num: r[0], db2: r[1], d2b2: r[2], neff: r[3],   // classical (blue)
           q: { b2: r[4], db2: r[5], d2b2: r[6], neff: r[7] }, // WK order-N (green)
           ex: null };
}
```
In `compute()`, when mode is `real`, read `gas = +$("gas").value`, `order = +$("wkorder").value`, build the T grid from the real T-range inputs, and push `noblegasRow(gas, T, order)` per temperature. The row shape stays `{T, num, db2, d2b2, neff, …}` plus a `q` sub-object for the second curve (mirrors how `ex` carries the overlay today, but `q` is a real second *line*, not dots).

## 6. Web — plot (`renderPlotly` parameterized)

Generalize `renderPlotly(rows, hasExact, logX)` to accept a small `opts` describing labels and the second series, **without breaking the reduced callers**:
- Add an `opts` param (default = the current reduced behavior): `{ xlabel, panelLabels:[B₂,dB₂,d²B₂,neff], series2: null|{key, name, color}, ref: bool }`.
- x-axis title, the four y-titles, and the hovertemplate `k_BT/ε` string come from `opts` (default to the current starred strings).
- When `opts.series2` is set (real mode: `{key:"q", name:"Wigner–Kirkwood", color:GREEN}`), push four more line traces reading `r.q.b2 / r.q.db2 / r.q.d2b2 / r.q.neff` onto y4/y3/y2/y, dashed green, and set `showlegend:true` with `name:"classical"` on the blue series. (Reduced mode keeps the red-dot exact overlay via the existing `hasExact` path; real mode passes `hasExact=false`.)
- Real mode passes `ref:false` (no dashed IPL reference — TT repulsion is exponential, so `n_eff` has no fixed high-T limit). Reduced mode keeps `repExponent()`.

Real-mode labels: x = `T (K)`, panels `B₂ (cm³/mol)`, `dB₂/dT (cm³/mol/K)`, `d²B₂/dT² (cm³/mol/K²)`, `n_eff`.

## 7. Web — table, copy, definitions (physical fork)

- **`renderTable`** gains a mode-aware header/format path. Real mode: columns `T (K)`, `B₂ᶜˡ`, `B₂ᵂᴷ`, `dB₂/dTᶜˡ`, `dB₂/dTᵂᴷ`, `n_effᶜˡ`, `n_effᵂᴷ` (or a compact subset: `T, B₂ cl, B₂ WK, n_eff cl, n_eff WK`). Use `toPrecision(6)` / exponential for B₂ (spans ~10⁰–10⁴ cm³/mol), `toFixed(3)` for `n_eff`. Reduced mode unchanged.
- **`copyTable`** TSV mirrors the active mode's columns (physical headers `T_K, B2_cl_cm3mol, B2_WK_cm3mol, neff_cl, neff_WK, …`, full precision).
- **Definitions/intro:** a real-mode `<p class="sub">` + `<details>` block: B₂ in cm³/mol, T in K; classical vs **Wigner–Kirkwood order-N** quantum correction; **Tang–Toennies ab initio potentials** with the literature references (Ne: Bich MP 2008; Ar: Vogel MP 2010; Kr: Jäger JCP 2016; Xe: Hellmann JCP 2017 — from `integrate_potentials.py`). Shown only in real mode; the reduced-units block shown only in model mode.

## 8. Validation

1. **Rust** (`src/main.rs`): `b2_neff_with_grid(t, order, &g.grid())` equals the existing `b2_neff(t, order)` for Argon at a couple of T/orders (to 1e-12) — proves the grid-reuse refactor is behavior-preserving. (Existing 36 tests must still pass.)
2. **Node e2e**: instantiate the built wasm, call `poc_noblegas(gas, t, order, out)` for Ne and Ar at 50/300 K, order 3; confirm the 8 returned values match the Rust `b2_neff` reference (classical slot == order-0 B₂/neff; quantum slot == order-3), and that classical ≠ quantum at 50 K. (Mirrors the existing `node/stockmayer-e2e.mjs`.)
3. **Build/syntax**: `./web/build.sh` regenerates `docs/index.html`; `node --check` the embedded JS; confirm `cargo build --lib --target wasm32-unknown-unknown` still compiles (num-dual is pure Rust).
4. **Manual smoke** (noted, not automated): toggle to Real fluids, pick Ar, see classical & WK-3 curves with the low-T gap; switch gas/order/spacing recomputes; toggle back to Model potentials and confirm the reduced app is unchanged.

## 9. Out of scope (v1)

- **Experimental / reference virial-data overlay** (needs bundled data files; the two computed curves are the v1 story).
- **Helium** (deferred phase-shift / Beth–Uhlenbeck route; WK diverges for He).
- **QFH** variant.
- Mixing physical and reduced units on one plot (the surfaces stay separate by design).

## 10. File-by-file

- `src/noblegas.rs` — `pub fn grid`, `pub fn b2_neff_with_grid`; `b2_neff` rewritten as a wrapper.
- `src/lib.rs` — `poc_noblegas` wasm export (+ `use super::{neon_tt,…}` in `wasm_exports`).
- `web/index.html.in` — mode toggle + `syncMode`; real-fluids controls; `noblegasRow`; `computeRow`/`compute` real branch; `renderPlotly` `opts` parameterization + second curve; `renderTable`/`copyTable` physical fork; real-mode intro/defs; boot wiring.
- `web/build.sh` regenerates `docs/index.html`.
- `src/main.rs` — the grid-reuse equivalence test.
- A node e2e script (e.g. `node/noblegas-e2e.mjs`) for §8.2.
