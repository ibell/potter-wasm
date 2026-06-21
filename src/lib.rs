//! potter_poc — proof of concept that a DSL-defined pair potential can be
//! integrated to virial coefficients (B2, B3) entirely in (pure) Rust, and
//! shipped as a self-contained WASM module that a host (node.js) calls into.

pub mod aot;
pub mod cubature;
pub mod dsl;
pub mod fastpot;
pub mod he_potential;
pub mod integrate;
pub mod molecule;
pub mod msmc;
pub mod noblegas;
pub use noblegas::{argon_tt, krypton_tt, neon_tt, xenon_tt, TangToennies};
pub mod physics;

// cranelift emits native machine code, so the JIT backend is native-only.
#[cfg(not(target_arch = "wasm32"))]
pub mod jit;
#[cfg(not(target_arch = "wasm32"))]
pub use jit::JitPotential;

pub use fastpot::FastPotential;
pub use physics::{
    b2, b2_and_derivs, b2_and_derivs_v, b2_finegrid, b2_lj_series, b2_lj_series_derivs, b2_v,
    b2_v_grid, b3, b3_cubature, b3_cubature_v, b3_v, b3_v_grid, neff, B2Derivs, CsePotential,
    Potential, LJ_BOYLE_TSTAR,
};

/// Compile a DSL potential string and compute B2 at temperature `t`.
pub fn b2_from_dsl(src: &str, eps: f64, sig: f64, t: f64, tol: f64) -> Result<f64, String> {
    let pot = Potential::compile(src, eps, sig)?;
    Ok(b2(&pot, t, tol))
}

/// Compile a DSL potential string and compute B₂ with its first two T-derivatives.
pub fn b2_derivs_from_dsl(
    src: &str,
    eps: f64,
    sig: f64,
    t: f64,
    tol: f64,
) -> Result<B2Derivs, String> {
    let pot = Potential::compile(src, eps, sig)?;
    Ok(b2_and_derivs(&pot, t, tol))
}

/// Reduced Stockmayer (ε=σ=1) B₂ and its first two T*-derivatives at reduced
/// temperature `tstar` and dipole strength `mu2 = (μ*)²`.
pub fn stockmayer_b2_derivs(tstar: f64, mu2: f64, reltol: f64) -> B2Derivs {
    let sm = crate::molecule::Stockmayer { eps: 1.0, sig: 1.0, mu2 };
    sm.b2_and_derivs(tstar, reltol).0
}

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

/// Molecular B₂ in physical units (cm³/mol, K): classical + (for the Hellmann ab
/// initio models) the QFH quantum correction. `mol`: 0=TraPPE N₂, 1=EPM2 CO₂,
/// 2=Hellmann N₂, 3=Hellmann CO₂. Returns
/// `[b2_cl, db2_cl, d2b2_cl, neff_cl,  b2_q, db2_q, d2b2_q, neff_q]`; the quantum
/// slots are NaN for the empirical (classical-only) models. μ/I per Hellmann SI.
pub fn molecule_b2_neff(mol: u32, t: f64, reltol: f64) -> [f64; 8] {
    use crate::molecule::{co2_epm2, co2_hellmann, n2_hellmann, n2_trappe};
    let q4 = |d: B2Derivs| [d.b2, d.db2_dt, d.d2b2_dt2, d.neff(t)];
    let (cl, qu): ([f64; 4], [f64; 4]) = match mol {
        0 => (q4(n2_trappe().b2_and_derivs(t, reltol).0), [f64::NAN; 4]),
        1 => (q4(co2_epm2().b2_and_derivs(t, reltol).0), [f64::NAN; 4]),
        2 => {
            let m = n2_hellmann();
            (
                q4(m.b2_and_derivs(t, reltol).0),
                q4(m.b2_qfh_and_derivs(t, reltol, 14.0067, 8.473).0),
            )
        }
        _ => {
            let m = co2_hellmann();
            (
                q4(m.b2_and_derivs(t, reltol).0),
                q4(m.b2_qfh_and_derivs(t, reltol, 22.0045, 43.202).0),
            )
        }
    };
    [cl[0], cl[1], cl[2], cl[3], qu[0], qu[1], qu[2], qu[3]]
}

/// Compile a DSL potential string and compute B3 at temperature `t`.
pub fn b3_from_dsl(src: &str, eps: f64, sig: f64, t: f64, tol: f64) -> Result<f64, String> {
    let pot = Potential::compile(src, eps, sig)?;
    Ok(b3(&pot, t, tol))
}

/// Raw C-ABI exports for the WASM "plugin". The host writes the DSL string into
/// linear memory via `poc_alloc`, then calls `poc_b2` / `poc_b3`.
#[cfg(target_arch = "wasm32")]
mod wasm_exports {
    use super::{
        b2_derivs_from_dsl, b2_from_dsl, b3_from_dsl, molecule_b2_neff, noblegas_b2_neff,
        stockmayer_b2_derivs,
    };
    use std::alloc::{alloc, dealloc, Layout};

    #[no_mangle]
    pub extern "C" fn poc_alloc(len: usize) -> *mut u8 {
        if len == 0 {
            return std::ptr::null_mut();
        }
        let layout = Layout::from_size_align(len, 1).unwrap();
        unsafe { alloc(layout) }
    }

    #[no_mangle]
    pub extern "C" fn poc_dealloc(ptr: *mut u8, len: usize) {
        if ptr.is_null() || len == 0 {
            return;
        }
        let layout = Layout::from_size_align(len, 1).unwrap();
        unsafe { dealloc(ptr, layout) }
    }

    fn read_dsl<'a>(ptr: *const u8, len: usize) -> Option<&'a str> {
        let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
        std::str::from_utf8(slice).ok()
    }

    /// Parse the DSL string at [ptr, ptr+len) and return B2(eps, sig, t). NaN on error.
    #[no_mangle]
    pub extern "C" fn poc_b2(ptr: *const u8, len: usize, eps: f64, sig: f64, t: f64) -> f64 {
        match read_dsl(ptr, len) {
            Some(src) => b2_from_dsl(src, eps, sig, t, 1e-12).unwrap_or(f64::NAN),
            None => f64::NAN,
        }
    }

    /// Parse the DSL string at [ptr, ptr+len) and return B3(eps, sig, t). NaN on error.
    #[no_mangle]
    pub extern "C" fn poc_b3(ptr: *const u8, len: usize, eps: f64, sig: f64, t: f64) -> f64 {
        match read_dsl(ptr, len) {
            Some(src) => b3_from_dsl(src, eps, sig, t, 1e-7).unwrap_or(f64::NAN),
            None => f64::NAN,
        }
    }

    /// Parse the DSL at [ptr,len) and write `[B2, dB2/dT, d2B2/dT2, n_eff]` (4 f64)
    /// into the caller-provided `out` array. All NaN on parse/eval error. One
    /// integration pass — avoids recomputing B2 three times.
    #[no_mangle]
    pub extern "C" fn poc_b2_derivs(
        ptr: *const u8,
        len: usize,
        eps: f64,
        sig: f64,
        t: f64,
        out: *mut f64,
    ) {
        let vals = match read_dsl(ptr, len) {
            Some(src) => match b2_derivs_from_dsl(src, eps, sig, t, 1e-12) {
                Ok(d) => [d.b2, d.db2_dt, d.d2b2_dt2, d.neff(t)],
                Err(_) => [f64::NAN; 4],
            },
            None => [f64::NAN; 4],
        };
        unsafe {
            for (k, v) in vals.iter().enumerate() {
                *out.add(k) = *v;
            }
        }
    }

    /// Reduced Stockmayer: write [B2*, dB2*/dT*, d2B2*/dT*2, n_eff] (4 f64) into the
    /// caller `out` array. Uses unaligned writes so `out` need not be 8-byte aligned.
    #[no_mangle]
    pub extern "C" fn poc_stockmayer(tstar: f64, mu2: f64, reltol: f64, out: *mut f64) {
        let d = stockmayer_b2_derivs(tstar, mu2, reltol);
        let vals = [d.b2, d.db2_dt, d.d2b2_dt2, d.neff(tstar)];
        unsafe {
            for (k, v) in vals.iter().enumerate() {
                out.add(k).write_unaligned(*v);
            }
        }
    }

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

    /// Molecular B₂ (cm³/mol, K): classical + QFH quantum (Hellmann models). Writes
    /// 8 f64 into `out`: [b2_cl,db2_cl,d2b2_cl,neff_cl, b2_q,db2_q,d2b2_q,neff_q]
    /// (quantum slots NaN for the empirical classical-only models). `mol`: 0=TraPPE
    /// N₂, 1=EPM2 CO₂, 2=Hellmann N₂, 3=Hellmann CO₂. Unaligned writes.
    #[no_mangle]
    pub extern "C" fn poc_molecule(mol: u32, t: f64, reltol: f64, out: *mut f64) {
        let vals = molecule_b2_neff(mol, t, reltol);
        unsafe {
            for (k, v) in vals.iter().enumerate() {
                out.add(k).write_unaligned(*v);
            }
        }
    }

    /// Parse the DSL at [ptr,len) and return the effective repulsive exponent
    /// n_eff(T). NaN on error.
    #[no_mangle]
    pub extern "C" fn poc_neff(ptr: *const u8, len: usize, eps: f64, sig: f64, t: f64) -> f64 {
        match read_dsl(ptr, len) {
            Some(src) => b2_derivs_from_dsl(src, eps, sig, t, 1e-12)
                .map(|d| d.neff(t))
                .unwrap_or(f64::NAN),
            None => f64::NAN,
        }
    }

    /// AOT-compile the DSL at [ptr,len) to a standalone wasm module exporting
    /// `potential(r,eps,sig)`. Returns the generated bytes packed as
    /// `(ptr << 32) | len` (0 on parse error). Free with `poc_free`.
    #[no_mangle]
    pub extern "C" fn poc_compile_wasm(ptr: *const u8, len: usize) -> u64 {
        let src = match read_dsl(ptr, len) {
            Some(s) => s,
            None => return 0,
        };
        match crate::aot::compile_to_wasm(src, &["r", "eps", "sig"]) {
            Ok(bytes) => {
                let n = bytes.len() as u64;
                let boxed = bytes.into_boxed_slice();
                let p = boxed.as_ptr() as u64;
                std::mem::forget(boxed); // ownership handed to caller; freed via poc_free
                (p << 32) | n
            }
            Err(_) => 0,
        }
    }

    /// Free a buffer returned by `poc_compile_wasm`.
    #[no_mangle]
    pub extern "C" fn poc_free(ptr: *mut u8, len: usize) {
        if !ptr.is_null() && len > 0 {
            unsafe {
                drop(Vec::from_raw_parts(ptr, len, len));
            }
        }
    }

    /// Exact LJ (12-6) reduced B2 via the HCB Gamma series (for the exact-solution
    /// overlay). `tstar` = T/eps; multiply the result by sig^3 for general sigma.
    #[no_mangle]
    pub extern "C" fn poc_b2_lj_series(tstar: f64, nterms: u32) -> f64 {
        crate::physics::b2_lj_series(tstar, nterms as usize)
    }

    /// Gamma function (for the inverse-power exact solution: B2 = (2pi/3) sig^3
    /// Gamma(1-3/n) (T/eps)^(-3/n)).
    #[no_mangle]
    pub extern "C" fn poc_tgamma(x: f64) -> f64 {
        libm::tgamma(x)
    }
}
