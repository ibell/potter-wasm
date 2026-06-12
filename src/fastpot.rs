//! fasteval-backed pair potential — the "use a real expression-eval crate instead
//! of the hand-rolled DSL" option. Same `.v(r)` interface as `Potential`, so it
//! drops straight into `b2_v` / `b3_v`.
//!
//! fasteval parses once into a `Slab` and compiles to an `Instruction` (indices
//! into the slab), then evaluates many times — the exprtk "compile once, eval
//! many" model. Note fasteval uses `^` for exponentiation, so we translate the
//! Python-like `**` from our DSL strings to `^` before parsing.

use fasteval::{Compiler, Evaler};

pub struct FastPotential {
    slab: fasteval::Slab,
    instr: fasteval::Instruction,
    eps: f64,
    sig: f64,
}

impl FastPotential {
    pub fn compile(src: &str, eps: f64, sig: f64) -> Result<Self, String> {
        let translated = src.replace("**", "^"); // Python ** -> fasteval ^
        let parser = fasteval::Parser::new();
        let mut slab = fasteval::Slab::new();
        let expr_i = parser
            .parse(&translated, &mut slab.ps)
            .map_err(|e| format!("fasteval parse error: {e}"))?;
        let instr = expr_i.from(&slab.ps).compile(&slab.ps, &mut slab.cs);
        Ok(FastPotential {
            slab,
            instr,
            eps,
            sig,
        })
    }

    #[inline]
    pub fn v(&self, r: f64) -> f64 {
        let (eps, sig) = (self.eps, self.sig);
        // Variable lookups pass an empty args Vec (no allocation); built-in
        // functions like exp() are handled inside fasteval, not via this callback.
        let mut ns = |name: &str, _args: Vec<f64>| match name {
            "r" => Some(r),
            "eps" => Some(eps),
            "sig" => Some(sig),
            _ => None,
        };
        self.instr.eval(&self.slab, &mut ns).unwrap_or(f64::NAN)
    }
}
