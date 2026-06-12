//! Cranelift JIT backend: compile a DSL potential to native machine code at
//! runtime, so the hot loop runs as a real `extern "C" fn(f64,f64,f64) -> f64`
//! with no interpreter dispatch — the route to hard-coded parity.
//!
//! Native-only: cranelift emits machine code for the host, so this is gated out
//! of the wasm builds (where the interpreter/CSE backend is used instead). For an
//! in-browser story you'd instead AOT-compile the expression to a wasm function.
//!
//! Arithmetic, integer powers (`IPow` -> multiplication chains), `sqrt` and `abs`
//! lower to native cranelift instructions; transcendentals (`exp`, `ln`, `sin`,
//! …) and general `pow` call out to small `extern "C"` libm shims registered with
//! the JIT. The Lennard-Jones potential uses none of the latter — it is pure
//! arithmetic + integer powers, so its JIT'd code makes zero external calls.

use crate::dsl::{self, Expr, Func};
use cranelift::codegen::ir::{FuncRef, UserFuncName};
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};

// libm shims, registered as JIT symbols and called from generated code.
extern "C" fn s_exp(x: f64) -> f64 {
    x.exp()
}
extern "C" fn s_ln(x: f64) -> f64 {
    x.ln()
}
extern "C" fn s_log10(x: f64) -> f64 {
    x.log10()
}
extern "C" fn s_sin(x: f64) -> f64 {
    x.sin()
}
extern "C" fn s_cos(x: f64) -> f64 {
    x.cos()
}
extern "C" fn s_tan(x: f64) -> f64 {
    x.tan()
}
extern "C" fn s_powf(b: f64, e: f64) -> f64 {
    b.powf(e)
}

struct Funcs {
    exp: FuncRef,
    ln: FuncRef,
    log10: FuncRef,
    sin: FuncRef,
    cos: FuncRef,
    tan: FuncRef,
    powf: FuncRef,
}

/// A potential JIT-compiled to native code.
pub struct JitPotential {
    _module: JITModule, // owns the executable code; must outlive `func`
    func: extern "C" fn(f64, f64, f64) -> f64,
    eps: f64,
    sig: f64,
}

impl JitPotential {
    pub fn compile(src: &str, eps: f64, sig: f64) -> Result<Self, String> {
        // Optimized tree (integer powers already folded to IPow).
        let expr = dsl::compile(src, &["r", "eps", "sig"])?;

        let mut flags = settings::builder();
        let _ = flags.set("use_colocated_libcalls", "false");
        let isa_builder = cranelift_native::builder().map_err(|e| e.to_string())?;
        let isa = isa_builder
            .finish(settings::Flags::new(flags))
            .map_err(|e| e.to_string())?;

        let mut jb = JITBuilder::with_isa(isa, default_libcall_names());
        jb.symbol("s_exp", s_exp as *const u8);
        jb.symbol("s_ln", s_ln as *const u8);
        jb.symbol("s_log10", s_log10 as *const u8);
        jb.symbol("s_sin", s_sin as *const u8);
        jb.symbol("s_cos", s_cos as *const u8);
        jb.symbol("s_tan", s_tan as *const u8);
        jb.symbol("s_powf", s_powf as *const u8);
        let mut module = JITModule::new(jb);

        // signatures for the libm shims
        let mut sig1 = module.make_signature();
        sig1.params.push(AbiParam::new(types::F64));
        sig1.returns.push(AbiParam::new(types::F64));
        let mut sig2 = module.make_signature();
        sig2.params.push(AbiParam::new(types::F64));
        sig2.params.push(AbiParam::new(types::F64));
        sig2.returns.push(AbiParam::new(types::F64));

        let id_exp = module.declare_function("s_exp", Linkage::Import, &sig1).map_err(|e| e.to_string())?;
        let id_ln = module.declare_function("s_ln", Linkage::Import, &sig1).map_err(|e| e.to_string())?;
        let id_log10 = module.declare_function("s_log10", Linkage::Import, &sig1).map_err(|e| e.to_string())?;
        let id_sin = module.declare_function("s_sin", Linkage::Import, &sig1).map_err(|e| e.to_string())?;
        let id_cos = module.declare_function("s_cos", Linkage::Import, &sig1).map_err(|e| e.to_string())?;
        let id_tan = module.declare_function("s_tan", Linkage::Import, &sig1).map_err(|e| e.to_string())?;
        let id_powf = module.declare_function("s_powf", Linkage::Import, &sig2).map_err(|e| e.to_string())?;

        // main function: fn(r, eps, sig) -> f64
        let mut main_sig = module.make_signature();
        for _ in 0..3 {
            main_sig.params.push(AbiParam::new(types::F64));
        }
        main_sig.returns.push(AbiParam::new(types::F64));
        let main_id = module
            .declare_function("potential", Linkage::Export, &main_sig)
            .map_err(|e| e.to_string())?;

        let mut ctx = module.make_context();
        ctx.func.signature = main_sig;
        ctx.func.name = UserFuncName::user(0, main_id.as_u32());
        let mut fctx = FunctionBuilderContext::new();
        {
            let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
            let blk = b.create_block();
            b.switch_to_block(blk);
            b.append_block_params_for_function_params(blk);
            let args = [
                b.block_params(blk)[0],
                b.block_params(blk)[1],
                b.block_params(blk)[2],
            ];
            let fr = Funcs {
                exp: module.declare_func_in_func(id_exp, &mut b.func),
                ln: module.declare_func_in_func(id_ln, &mut b.func),
                log10: module.declare_func_in_func(id_log10, &mut b.func),
                sin: module.declare_func_in_func(id_sin, &mut b.func),
                cos: module.declare_func_in_func(id_cos, &mut b.func),
                tan: module.declare_func_in_func(id_tan, &mut b.func),
                powf: module.declare_func_in_func(id_powf, &mut b.func),
            };
            let result = emit(&mut b, &expr, &args, &fr);
            b.ins().return_(&[result]);
            b.seal_all_blocks();
            b.finalize();
        }
        module.define_function(main_id, &mut ctx).map_err(|e| e.to_string())?;
        module.clear_context(&mut ctx);
        module.finalize_definitions().map_err(|e| e.to_string())?;

        let code = module.get_finalized_function(main_id);
        // SAFETY: `code` points to a finalized function with the declared C ABI;
        // `_module` keeps the backing memory alive for as long as `func` is used.
        let func = unsafe {
            std::mem::transmute::<*const u8, extern "C" fn(f64, f64, f64) -> f64>(code)
        };

        Ok(JitPotential {
            _module: module,
            func,
            eps,
            sig,
        })
    }

    #[inline]
    pub fn v(&self, r: f64) -> f64 {
        (self.func)(r, self.eps, self.sig)
    }
}

fn call1(b: &mut FunctionBuilder, f: FuncRef, x: Value) -> Value {
    let c = b.ins().call(f, &[x]);
    b.inst_results(c)[0]
}

fn call2(b: &mut FunctionBuilder, f: FuncRef, x: Value, y: Value) -> Value {
    let c = b.ins().call(f, &[x, y]);
    b.inst_results(c)[0]
}

/// Integer power via exponentiation-by-squaring in IR (matches `.powi`).
fn emit_ipow(b: &mut FunctionBuilder, base: Value, n: i32) -> Value {
    if n == 0 {
        return b.ins().f64const(1.0);
    }
    let neg = n < 0;
    let mut e = n.unsigned_abs();
    let mut acc = base;
    let mut result: Option<Value> = None;
    while e > 0 {
        if e & 1 == 1 {
            result = Some(match result {
                Some(r) => b.ins().fmul(r, acc),
                None => acc,
            });
        }
        e >>= 1;
        if e > 0 {
            acc = b.ins().fmul(acc, acc);
        }
    }
    let r = result.unwrap();
    if neg {
        let one = b.ins().f64const(1.0);
        b.ins().fdiv(one, r)
    } else {
        r
    }
}

fn emit(b: &mut FunctionBuilder, e: &Expr, args: &[Value; 3], fr: &Funcs) -> Value {
    match e {
        Expr::Num(v) => b.ins().f64const(*v),
        Expr::Var(i) => args[*i],
        Expr::Neg(a) => {
            let x = emit(b, a, args, fr);
            b.ins().fneg(x)
        }
        Expr::Add(a, c) => {
            let (x, y) = (emit(b, a, args, fr), emit(b, c, args, fr));
            b.ins().fadd(x, y)
        }
        Expr::Sub(a, c) => {
            let (x, y) = (emit(b, a, args, fr), emit(b, c, args, fr));
            b.ins().fsub(x, y)
        }
        Expr::Mul(a, c) => {
            let (x, y) = (emit(b, a, args, fr), emit(b, c, args, fr));
            b.ins().fmul(x, y)
        }
        Expr::Div(a, c) => {
            let (x, y) = (emit(b, a, args, fr), emit(b, c, args, fr));
            b.ins().fdiv(x, y)
        }
        Expr::Pow(a, c) => {
            let (x, y) = (emit(b, a, args, fr), emit(b, c, args, fr));
            call2(b, fr.powf, x, y)
        }
        Expr::IPow(a, n) => {
            let x = emit(b, a, args, fr);
            emit_ipow(b, x, *n)
        }
        Expr::Call(Func::Pow, a2) if a2.len() == 2 => {
            let (x, y) = (emit(b, &a2[0], args, fr), emit(b, &a2[1], args, fr));
            call2(b, fr.powf, x, y)
        }
        Expr::Call(f, a2) => {
            let x = emit(b, &a2[0], args, fr);
            match f {
                Func::Exp => call1(b, fr.exp, x),
                Func::Ln => call1(b, fr.ln, x),
                Func::Log => call1(b, fr.log10, x),
                Func::Sqrt => b.ins().sqrt(x),
                Func::Sin => call1(b, fr.sin, x),
                Func::Cos => call1(b, fr.cos, x),
                Func::Tan => call1(b, fr.tan, x),
                Func::Abs => b.ins().fabs(x),
                Func::Pow => unreachable!("2-arg pow handled above"),
            }
        }
    }
}
