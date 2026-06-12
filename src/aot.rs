//! AOT backend: compile a DSL potential to a standalone **WebAssembly module**
//! exporting `potential(r, eps, sig) -> f64`. The browser (or any wasm engine)
//! then JITs that module to native code — this is the JIT story for the web,
//! where you cannot run cranelift-native.
//!
//! The expression tree maps directly onto wasm's stack machine: post-order
//! emission of operands followed by the operator. Arithmetic, integer powers
//! (`IPow` -> multiplication chains using two scratch locals), `sqrt` and `abs`
//! are native wasm instructions; transcendentals (`exp`/`ln`/`sin`/…/`powf`) are
//! imported from `env` and supplied by the host at instantiation. The
//! Lennard-Jones potential needs no imports, so its module is self-contained.
//!
//! Pure Rust + `wasm-encoder`, so this works on native (emit a `.wasm` file) and
//! inside our own wasm module (wasm generating wasm).

use crate::dsl::{self, Expr, Func};
use wasm_encoder::{
    CodeSection, EntityType, ExportKind, ExportSection, Function as WFunction, FunctionSection,
    ImportSection, Instruction, Module, TypeSection, ValType,
};

/// Which `env` imports the expression actually needs, and their assigned
/// wasm function indices (imports occupy the low indices).
#[derive(Default)]
struct FuncIdx {
    exp: Option<u32>,
    ln: Option<u32>,
    log10: Option<u32>,
    sin: Option<u32>,
    cos: Option<u32>,
    tan: Option<u32>,
    powf: Option<u32>,
}

fn needs(e: &Expr, n: &mut FuncIdx) {
    // Reuse the Option fields as "needed" flags first (set to Some(0) as a marker),
    // then assign real indices afterward.
    match e {
        Expr::Num(_) | Expr::Var(_) => {}
        Expr::Neg(a) | Expr::IPow(a, _) => needs(a, n),
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) => {
            needs(a, n);
            needs(b, n);
        }
        Expr::Pow(a, b) => {
            n.powf = Some(0);
            needs(a, n);
            needs(b, n);
        }
        Expr::Call(f, args) => {
            match f {
                Func::Exp => n.exp = Some(0),
                Func::Ln => n.ln = Some(0),
                Func::Log => n.log10 = Some(0),
                Func::Sin => n.sin = Some(0),
                Func::Cos => n.cos = Some(0),
                Func::Tan => n.tan = Some(0),
                Func::Pow => n.powf = Some(0),
                Func::Sqrt | Func::Abs => {} // native wasm instructions
            }
            for a in args {
                needs(a, n);
            }
        }
    }
}

/// Compile `src` to a wasm module exporting `potential`. `vars` are the function
/// parameters in order (e.g. `["r", "eps", "sig"]`).
pub fn compile_to_wasm(src: &str, vars: &[&str]) -> Result<Vec<u8>, String> {
    let expr = dsl::compile(src, vars)?; // optimized tree (IPow folded)
    let nargs = vars.len() as u32;

    // Type section: 0 = (f64 * nargs) -> f64 (potential); 1 = (f64)->f64; 2 = (f64,f64)->f64.
    let mut types = TypeSection::new();
    let main_params: Vec<ValType> = (0..nargs).map(|_| ValType::F64).collect();
    types.ty().function(main_params, [ValType::F64]);
    types.ty().function([ValType::F64], [ValType::F64]);
    types
        .ty()
        .function([ValType::F64, ValType::F64], [ValType::F64]);

    // Determine needed imports and assign indices in a fixed order.
    let mut map = FuncIdx::default();
    needs(&expr, &mut map);
    let mut imports = ImportSection::new();
    let mut idx = 0u32;
    let mut unary = |imports: &mut ImportSection, slot: &mut Option<u32>, idx: &mut u32, name| {
        if slot.is_some() {
            imports.import("env", name, EntityType::Function(1));
            *slot = Some(*idx);
            *idx += 1;
        }
    };
    unary(&mut imports, &mut map.exp, &mut idx, "exp");
    unary(&mut imports, &mut map.ln, &mut idx, "ln");
    unary(&mut imports, &mut map.log10, &mut idx, "log10");
    unary(&mut imports, &mut map.sin, &mut idx, "sin");
    unary(&mut imports, &mut map.cos, &mut idx, "cos");
    unary(&mut imports, &mut map.tan, &mut idx, "tan");
    if map.powf.is_some() {
        imports.import("env", "powf", EntityType::Function(2));
        map.powf = Some(idx);
        idx += 1;
    }
    let num_imports = idx;

    // One defined function, of type 0.
    let mut funcs = FunctionSection::new();
    funcs.function(0);

    // Export it (its index is after the imported functions).
    let mut exports = ExportSection::new();
    exports.export("potential", ExportKind::Func, num_imports);

    // Code: two f64 scratch locals (for integer powers), then the body.
    let mut code = CodeSection::new();
    let mut f = WFunction::new([(2, ValType::F64)]);
    emit(&mut f, &expr, nargs, &map);
    f.instruction(&Instruction::End);
    code.function(&f);

    let mut module = Module::new();
    module.section(&types);
    if num_imports > 0 {
        module.section(&imports);
    }
    module.section(&funcs);
    module.section(&exports);
    module.section(&code);
    Ok(module.finish())
}

/// Validate generated bytes (so malformed bytecode is caught without a runtime).
pub fn validate(bytes: &[u8]) -> Result<(), String> {
    wasmparser::validate(bytes).map(|_| ()).map_err(|e| e.to_string())
}

fn emit(f: &mut WFunction, e: &Expr, nargs: u32, m: &FuncIdx) {
    match e {
        Expr::Num(v) => {
            f.instruction(&Instruction::F64Const((*v).into()));
        }
        Expr::Var(i) => {
            f.instruction(&Instruction::LocalGet(*i as u32));
        }
        Expr::Neg(a) => {
            emit(f, a, nargs, m);
            f.instruction(&Instruction::F64Neg);
        }
        Expr::Add(a, b) => {
            emit(f, a, nargs, m);
            emit(f, b, nargs, m);
            f.instruction(&Instruction::F64Add);
        }
        Expr::Sub(a, b) => {
            emit(f, a, nargs, m);
            emit(f, b, nargs, m);
            f.instruction(&Instruction::F64Sub);
        }
        Expr::Mul(a, b) => {
            emit(f, a, nargs, m);
            emit(f, b, nargs, m);
            f.instruction(&Instruction::F64Mul);
        }
        Expr::Div(a, b) => {
            emit(f, a, nargs, m);
            emit(f, b, nargs, m);
            f.instruction(&Instruction::F64Div);
        }
        Expr::Pow(a, b) => {
            emit(f, a, nargs, m);
            emit(f, b, nargs, m);
            f.instruction(&Instruction::Call(m.powf.unwrap()));
        }
        Expr::IPow(a, n) => {
            emit(f, a, nargs, m);
            emit_ipow(f, *n, nargs);
        }
        Expr::Call(Func::Pow, args) if args.len() == 2 => {
            emit(f, &args[0], nargs, m);
            emit(f, &args[1], nargs, m);
            f.instruction(&Instruction::Call(m.powf.unwrap()));
        }
        Expr::Call(func, args) => {
            emit(f, &args[0], nargs, m);
            match func {
                Func::Sqrt => {
                    f.instruction(&Instruction::F64Sqrt);
                }
                Func::Abs => {
                    f.instruction(&Instruction::F64Abs);
                }
                Func::Exp => {
                    f.instruction(&Instruction::Call(m.exp.unwrap()));
                }
                Func::Ln => {
                    f.instruction(&Instruction::Call(m.ln.unwrap()));
                }
                Func::Log => {
                    f.instruction(&Instruction::Call(m.log10.unwrap()));
                }
                Func::Sin => {
                    f.instruction(&Instruction::Call(m.sin.unwrap()));
                }
                Func::Cos => {
                    f.instruction(&Instruction::Call(m.cos.unwrap()));
                }
                Func::Tan => {
                    f.instruction(&Instruction::Call(m.tan.unwrap()));
                }
                Func::Pow => unreachable!("2-arg pow handled above"),
            }
        }
    }
}

/// Integer power on a value already on the stack (uses two scratch locals at
/// indices `nargs` and `nargs+1`).
fn emit_ipow(f: &mut WFunction, n: i32, nargs: u32) {
    let base = nargs;
    let tmp = nargs + 1;
    if n == 0 {
        f.instruction(&Instruction::Drop);
        f.instruction(&Instruction::F64Const(1.0f64.into()));
        return;
    }
    let m = n.unsigned_abs();
    f.instruction(&Instruction::LocalSet(base));
    f.instruction(&Instruction::LocalGet(base));
    for _ in 1..m {
        f.instruction(&Instruction::LocalGet(base));
        f.instruction(&Instruction::F64Mul);
    }
    if n < 0 {
        f.instruction(&Instruction::LocalSet(tmp));
        f.instruction(&Instruction::F64Const(1.0f64.into()));
        f.instruction(&Instruction::LocalGet(tmp));
        f.instruction(&Instruction::F64Div);
    }
}
