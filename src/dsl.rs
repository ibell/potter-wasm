//! A tiny Python-like expression DSL: tokenizer + Pratt parser -> AST -> evaluator.
//!
//! This is hand-rolled to keep the prototype dependency-light and to make the
//! compiled hot-loop representation explicit. In a production build you would
//! instead reach for an existing crate:
//!   - `fasteval` : exprtk-style "parse once, evaluate millions of times" speed
//!   - `exmex`    : fast, and generic over the number type (f64 / dual / complex)
//!   - `evalexpr` / `meval` : simpler general-purpose evaluators
//!
//! Grammar (Python-like): + - * /, ** for power (right-assoc, binds tighter than
//! unary minus, so -2**2 == -4), parentheses, function calls exp/ln/log/sqrt/
//! sin/cos/tan/abs/pow, numbers (incl. 1.5e-6), and named variables.

#[derive(Debug, Clone)]
enum Tok {
    Num(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Pow,
    LParen,
    RParen,
    Comma,
}

fn tokenize(src: &str) -> Result<Vec<Tok>, String> {
    let b = src.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i] as char;
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        match c {
            '+' => {
                out.push(Tok::Plus);
                i += 1;
            }
            '-' => {
                out.push(Tok::Minus);
                i += 1;
            }
            '*' => {
                if i + 1 < b.len() && b[i + 1] == b'*' {
                    out.push(Tok::Pow);
                    i += 2;
                } else {
                    out.push(Tok::Star);
                    i += 1;
                }
            }
            '/' => {
                out.push(Tok::Slash);
                i += 1;
            }
            '(' => {
                out.push(Tok::LParen);
                i += 1;
            }
            ')' => {
                out.push(Tok::RParen);
                i += 1;
            }
            ',' => {
                out.push(Tok::Comma);
                i += 1;
            }
            _ if c.is_ascii_digit() || c == '.' => {
                let start = i;
                while i < b.len() {
                    let d = b[i] as char;
                    if d.is_ascii_digit() || d == '.' {
                        i += 1;
                    } else if d == 'e' || d == 'E' {
                        i += 1;
                        if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
                            i += 1;
                        }
                    } else {
                        break;
                    }
                }
                let s = &src[start..i];
                let v: f64 = s.parse().map_err(|_| format!("bad number '{}'", s))?;
                out.push(Tok::Num(v));
            }
            _ if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                while i < b.len() {
                    let d = b[i] as char;
                    if d.is_ascii_alphanumeric() || d == '_' {
                        i += 1;
                    } else {
                        break;
                    }
                }
                out.push(Tok::Ident(src[start..i].to_string()));
            }
            _ => return Err(format!("unexpected char '{}'", c)),
        }
    }
    Ok(out)
}

/// Built-in functions the DSL understands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Func {
    Exp,
    Ln,
    Log,
    Sqrt,
    Sin,
    Cos,
    Tan,
    Abs,
    Pow,
}

fn func_from_name(name: &str) -> Result<Func, String> {
    Ok(match name {
        "exp" => Func::Exp,
        "ln" => Func::Ln,
        "log" => Func::Log,
        "sqrt" => Func::Sqrt,
        "sin" => Func::Sin,
        "cos" => Func::Cos,
        "tan" => Func::Tan,
        "abs" => Func::Abs,
        "pow" => Func::Pow,
        other => return Err(format!("unknown function '{}'", other)),
    })
}

/// Compiled expression tree. Variables are resolved to slot indices at compile
/// time, so evaluation is a plain index into the environment slice.
#[derive(Debug, Clone)]
pub enum Expr {
    Num(f64),
    Var(usize),
    Neg(Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Pow(Box<Expr>, Box<Expr>),
    /// Integer power, `base^n` — emitted by the optimizer when the exponent is an
    /// integer literal. Evaluates via `.powi` (exponentiation by squaring) instead
    /// of the transcendental `powf`, which is far cheaper in the hot loop.
    IPow(Box<Expr>, i32),
    Call(Func, Vec<Expr>),
}

/// If `e` is an integer-valued literal (possibly negated) in a sane range, return
/// it as an `i32` exponent. Used to turn `x**12` into `IPow(x, 12)`.
fn as_int_exp(e: &Expr) -> Option<i32> {
    match e {
        Expr::Num(v) if v.fract() == 0.0 && v.abs() <= 1024.0 => Some(*v as i32),
        Expr::Neg(inner) => as_int_exp(inner).map(|n| -n),
        _ => None,
    }
}

/// Rewrite the tree, replacing `Pow`/`pow(...)` with a constant integer exponent
/// by the cheap `IPow` node, and recursing into all children.
fn optimize(e: Expr) -> Expr {
    match e {
        Expr::Pow(a, b) => {
            let a = optimize(*a);
            if let Some(n) = as_int_exp(&b) {
                Expr::IPow(Box::new(a), n)
            } else {
                Expr::Pow(Box::new(a), Box::new(optimize(*b)))
            }
        }
        Expr::Call(Func::Pow, args) if args.len() == 2 => {
            let mut it = args.into_iter();
            let base = optimize(it.next().unwrap());
            let exp = it.next().unwrap();
            if let Some(n) = as_int_exp(&exp) {
                Expr::IPow(Box::new(base), n)
            } else {
                Expr::Call(Func::Pow, vec![base, optimize(exp)])
            }
        }
        Expr::Neg(a) => Expr::Neg(Box::new(optimize(*a))),
        Expr::Add(a, b) => Expr::Add(Box::new(optimize(*a)), Box::new(optimize(*b))),
        Expr::Sub(a, b) => Expr::Sub(Box::new(optimize(*a)), Box::new(optimize(*b))),
        Expr::Mul(a, b) => Expr::Mul(Box::new(optimize(*a)), Box::new(optimize(*b))),
        Expr::Div(a, b) => Expr::Div(Box::new(optimize(*a)), Box::new(optimize(*b))),
        Expr::Call(f, args) => Expr::Call(f, args.into_iter().map(optimize).collect()),
        other => other, // Num, Var, IPow
    }
}

struct Parser<'a> {
    toks: Vec<Tok>,
    pos: usize,
    vars: &'a [&'a str],
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }
    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    // Pratt / precedence-climbing parser. `min_bp` is the minimum binding power
    // an infix operator must have to be consumed at this level.
    fn parse_expr(&mut self, min_bp: u8) -> Result<Expr, String> {
        let mut lhs = self.parse_prefix()?;
        loop {
            let (lbp, rbp, ctor): (u8, u8, fn(Box<Expr>, Box<Expr>) -> Expr) = match self.peek() {
                Some(Tok::Plus) => (10, 11, |a, b| Expr::Add(a, b)),
                Some(Tok::Minus) => (10, 11, |a, b| Expr::Sub(a, b)),
                Some(Tok::Star) => (20, 21, |a, b| Expr::Mul(a, b)),
                Some(Tok::Slash) => (20, 21, |a, b| Expr::Div(a, b)),
                // ** : higher than unary minus (bp 35) and right-associative
                // (left_bp 41 > right_bp 40), so -2**2 == -4 and 2**3**2 == 512.
                Some(Tok::Pow) => (41, 40, |a, b| Expr::Pow(a, b)),
                _ => break,
            };
            if lbp < min_bp {
                break;
            }
            self.next(); // consume the operator
            let rhs = self.parse_expr(rbp)?;
            lhs = ctor(Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_prefix(&mut self) -> Result<Expr, String> {
        match self.next() {
            Some(Tok::Num(v)) => Ok(Expr::Num(v)),
            Some(Tok::Minus) => Ok(Expr::Neg(Box::new(self.parse_expr(35)?))),
            Some(Tok::Plus) => self.parse_expr(35),
            Some(Tok::LParen) => {
                let e = self.parse_expr(0)?;
                match self.next() {
                    Some(Tok::RParen) => Ok(e),
                    _ => Err("expected ')'".into()),
                }
            }
            Some(Tok::Ident(name)) => {
                if matches!(self.peek(), Some(Tok::LParen)) {
                    self.next(); // consume '('
                    let mut args = Vec::new();
                    if !matches!(self.peek(), Some(Tok::RParen)) {
                        loop {
                            args.push(self.parse_expr(0)?);
                            match self.peek() {
                                Some(Tok::Comma) => {
                                    self.next();
                                }
                                _ => break,
                            }
                        }
                    }
                    match self.next() {
                        Some(Tok::RParen) => {}
                        _ => return Err("expected ')'".into()),
                    }
                    Ok(Expr::Call(func_from_name(&name)?, args))
                } else {
                    let idx = self
                        .vars
                        .iter()
                        .position(|v| *v == name)
                        .ok_or_else(|| format!("unknown variable '{}'", name))?;
                    Ok(Expr::Var(idx))
                }
            }
            other => Err(format!("unexpected token {:?}", other)),
        }
    }
}

fn parse_only(src: &str, vars: &[&str]) -> Result<Expr, String> {
    let toks = tokenize(src)?;
    let mut p = Parser {
        toks,
        pos: 0,
        vars,
    };
    let e = p.parse_expr(0)?;
    if p.pos != p.toks.len() {
        return Err("trailing tokens after expression".into());
    }
    Ok(e)
}

/// Parse `src` into an optimized expression tree (integer powers folded to `IPow`).
/// `vars` lists the variable names in the order they appear in the env slice.
pub fn compile(src: &str, vars: &[&str]) -> Result<Expr, String> {
    Ok(optimize(parse_only(src, vars)?))
}

/// Same as `compile` but WITHOUT the integer-power optimization — keeps `**`/`pow`
/// as transcendental `powf`. Used to measure the optimization's effect.
pub fn compile_unoptimized(src: &str, vars: &[&str]) -> Result<Expr, String> {
    parse_only(src, vars)
}

/// Evaluate a compiled expression against an environment slice (indexed by the
/// variable order passed to `compile`).
pub fn eval(e: &Expr, env: &[f64]) -> f64 {
    match e {
        Expr::Num(v) => *v,
        Expr::Var(i) => env[*i],
        Expr::Neg(a) => -eval(a, env),
        Expr::Add(a, b) => eval(a, env) + eval(b, env),
        Expr::Sub(a, b) => eval(a, env) - eval(b, env),
        Expr::Mul(a, b) => eval(a, env) * eval(b, env),
        Expr::Div(a, b) => eval(a, env) / eval(b, env),
        // std f64 methods use the platform libm on native (same as C++), so the
        // hot loop is fair against the C++ side-by-side; on wasm they resolve to
        // Rust's built-in libm. (We avoid the `libm` *crate* here on purpose.)
        Expr::Pow(a, b) => eval(a, env).powf(eval(b, env)),
        Expr::IPow(a, n) => eval(a, env).powi(*n),
        Expr::Call(f, args) => {
            let x = eval(&args[0], env);
            match f {
                Func::Exp => x.exp(),
                Func::Ln => x.ln(),
                Func::Log => x.log10(),
                Func::Sqrt => x.sqrt(),
                Func::Sin => x.sin(),
                Func::Cos => x.cos(),
                Func::Tan => x.tan(),
                Func::Abs => x.abs(),
                Func::Pow => x.powf(eval(&args[1], env)),
            }
        }
    }
}

// ----------------------- CSE / bytecode backend -----------------------
//
// Flatten the (optimized) expression tree into a value-numbered DAG: a flat list
// of operations where each op references earlier results by slot index. Building
// with a structural cache means identical subexpressions (e.g. `(sig/r)` shared
// by `**12` and `**6`) collapse to a single slot — common-subexpression
// elimination — and evaluation becomes a straight array loop, with no recursion
// or pointer-chasing.

use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Op {
    Const(u64), // f64 bit pattern
    Var(usize), // env index
    Neg(usize),
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Div(usize, usize),
    Pow(usize, usize),
    IPow(usize, i32),
    Call1(Func, usize),
}

/// A compiled, CSE'd potential: a flat op list evaluated in order.
pub struct Program {
    ops: Vec<Op>,
    root: usize,
}

impl Program {
    /// Number of distinct operations after CSE (a tree node count would be larger).
    pub fn len(&self) -> usize {
        self.ops.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

struct Flattener {
    ops: Vec<Op>,
    cache: HashMap<Op, usize>,
}

impl Flattener {
    fn push(&mut self, op: Op) -> usize {
        if let Some(&i) = self.cache.get(&op) {
            return i; // CSE: identical op already computed -> reuse its slot
        }
        let i = self.ops.len();
        self.ops.push(op);
        self.cache.insert(op, i);
        i
    }

    fn build(&mut self, e: &Expr) -> usize {
        let op = match e {
            Expr::Num(v) => Op::Const(v.to_bits()),
            Expr::Var(i) => Op::Var(*i),
            Expr::Neg(a) => {
                let x = self.build(a);
                Op::Neg(x)
            }
            Expr::Add(a, b) => {
                let (x, y) = (self.build(a), self.build(b));
                Op::Add(x, y)
            }
            Expr::Sub(a, b) => {
                let (x, y) = (self.build(a), self.build(b));
                Op::Sub(x, y)
            }
            Expr::Mul(a, b) => {
                let (x, y) = (self.build(a), self.build(b));
                Op::Mul(x, y)
            }
            Expr::Div(a, b) => {
                let (x, y) = (self.build(a), self.build(b));
                Op::Div(x, y)
            }
            Expr::Pow(a, b) => {
                let (x, y) = (self.build(a), self.build(b));
                Op::Pow(x, y)
            }
            Expr::IPow(a, n) => {
                let x = self.build(a);
                Op::IPow(x, *n)
            }
            Expr::Call(Func::Pow, args) if args.len() == 2 => {
                let (x, y) = (self.build(&args[0]), self.build(&args[1]));
                Op::Pow(x, y)
            }
            Expr::Call(f, args) => {
                let x = self.build(&args[0]);
                Op::Call1(*f, x)
            }
        };
        self.push(op)
    }
}

/// Parse, optimize (integer powers), and flatten to a CSE'd program.
pub fn compile_program(src: &str, vars: &[&str]) -> Result<Program, String> {
    let expr = compile(src, vars)?;
    let mut f = Flattener {
        ops: Vec::new(),
        cache: HashMap::new(),
    };
    let root = f.build(&expr);
    Ok(Program { ops: f.ops, root })
}

/// Number of scratch slots an evaluation needs (== op count).
pub fn program_slots(p: &Program) -> usize {
    p.ops.len()
}

/// Evaluate a compiled program against an env slice. `scratch` is a caller-owned
/// buffer of length >= program_slots() — reused across calls so the hot loop does
/// no allocation and no oversized initialization.
pub fn eval_program(p: &Program, env: &[f64], s: &mut [f64]) -> f64 {
    for (i, op) in p.ops.iter().enumerate() {
        s[i] = match *op {
            Op::Const(b) => f64::from_bits(b),
            Op::Var(j) => env[j],
            Op::Neg(a) => -s[a],
            Op::Add(a, b) => s[a] + s[b],
            Op::Sub(a, b) => s[a] - s[b],
            Op::Mul(a, b) => s[a] * s[b],
            Op::Div(a, b) => s[a] / s[b],
            Op::Pow(a, b) => s[a].powf(s[b]),
            Op::IPow(a, n) => s[a].powi(n),
            Op::Call1(func, a) => {
                let x = s[a];
                match func {
                    Func::Exp => x.exp(),
                    Func::Ln => x.ln(),
                    Func::Log => x.log10(),
                    Func::Sqrt => x.sqrt(),
                    Func::Sin => x.sin(),
                    Func::Cos => x.cos(),
                    Func::Tan => x.tan(),
                    Func::Abs => x.abs(),
                    Func::Pow => unreachable!("2-arg pow is lowered to Op::Pow"),
                }
            }
        };
    }
    s[p.root]
}
