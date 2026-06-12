//! Generate standalone .wasm modules for DSL potentials (AOT). Each module
//! exports `potential(r, eps, sig) -> f64`. node/run-aot.mjs (or a browser) then
//! instantiates and runs them at native speed.

use potter_poc::aot::{compile_to_wasm, validate};
use std::fs;

fn gen(name: &str, src: &str, path: &str) {
    let bytes = compile_to_wasm(src, &["r", "eps", "sig"]).expect("compile_to_wasm");
    validate(&bytes).expect("generated wasm is valid");
    fs::write(path, &bytes).expect("write wasm");
    let imports = if src.contains("exp") || src.contains("sqrt") || src.contains("ln") {
        "imports env.* transcendentals"
    } else {
        "self-contained (no imports)"
    };
    println!("  {name:<22} {} bytes  -> {path}  ({imports})", bytes.len());
    println!("       V(r) = {src}");
}

fn main() {
    fs::create_dir_all("target").ok();
    println!("AOT: DSL potential -> standalone .wasm exporting potential(r,eps,sig)");
    gen(
        "lennard-jones",
        "4*eps*((sig/r)**12 - (sig/r)**6)",
        "target/lj_potential.wasm",
    );
    gen(
        "morse",
        "eps*(exp(-2*(r-sig)) - 2*exp(-(r-sig)))",
        "target/morse_potential.wasm",
    );
}
