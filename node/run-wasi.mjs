// node.js test (WASI flavor): run the *full program* compiled to wasm32-wasip1
// under node's built-in WASI, so the same binary that wasmtime would run prints
// its verification table here.
//
//   build:  cargo build --release --target wasm32-wasip1 --bin potter_poc
//   run:    node node/run-wasi.mjs

import { readFileSync } from "node:fs";
import { WASI } from "node:wasi";

const wasi = new WASI({ version: "preview1", args: [], env: {} });
const wasmPath = new URL(
  "../target/wasm32-wasip1/release/potter_poc.wasm",
  import.meta.url,
);
const wasm = await WebAssembly.compile(readFileSync(wasmPath));
const instance = await WebAssembly.instantiate(wasm, wasi.getImportObject());
wasi.start(instance);
