#!/bin/sh
# Build the browser wasm and embed it (base64) into a single standalone index.html
# that opens directly via file:// — no server needed.
set -e
cd "$(dirname "$0")/.."

echo "building wasm32-unknown-unknown lib ..."
cargo build --release --target wasm32-unknown-unknown --lib >/dev/null 2>&1

mkdir -p docs
WASM=target/wasm32-unknown-unknown/release/potter_poc.wasm
python3 - "$WASM" web/index.html.in docs/index.html <<'PY'
import sys, base64
wasm, tin, tout = sys.argv[1], sys.argv[2], sys.argv[3]
b64 = base64.b64encode(open(wasm, "rb").read()).decode()
html = open(tin).read().replace("__WASM_B64__", b64)
open(tout, "w").write(html)
print(f"  wrote {tout} ({len(html)//1024} KB, wasm {len(b64)*3//4//1024} KB embedded)")
PY
echo "open docs/index.html in a browser (file:// works); also served by GitHub Pages."
