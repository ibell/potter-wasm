#!/bin/sh
# Build and run the Rust vs C++ B3 side-by-side.
set -e
cd "$(dirname "$0")"

echo "building Rust bench (release, LTO) ..."
cargo build --release --bin bench >/dev/null 2>&1

echo "building C++ (clang -O3) ..."
mkdir -p target
clang++ -O3 -std=c++17 -o target/b2b3_cpp cpp/b2b3.cpp

echo
echo "================ Rust ================"
./target/release/bench
echo
echo "================ C++  ================"
./target/b2b3_cpp
