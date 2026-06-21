// Generate n_eff(T) for 4He and 3He from the full-quantum phase-shift engine, as CSV on
// stdout: "T,neff_he4,neff_he3". Used to plot vs the Cencek 2012 tabulated values
// (scripts/plot_neff.py). HEAVY: each point is a phase-shift B2 evaluation; the grid is
// focused on the 5-20 K n_eff peak (Bell 2020 Fig. 8). Run:
//   cargo run --release --example neff_curve > figures/neff_engine.csv
use potter_poc::quantum::{quantum_b2_neff, Species};
use std::io::Write;

fn main() {
    let ts = [
        2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 14.0, 17.0, 20.0, 25.0, 30.0,
        40.0, 60.0, 100.0, 200.0,
    ];
    let mut out = std::io::stdout();
    println!("T,neff_he4,neff_he3");
    for &t in &ts {
        let n4 = quantum_b2_neff(Species::He4, t).3;
        let n3 = quantum_b2_neff(Species::He3, t).3;
        println!("{t},{n4},{n3}");
        out.flush().unwrap();
        eprintln!("done T={t}: 4He={n4:.2} 3He={n3:.2}");
    }
}
