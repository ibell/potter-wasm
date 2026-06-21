use potter_poc::quantum::{classical_b2, quantum_b2, Species};

fn main() {
    let refs = [
        (4.0, -85.061),
        (10.0, -23.125),
        (20.0, -2.7464),
        (100.0, 11.6747),
        (500.0, 11.00715),
    ];
    for (t, r) in refs {
        let g = quantum_b2(Species::He4, t);
        println!("T={t:>6}  quantum={g:>12.5}  ref={r:>10.5}  diff={:>10.5}", g - r);
    }
    for t in [2000.0_f64, 5000.0] {
        let q = quantum_b2(Species::He4, t);
        let c = classical_b2(Species::He4, t);
        println!(
            "T={t:>6}  quantum={q:>12.5}  classical={c:>12.5}  rel={:>10.5}",
            (q - c).abs() / c.abs()
        );
    }
    println!("classical_b2(He4,500) = {}", classical_b2(Species::He4, 500.0));
}
