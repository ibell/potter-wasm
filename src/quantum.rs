//! Full-quantum B2 via Beth-Uhlenbeck phase shifts (variable-phase method).
//! Atomic units internally (Bohr, Hartree, electron mass, hbar=1).

/// Riccati-Bessel functions up to order `lmax`: jhat_l(x)=x j_l(x), yhat_l(x)=x y_l(x).
/// jhat_0=sin x, yhat_0=-cos x; jhat_1=sin x/x - cos x, yhat_1=-cos x/x - sin x;
/// both satisfy f_{l+1} = (2l+1)/x f_l - f_{l-1}. Upward recurrence (adequate for the
/// kr range used here; the full B2 vs the Cencek table is the high-l check).
pub fn riccati(lmax: usize, x: f64) -> (Vec<f64>, Vec<f64>) {
    let (s, c) = (x.sin(), x.cos());
    let mut j = vec![0.0; lmax + 1];
    let mut y = vec![0.0; lmax + 1];
    j[0] = s;
    y[0] = -c;
    if lmax >= 1 {
        j[1] = s / x - c;
        y[1] = -c / x - s;
    }
    for l in 1..lmax {
        let f = (2 * l + 1) as f64 / x;
        j[l + 1] = f * j[l] - j[l - 1];
        y[l + 1] = f * y[l] - y[l - 1];
    }
    (j, y)
}

/// Phase shift delta_l(k) for U(r)=2 mu V(r) via the Calogero variable-phase eq.,
/// integrated r0->rmax by RK4: delta_l'(r) = -(1/k) U(r) [cos d jhat_l(kr) - sin d yhat_l(kr)]^2.
/// `v`: V(r) [Hartree] closure. Returns delta_l (radians) for l=0..lmax.
pub fn phase_shifts<V: Fn(f64) -> f64>(
    v: &V,
    mu: f64,
    k: f64,
    lmax: usize,
    r0: f64,
    rmax: f64,
    steps: usize,
) -> Vec<f64> {
    let h = (rmax - r0) / steps as f64;
    let mut d = vec![0.0_f64; lmax + 1];
    let deriv = |r: f64, dl: &[f64]| -> Vec<f64> {
        let u = 2.0 * mu * v(r);
        let (j, y) = riccati(lmax, k * r);
        (0..=lmax)
            .map(|l| {
                let b = d_cos_sin(dl[l], j[l], y[l]);
                -(1.0 / k) * u * b * b
            })
            .collect()
    };
    let mut r = r0;
    for _ in 0..steps {
        let k1 = deriv(r, &d);
        let d2: Vec<f64> = (0..=lmax).map(|l| d[l] + 0.5 * h * k1[l]).collect();
        let k2 = deriv(r + 0.5 * h, &d2);
        let d3: Vec<f64> = (0..=lmax).map(|l| d[l] + 0.5 * h * k2[l]).collect();
        let k3 = deriv(r + 0.5 * h, &d3);
        let d4: Vec<f64> = (0..=lmax).map(|l| d[l] + h * k3[l]).collect();
        let k4 = deriv(r + h, &d4);
        for l in 0..=lmax {
            d[l] += h / 6.0 * (k1[l] + 2.0 * k2[l] + 2.0 * k3[l] + k4[l]);
        }
        r += h;
    }
    d
}

#[inline]
fn d_cos_sin(d: f64, jl: f64, yl: f64) -> f64 {
    d.cos() * jl - d.sin() * yl
}

/// Test helper: s-wave phase shift for a square well V=-v0 (r<rr) else 0.
pub fn s_wave_phase_for_test(mu: f64, v0: f64, rr: f64, k: f64) -> f64 {
    let v = |r: f64| if r < rr { -v0 } else { 0.0 };
    phase_shifts(&v, mu, k, 0, 1e-6, rr + 30.0, 6000)[0]
}
