//! Noble-gas (Ne, Ar, Kr, Xe) Tang-Toennies pair potentials and their second
//! virial coefficients with Wigner-Kirkwood quantum corrections (to 3rd order),
//! reproducing potter's `integrate_potentials.py`. Real units: R in nm, V/k_B in
//! K, B2 in cm^3/mol. Spatial and temperature derivatives use `num-dual` autodiff.

use num_dual::DualNum;

// physical constants (SI), matching integrate_potentials.py
#[allow(dead_code)]
const KB: f64 = 1.380649e-23; // J/K
#[allow(dead_code)]
const HBAR: f64 = 1.054571817e-34; // J s
#[allow(dead_code)]
const U_AMU: f64 = 1.66053906660e-27; // kg
#[allow(dead_code)]
const N_A: f64 = 8.314462618 / KB; // 1/mol  (= 6.02214076e23)
#[allow(dead_code)]
const PI: f64 = std::f64::consts::PI;

/// A Tang-Toennies noble-gas potential. `V/k_B = A·exp(a1 R + a2 R² + an1/R + an2/R²)
/// − Σ_{n=3}^{8} C[2n]·[1 − e^{−bR}Σ_{k=0}^{2n}(bR)^k/k!]/R^{2n}`, R in nm. Below
/// `rcutoff·repsilon` a short-range `tilde_amp/R·e^{−tilde_exp·R}` form is used.
pub struct TangToennies {
    pub a: f64,
    pub a1: f64,
    pub a2: f64,
    pub an1: f64,
    pub an2: f64,
    pub b: f64,
    pub nmax: usize,
    pub c: [f64; 6], // C[6], C[8], C[10], C[12], C[14], C[16] — index n-3
    pub tilde_amp: f64,
    pub tilde_exp: f64,
    pub rcutoff: f64,
    pub repsilon: f64,
    pub mass_rel: f64,
}

impl TangToennies {
    /// Fill C[12], C[14], C[16] from C[6..10] by the recurrence
    /// C[2n] = C[2n-6]·(C[2n-2]/C[2n-4])³ for n = 6, 7, 8.
    fn add_recursive(&mut self) {
        for n in 6..=8 {
            let i = n - 3;
            self.c[i] = self.c[i - 3] * (self.c[i - 1] / self.c[i - 2]).powi(3);
        }
    }

    /// V/k_B [K] generic over the dual scalar (used for both value and derivatives).
    pub fn v_full<D: DualNum<f64> + Copy>(&self, r: D) -> D {
        if r.re() < self.rcutoff * self.repsilon {
            // short-range: tilde_amp/r · exp(-tilde_exp·r)
            r.recip() * self.tilde_amp * (r * (-self.tilde_exp)).exp()
        } else {
            let ri = r.recip();
            let mut out = (r * self.a1 + r * r * self.a2 + ri * self.an1 + ri * ri * self.an2)
                .exp()
                * self.a;
            for n in 3..=self.nmax {
                let br = r * self.b;
                // sum_{k=0}^{2n} br^k / k!
                let mut s = D::from(1.0); // k = 0
                let mut term = D::from(1.0);
                let mut fact = 1.0;
                for k in 1..=(2 * n) {
                    term = term * br;
                    fact *= k as f64;
                    s = s + term * (1.0 / fact);
                }
                let bracket = -((-br).exp() * s) + 1.0; // 1 - e^{-br}·s
                out = out - bracket * self.c[n - 3] * r.powi(2 * n as i32).recip();
            }
            out
        }
    }

    /// V/k_B [K] at R [nm].
    pub fn v(&self, r_nm: f64) -> f64 {
        self.v_full(r_nm)
    }

    /// (V, V', V'', V''') in K/nmᵏ at R [nm], via num-dual third derivative.
    pub fn v_derivs(&self, r_nm: f64) -> (f64, f64, f64, f64) {
        num_dual::third_derivative(|r| self.v_full(r), r_nm)
    }
}

fn ne() -> TangToennies {
    TangToennies {
        a: 0.402915058383e8, a1: -0.428654039586e2, a2: -0.333818674327e1,
        an1: -0.534644860719e-1, an2: 0.501774999419e-2, b: 0.492438731676e2, nmax: 8,
        c: [0.440676750157e-1, 0.164892507701e-2, 0.790473640524e-4,
            0.485489170103e-5, 0.382012334054e-6, 0.385106552963e-7],
        tilde_amp: 2.36770343e6, tilde_exp: 3.93124973e1,
        rcutoff: 0.4, repsilon: 0.30894556, mass_rel: 20.1797,
    }
}
pub fn neon_tt() -> TangToennies { ne() }

pub fn argon_tt() -> TangToennies {
    TangToennies {
        a: 4.61330146e7, a1: -2.98337630e1, a2: -9.71208881,
        an1: 2.75206827e-2, an2: -1.01489050e-2, b: 4.02517211e1, nmax: 8,
        c: [4.42812017e-1, 3.26707684e-2, 2.45656537e-3,
            1.88246247e-4, 1.47012192e-5, 1.17006343e-6],
        tilde_amp: 9.36167467e5, tilde_exp: 2.15969557e1,
        rcutoff: 0.4, repsilon: 0.376182, mass_rel: 39.948,
    }
}

pub fn krypton_tt() -> TangToennies {
    let mut k = TangToennies {
        a: 0.3200711798e8, a1: -0.2430565544e1 * 10.0, a2: -0.1435536209 * 100.0,
        an1: -0.4532273868 / 10.0, an2: 0.0, b: 0.2786344368e1 * 10.0, nmax: 8,
        c: [0.8992209265e6 / 1e6, 0.7316713603e7 / 1e8, 0.7835488511e8 / 1e10, 0.0, 0.0, 0.0],
        tilde_amp: 0.8268005465e7 / 10.0, tilde_exp: 0.1682493666e1 * 10.0,
        rcutoff: 0.3, repsilon: 4.015802 / 10.0, mass_rel: 83.798,
    };
    k.add_recursive();
    k
}

pub fn xenon_tt() -> TangToennies {
    let mut x = TangToennies {
        a: 0.579317071e8, a1: -0.208311994e1 * 10.0, a2: -0.147746919 * 100.0,
        an1: -0.289687722e1 / 10.0, an2: 0.258976595e1 / 100.0, b: 0.244337880e1 * 10.0, nmax: 8,
        c: [0.200298034e7 / 1e6, 0.199130481e8 / 1e8, 0.286841040e9 / 1e10, 0.0, 0.0, 0.0],
        tilde_amp: 4.18081481e6, tilde_exp: 2.38954061e1,
        rcutoff: 0.3, repsilon: 4.37798 / 10.0, mass_rel: 131.293,
    };
    x.add_recursive();
    x
}
