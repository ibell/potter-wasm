//! Rigid linear multi-site molecules — the original motivation for potter.
//!
//! A molecule is a set of interaction sites placed along its axis (offset `d`),
//! each carrying LJ parameters (eps in K, sigma in Angstrom) and a point charge
//! (in units of e). The pair energy between two molecules is the sum over all
//! site-site interactions (LJ with Lorentz-Berthelot mixing + Coulomb), and
//! depends on the centre-of-mass separation `r` and the two orientations.
//!
//! The second virial coefficient of a linear molecule is the 4-D integral
//!   B2 = -1/4  integral_0^inf r^2 dr  integral_0^pi sin(th1) dth1
//!             integral_0^pi sin(th2) dth2  integral_0^2pi dphi  (e^{-U/T} - 1)
//! evaluated with the Genz-Malik `hcubature` from this crate. Real units in;
//! B2 out in cm^3/mol. Validated by the exact single-site limit (-> spherical B2).

use crate::cubature::hcubature;
use std::collections::HashMap;
use std::f64::consts::PI;

/// e^2 / (4 pi eps0 kB), in K*Angstrom — converts q_i q_j / r[A] to energy in K.
const COULOMB_K: f64 = 167101.0;
/// 1 Angstrom^3 * N_A, in cm^3/mol.
const ANG3_TO_CM3MOL: f64 = 0.602214;
/// hbar^2 / (24 kB), expressed in amu*Angstrom^2*K, for the first-order Wigner-
/// Kirkwood quantum correction: q = (Q_CONST/(mass[amu]*T^2)) exp(-U/T) * d2U,
/// with U in K, second derivatives in K/Angstrom^2 (or K/rad^2). ~= 2.0214.
const Q_CONST: f64 =
    1.054571817e-34 * 1.054571817e-34 / (24.0 * 1.380649e-23) / (1.66053907e-27 * 1e-10 * 1e-10);

/// Unit axis vector at polar angle `theta` from z and azimuth `phi`.
#[inline]
fn axis(theta: f64, phi: f64) -> [f64; 3] {
    [theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()]
}
#[inline]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
#[inline]
fn norm3(a: [f64; 3]) -> [f64; 3] {
    let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    [a[0] / n, a[1] / n, a[2] / n]
}
/// Two orthonormal axes perpendicular to unit vector `u` (a linear molecule's two
/// principal rotation axes).
#[inline]
fn perp_axes(u: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let t = if u[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
    let a = norm3(cross(u, t));
    (a, cross(u, a))
}

#[derive(Clone, Copy)]
pub struct Site {
    pub d: f64,   // offset along the molecular axis, Angstrom
    pub eps: f64, // LJ well depth, K (0 for a charge-only site)
    pub sig: f64, // LJ diameter, Angstrom
    pub q: f64,   // point charge, units of e
}

pub struct Linear {
    pub sites: Vec<Site>,
}

impl Linear {
    /// Pair energy (K) for COM separation `r` (along z), molecule 1 tilted by
    /// `th1` in the xz-plane, molecule 2 by polar/azimuthal (`th2`, `phi`).
    fn energy(&self, r: f64, th1: f64, th2: f64, phi: f64) -> f64 {
        let u1 = [th1.sin(), 0.0, th1.cos()];
        let u2 = [th2.sin() * phi.cos(), th2.sin() * phi.sin(), th2.cos()];
        let mut u = 0.0;
        for a in &self.sites {
            let pa = [a.d * u1[0], a.d * u1[1], a.d * u1[2]];
            for b in &self.sites {
                let pb = [b.d * u2[0], b.d * u2[1], r + b.d * u2[2]];
                let dx = pa[0] - pb[0];
                let dy = pa[1] - pb[1];
                let dz = pa[2] - pb[2];
                let rij = (dx * dx + dy * dy + dz * dz).sqrt();
                if rij <= 1e-12 {
                    return f64::INFINITY;
                }
                let eps = (a.eps * b.eps).sqrt(); // Lorentz-Berthelot
                if eps > 0.0 {
                    let sig = 0.5 * (a.sig + b.sig);
                    let sr6 = (sig / rij).powi(6);
                    u += 4.0 * eps * (sr6 * sr6 - sr6);
                }
                if a.q != 0.0 && b.q != 0.0 {
                    u += COULOMB_K * a.q * b.q / rij;
                }
            }
        }
        u
    }

    /// B2(T) in cm^3/mol, via 4-D adaptive cubature. Returns (B2, integrand evals).
    pub fn b2(&self, t: f64, reltol: f64) -> (f64, usize) {
        let integrand = |x: &[f64]| -> f64 {
            let (s, th1, th2, phi) = (x[0], x[1], x[2], x[3]);
            let om = 1.0 - s;
            if om <= 0.0 {
                return 0.0;
            }
            let r = s / om;
            let jac = 1.0 / (om * om); // dr/ds
            let u = self.energy(r, th1, th2, phi);
            let mayer = if u.is_finite() {
                (-u / t).exp() - 1.0
            } else {
                -1.0
            };
            let val = r * r * jac * th1.sin() * th2.sin() * mayer;
            if val.is_finite() {
                val
            } else {
                0.0
            }
        };
        let (i, _e, nev) = hcubature(
            4,
            &integrand,
            &[0.0, 0.0, 0.0, 0.0],
            &[1.0, PI, PI, 2.0 * PI],
            1e-13,
            reltol,
            20_000_000,
        );
        (-0.25 * i * ANG3_TO_CM3MOL, nev)
    }
}

/// 4-D orientational B2 integral for a linear molecule, shared by all models.
/// `energy(r, th1, th2, phi)` returns the pair energy in K. B2 out in cm^3/mol.
///
/// `rmin` (Angstrom) is a hard-core cutoff: for r < rmin the molecules are treated
/// as fully excluded (Mayer = -1) and that region is added analytically as
/// (2*pi/3) rmin^3. This is essential for site-site potentials with bounded
/// exponential repulsion + point charges (Hellmann), where the unscreened 1/R
/// Coulomb attraction would otherwise make the short-range integrand diverge.
/// LJ models (r^-12 core) are safe with rmin = 0.
fn b2_orientational<F: Fn(f64, f64, f64, f64) -> f64>(
    energy: F,
    t: f64,
    reltol: f64,
    rmin: f64,
) -> (f64, usize) {
    let s_lo = rmin / (1.0 + rmin); // r = s/(1-s) maps s_lo -> rmin
    let integrand = |x: &[f64]| -> f64 {
        let (s, th1, th2, phi) = (x[0], x[1], x[2], x[3]);
        let om = 1.0 - s;
        if om <= 0.0 {
            return 0.0;
        }
        let r = s / om;
        let jac = 1.0 / (om * om);
        let u = energy(r, th1, th2, phi);
        let mayer = if u.is_finite() {
            (-u / t).exp() - 1.0
        } else {
            -1.0
        };
        let val = r * r * jac * th1.sin() * th2.sin() * mayer;
        if val.is_finite() {
            val
        } else {
            0.0
        }
    };
    // Absolute-tolerance floor on the raw integral (Angstrom^3): 0.03 here maps to
    // ~0.005 cm^3/mol (the SI rounding), so B2 near its Boyle zero stops on
    // absolute rather than *relative* error (which would chase forever / hit the
    // eval cap) while keeping the validated accuracy elsewhere.
    let (i, _e, nev) = hcubature(
        4,
        &integrand,
        &[s_lo, 0.0, 0.0, 0.0],
        &[1.0, PI, PI, 2.0 * PI],
        0.03,
        reltol,
        20_000_000,
    );
    let b2_ang3 = -0.25 * i + (2.0 * PI / 3.0) * rmin.powi(3);
    (b2_ang3 * ANG3_TO_CM3MOL, nev)
}

/// A high-accuracy ab initio site-site potential (Hellmann form): each site has a
/// position along the axis, a type, and a charge (in absorbed units, so the
/// Coulomb term q_i q_j / R is directly in K with R in Angstrom). The site-site
/// interaction per type-pair is
///   V_ij = A exp(-a R) - f6(b,R) C6/R^6 - f8(b,R) C8/R^8 + q_i q_j / R
/// with Tang-Toennies damping f_2n = 1 - exp(-bR) sum_{k=0}^{2n} (bR)^k/k!.
pub struct RigidLinear {
    sites: Vec<(f64, usize, f64)>, // (offset d in A, type index, charge)
    ntypes: usize,
    table: Vec<[f64; 5]>, // (A, alpha, b, C6, C8), flat ntypes*ntypes (symmetric)
}

impl RigidLinear {
    /// Build from a type-pair coefficient map; flattened to a dense table so the
    /// hot loop does an array index, not a hash lookup, per site-site term.
    fn new(
        sites: Vec<(f64, usize, f64)>,
        ntypes: usize,
        coeffs: HashMap<(usize, usize), [f64; 5]>,
    ) -> Self {
        let mut table = vec![[0.0; 5]; ntypes * ntypes];
        for (&(i, j), &c) in &coeffs {
            table[i * ntypes + j] = c;
            table[j * ntypes + i] = c;
        }
        RigidLinear { sites, ntypes, table }
    }

    #[inline]
    fn site_site(&self, ti: usize, tj: usize, r: f64) -> f64 {
        let [a, alpha, b, c6, c8] = self.table[ti * self.ntypes + tj];
        let br = b * r;
        let (mut term, mut s6, mut s8) = (1.0, 0.0, 0.0);
        for k in 0..=8 {
            if k <= 6 {
                s6 += term;
            }
            s8 += term;
            term *= br / ((k + 1) as f64);
        }
        let e = (-br).exp();
        let f6 = 1.0 - e * s6;
        let f8 = 1.0 - e * s8;
        a * (-alpha * r).exp() - f6 * c6 / r.powi(6) - f8 * c8 / r.powi(8)
    }

    /// Pair energy (K) for arbitrary geometry: molecule 1 at the origin with axis
    /// `u1`, molecule 2 with COM at `com2` and axis `u2` (both unit vectors).
    #[inline]
    fn energy_vec(&self, com2: [f64; 3], u1: [f64; 3], u2: [f64; 3]) -> f64 {
        let mut u = 0.0;
        for &(da, ta, qa) in &self.sites {
            let pa = [da * u1[0], da * u1[1], da * u1[2]];
            for &(db, tb, qb) in &self.sites {
                let pb = [com2[0] + db * u2[0], com2[1] + db * u2[1], com2[2] + db * u2[2]];
                let dx = pa[0] - pb[0];
                let dy = pa[1] - pb[1];
                let dz = pa[2] - pb[2];
                let rij = (dx * dx + dy * dy + dz * dz).sqrt();
                if rij < 0.02 {
                    return f64::INFINITY;
                }
                u += self.site_site(ta, tb, rij) + qa * qb / rij;
            }
        }
        u
    }

    /// Pair energy (K) for COM separation r, orientations (th1; th2, phi).
    pub fn energy(&self, r: f64, th1: f64, th2: f64, phi: f64) -> f64 {
        self.energy_vec([0.0, 0.0, r], axis(th1, 0.0), axis(th2, phi))
    }

    /// Type-pair site-site potential and its first two R-derivatives (g, g', g'').
    /// (The Coulomb part is added per-site by the caller.)
    #[inline]
    fn site_site_derivs(&self, ti: usize, tj: usize, r: f64) -> (f64, f64, f64) {
        let [a, alpha, b, c6, c8] = self.table[ti * self.ntypes + tj];
        let er = (-alpha * r).exp();
        let (mut g, mut gp, mut gpp) = (a * er, -alpha * a * er, alpha * alpha * a * er);

        let br = b * r;
        let ebr = (-br).exp();
        let (mut term, mut s6, mut s8) = (1.0, 0.0, 0.0);
        for k in 0..=8 {
            if k <= 6 {
                s6 += term;
            }
            s8 += term;
            term *= br / ((k + 1) as f64);
        }
        let f6 = 1.0 - ebr * s6;
        let f8 = 1.0 - ebr * s8;
        // f_2n'(R) = b e^{-bR}(bR)^{2n}/(2n)! ;  f_2n''(R) = b^2 e^{-bR}(bR)^{2n-1}(2n-bR)/(2n)!
        let f6p = b * ebr * br.powi(6) / 720.0;
        let f8p = b * ebr * br.powi(8) / 40320.0;
        let f6pp = b * b * ebr * br.powi(5) * (6.0 - br) / 720.0;
        let f8pp = b * b * ebr * br.powi(7) * (8.0 - br) / 40320.0;

        let (r6, r7, r8, r9, r10) = (r.powi(6), r.powi(7), r.powi(8), r.powi(9), r.powi(10));
        g += -c6 * f6 / r6 - c8 * f8 / r8;
        gp += -c6 * (f6p / r6 - 6.0 * f6 / r7) - c8 * (f8p / r8 - 8.0 * f8 / r9);
        gpp += -c6 * (f6pp / r6 - 12.0 * f6p / r7 + 42.0 * f6 / r8)
            - c8 * (f8pp / r8 - 16.0 * f8p / r9 + 72.0 * f8 / r10);
        (g, gp, gpp)
    }

    /// First-order quantum-correction term q12 (Hellmann Eq. 18-19), computed
    /// ANALYTICALLY: the translational Laplacian (Coulomb part is identically zero
    /// since 1/R is harmonic) and the rotational angular Laplacians of U, weighted
    /// by 1/mu and 1/I and the Boltzmann factor. `u0` is U at the centre.
    fn q_corr(
        &self,
        com2: [f64; 3],
        u1: [f64; 3],
        u2: [f64; 3],
        u0: f64,
        t: f64,
        inv_mu: f64,
        inv_i: f64,
    ) -> f64 {
        if !u0.is_finite() {
            return 0.0;
        }
        let (a1, b1) = perp_axes(u1);
        let (a2, b2) = perp_axes(u2);
        let mut lap_t = 0.0;
        let mut rot = 0.0;
        // second derivative of R = |w| along a rotation whose w-velocity is dw,
        // acceleration d2w: dR'' = (dw.dw + w.d2w)/R - (w.dw)^2/R^3
        let d2r = |w: [f64; 3], dw: [f64; 3], d2w: [f64; 3], rr: f64, r2: f64, vp: f64, vpp: f64| {
            let wdw = w[0] * dw[0] + w[1] * dw[1] + w[2] * dw[2];
            let dwdw = dw[0] * dw[0] + dw[1] * dw[1] + dw[2] * dw[2];
            let wd2w = w[0] * d2w[0] + w[1] * d2w[1] + w[2] * d2w[2];
            let dr = wdw / rr;
            let d2 = (dwdw + wd2w) / rr - wdw * wdw / (rr * r2);
            vpp * dr * dr + vp * d2
        };
        for &(da, ta, qa) in &self.sites {
            let pa = [da * u1[0], da * u1[1], da * u1[2]];
            for &(db, tb, qb) in &self.sites {
                let qpos = [com2[0] + db * u2[0], com2[1] + db * u2[1], com2[2] + db * u2[2]];
                let w = [pa[0] - qpos[0], pa[1] - qpos[1], pa[2] - qpos[2]];
                let r2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
                let rr = r2.sqrt();
                let (_g, gp, gpp) = self.site_site_derivs(ta, tb, rr);
                let qq = qa * qb;
                let vp = gp - qq / r2; // total v'  (Coulomb v' = -qq/R^2)
                let vpp = gpp + 2.0 * qq / (rr * r2); // total v'' (Coulomb v'' = 2qq/R^3)
                // translational Laplacian: type-pair only (Coulomb Laplacian = 0)
                lap_t += gpp + 2.0 / rr * gp;
                // rotational, molecule 1 (rotate site a about its COM): dw = ax x pa
                let qrel = [db * u2[0], db * u2[1], db * u2[2]];
                for ax in [a1, b1] {
                    let dw = cross(ax, pa);
                    let d2w = cross(ax, cross(ax, pa));
                    rot += d2r(w, dw, d2w, rr, r2, vp, vpp);
                }
                // rotational, molecule 2 (rotate site b about COM2): dw = -(ax x qrel)
                for ax in [a2, b2] {
                    let dq = cross(ax, qrel);
                    let d2q = cross(ax, cross(ax, qrel));
                    let dw = [-dq[0], -dq[1], -dq[2]];
                    let d2w = [-d2q[0], -d2q[1], -d2q[2]];
                    rot += d2r(w, dw, d2w, rr, r2, vp, vpp);
                }
            }
        }
        Q_CONST / (t * t) * (-u0 / t).exp() * (inv_mu * lap_t + inv_i * rot)
    }

    /// B2(T) with the first-order Wigner-Kirkwood quantum correction (Hellmann
    /// Eq. 22): integrate (Mayer - q12). `mu_amu` is the reduced mass of the
    /// molecule pair, `i_amu_a2` the molecular moment of inertia (amu*Angstrom^2).
    pub fn b2_quantum(&self, t: f64, reltol: f64, mu_amu: f64, i_amu_a2: f64) -> (f64, usize) {
        let (inv_mu, inv_i) = (1.0 / mu_amu, 1.0 / i_amu_a2);
        let rmin = 2.0;
        let s_lo = rmin / (1.0 + rmin);
        let integrand = |x: &[f64]| -> f64 {
            let (s, th1, th2, phi) = (x[0], x[1], x[2], x[3]);
            let om = 1.0 - s;
            if om <= 0.0 {
                return 0.0;
            }
            let r = s / om;
            let jac = 1.0 / (om * om);
            let (u1, u2) = (axis(th1, 0.0), axis(th2, phi));
            let com2 = [0.0, 0.0, r];
            let u = self.energy_vec(com2, u1, u2);
            let (mayer, q) = if u.is_finite() {
                ((-u / t).exp() - 1.0, self.q_corr(com2, u1, u2, u, t, inv_mu, inv_i))
            } else {
                (-1.0, 0.0)
            };
            let val = r * r * jac * th1.sin() * th2.sin() * (mayer - q);
            if val.is_finite() {
                val
            } else {
                0.0
            }
        };
        let (i, _e, nev) = hcubature(
            4,
            &integrand,
            &[s_lo, 0.0, 0.0, 0.0],
            &[1.0, PI, PI, 2.0 * PI],
            0.03,
            reltol,
            20_000_000,
        );
        let b2_ang3 = -0.25 * i + (2.0 * PI / 3.0) * rmin.powi(3);
        (b2_ang3 * ANG3_TO_CM3MOL, nev)
    }

    pub fn b2(&self, t: f64, reltol: f64) -> (f64, usize) {
        // 2 Angstrom hard-core cutoff (matches potter) to avoid the unphysical
        // short-range point-charge Coulomb catastrophe of the exp-repulsion form.
        b2_orientational(|r, t1, t2, p| self.energy(r, t1, t2, p), t, reltol, 2.0)
    }
}

/// Hellmann (2013) ab initio N2-N2 potential: 5 sites, types A/B/C, C6 dispersion
/// only. Reproduces experimental B2 to high accuracy.
pub fn n2_hellmann() -> RigidLinear {
    let sites = vec![
        (-0.680065710389, 0, -832.77884541),
        (-0.447763006688, 1, 1601.24507755),
        (0.000000000000, 2, -1536.93246428),
        (0.447763006688, 1, 1601.24507755),
        (0.680065710389, 0, -832.77884541),
    ];
    let mut coeffs = HashMap::new();
    coeffs.insert((0, 0), [0.973347918383e7, 3.06144571072, 2.58031350518, 0.298807116692e7, 0.0]);
    coeffs.insert((0, 1), [-0.954555977809e7, 2.58710992361, 3.45760438302, -0.608284467163e7, 0.0]);
    coeffs.insert((0, 2), [0.122158259267e8, 2.96686681629, 2.46746232590, 0.490318811890e7, 0.0]);
    coeffs.insert((1, 1), [0.299460243665e7, 2.15319940621, 2.42577961527, 0.146889670654e8, 0.0]);
    coeffs.insert((1, 2), [-0.819908034347e7, 2.84661195657, 2.02508542307, -0.129841807274e8, 0.0]);
    coeffs.insert((2, 2), [0.163947777734e8, 2.99548316813, 1.97117981681, 0.107874613877e8, 0.0]);
    RigidLinear::new(sites, 3, coeffs)
}

/// Hellmann (2014) ab initio CO2-CO2 potential: 7 sites, types A/B/C/D, with C6
/// and C8 dispersion.
pub fn co2_hellmann() -> RigidLinear {
    let sites = vec![
        (-1.28741781626, 0, -197.417207828),
        (-1.18192825424, 1, 168.070083318),
        (-0.18607849166, 2, -2559.64083227),
        (0.00000000000, 3, 5177.97591356),
        (0.18607849166, 2, -2559.64083227),
        (1.18192825424, 1, 168.070083318),
        (1.28741781626, 0, -197.417207828),
    ];
    let mut coeffs = HashMap::new();
    coeffs.insert((0, 0), [-0.247910365353e7, 2.08319218048, 3.14980106637, -0.306747626563e8, 0.211522217149e9]);
    coeffs.insert((0, 1), [0.659160470472e7, 3.16681447768, 2.46903752251, 0.698469835305e8, -0.810638994730e9]);
    coeffs.insert((0, 2), [-0.197776308389e8, 2.46163539534, 1.57103563097, -0.143806191593e9, 0.355929066714e10]);
    coeffs.insert((0, 3), [0.384165630648e8, 2.48589087370, 1.89845841233, 0.121226824365e9, -0.286891373977e10]);
    coeffs.insert((1, 1), [-0.124570324466e7, 1.67813668662, 2.14451960163, -0.109398472925e9, 0.114677667224e10]);
    coeffs.insert((1, 2), [0.451317323034e8, 2.65969570294, 1.46843191121, 0.811702881095e8, -0.210805303525e10]);
    coeffs.insert((1, 3), [-0.116048612008e9, 2.77169644514, 4.14021127755, 0.263241896284e8, -0.173569859005e9]);
    coeffs.insert((2, 2), [-0.103079402689e8, 2.98535796569, 2.72634741238, 0.126349448908e9, -0.496759975158e9]);
    coeffs.insert((2, 3), [0.340824968085e8, 2.75870881239, 2.44815795987, -0.285769208067e9, 0.122323855871e10]);
    coeffs.insert((3, 3), [-0.915027698701e8, 2.87267355769, 2.27614875317, 0.551179708953e9, -0.131218053988e10]);
    RigidLinear::new(sites, 4, coeffs)
}

/// Generic two-centre Lennard-Jones (2CLJ): two identical LJ sites a distance
/// `bond` apart (no charges).
pub fn two_center_lj(eps: f64, sig: f64, bond: f64) -> Linear {
    let h = 0.5 * bond;
    Linear {
        sites: vec![
            Site { d: -h, eps, sig, q: 0.0 },
            Site { d: h, eps, sig, q: 0.0 },
        ],
    }
}

/// TraPPE nitrogen: two N LJ sites + a central charge for the quadrupole.
/// (sigma=3.31 A, eps/k=36 K, bond=1.10 A, q_N=-0.482 e, q_centre=+0.964 e.)
pub fn n2_trappe() -> Linear {
    let h = 0.55;
    Linear {
        sites: vec![
            Site { d: -h, eps: 36.0, sig: 3.31, q: -0.482 },
            Site { d: 0.0, eps: 0.0, sig: 0.0, q: 0.964 },
            Site { d: h, eps: 36.0, sig: 3.31, q: -0.482 },
        ],
    }
}

/// EPM2 carbon dioxide (Harris & Yung 1995): C + 2 O LJ sites with charges.
/// (sigma_C=2.757, eps_C/k=28.129; sigma_O=3.033, eps_O/k=80.507; l_CO=1.149 A;
/// q_C=+0.6512, q_O=-0.3256 e.)
pub fn co2_epm2() -> Linear {
    let l = 1.149;
    Linear {
        sites: vec![
            Site { d: -l, eps: 80.507, sig: 3.033, q: -0.3256 },
            Site { d: 0.0, eps: 28.129, sig: 2.757, q: 0.6512 },
            Site { d: l, eps: 80.507, sig: 3.033, q: -0.3256 },
        ],
    }
}
