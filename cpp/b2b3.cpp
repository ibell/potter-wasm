// C++ side-by-side: the SAME B2/B3 algorithm as the Rust core (transformed
// domain + adaptive Simpson + nested adaptive for B3), with the Lennard-Jones
// potential hard-coded (the C++ ergonomic baseline). Used to compare both the
// computed values and the wall-clock time against the Rust implementation.
//
//   build: clang++ -O3 -std=c++17 -o target/b2b3_cpp cpp/b2b3.cpp

#include <chrono>
#include <cmath>
#include <cstdio>

static inline double Vlj(double r) {
    // explicit multiplication (NOT std::pow) so the op count matches the Rust
    // hard-coded closure exactly — a fair language comparison.
    double inv = 1.0 / r;
    double s2 = inv * inv;
    double s6 = s2 * s2 * s2;
    return 4.0 * (s6 * s6 - s6); // eps = sig = 1
}

static inline double mayer(double r, double T) {
    double v = Vlj(r);
    if (!std::isfinite(v)) return -1.0;
    return std::exp(-v / T) - 1.0;
}

template <class F>
static double asr(F &f, double a, double b, double fa, double fb, double fm,
                  double whole, double tol, int depth) {
    double m = 0.5 * (a + b), lm = 0.5 * (a + m), rm = 0.5 * (m + b);
    double flm = f(lm), frm = f(rm);
    double left = (m - a) / 6.0 * (fa + 4.0 * flm + fm);
    double right = (b - m) / 6.0 * (fm + 4.0 * frm + fb);
    double delta = left + right - whole;
    if (depth <= 0 || std::fabs(delta) <= 15.0 * tol)
        return left + right + delta / 15.0;
    return asr(f, a, m, fa, fm, flm, left, 0.5 * tol, depth - 1) +
           asr(f, m, b, fm, fb, frm, right, 0.5 * tol, depth - 1);
}

template <class F>
static double adaptive_simpson(F f, double a, double b, double tol, int max_depth) {
    double m = 0.5 * (a + b), fa = f(a), fb = f(b), fm = f(m);
    double whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb);
    return asr(f, a, b, fa, fb, fm, whole, tol, max_depth);
}

static double b2(double T, double tol) {
    auto integ = [&](double s) -> double {
        double om = 1.0 - s;
        if (om <= 0.0) return 0.0;
        double r = s / om, jac = 1.0 / (om * om);
        double val = mayer(r, T) * r * r * jac;
        return std::isfinite(val) ? val : 0.0;
    };
    return -2.0 * M_PI * adaptive_simpson(integ, 0.0, 1.0, tol, 60);
}

static double b3(double T, double tol) {
    auto f = [&](double r) { return mayer(r, T); };
    auto outer = [&](double s1) -> double {
        double om1 = 1.0 - s1;
        if (om1 <= 0.0) return 0.0;
        double r1 = s1 / om1, j1 = 1.0 / (om1 * om1);
        double f1 = f(r1);
        auto mid = [&](double s2) -> double {
            double om2 = 1.0 - s2;
            if (om2 <= 0.0) return 0.0;
            double r2 = s2 / om2, j2 = 1.0 / (om2 * om2);
            double f2 = f(r2);
            double lo = std::fabs(r1 - r2), hi = r1 + r2;
            auto inner = [&](double r3) { return r3 * f(r3); };
            double i3 = adaptive_simpson(inner, lo, hi, tol, 28);
            double val = r2 * j2 * f2 * i3;
            return std::isfinite(val) ? val : 0.0;
        };
        double i2 = adaptive_simpson(mid, 0.0, 1.0, tol, 28);
        double val = r1 * j1 * f1 * i2;
        return std::isfinite(val) ? val : 0.0;
    };
    return -(8.0 * M_PI * M_PI / 3.0) * adaptive_simpson(outer, 0.0, 1.0, tol, 28);
}

int main() {
    using clk = std::chrono::high_resolution_clock;
    double T = 1.5, tol = 1e-7;

    printf("C++ (clang -O3) LJ B2/B3\n");
    for (double t : {1.0, 2.0, 5.0})
        printf("  B2(T*=%.1f)  = %.8f\n", t, b2(t, 1e-12));

    double val = 0.0, best = 1e30;
    for (int rep = 0; rep < 5; rep++) {
        auto t0 = clk::now();
        val = b3(T, tol);
        auto t1 = clk::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best) best = ms;
    }
    printf("  B3(T*=%.1f)  = %.8f   [%.1f ms]  (hard-coded potential)\n", T, val, best);
    return 0;
}
