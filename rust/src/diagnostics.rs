//! Independent checks of a subproblem solution.
//!
//! Port of `src/mmadual/diagnostics.py`.

use crate::approximations::Subproblem;
use crate::dual_solver::dual_function;
use ndarray::Array1;
use std::collections::HashMap;

/// `g_i(x) - b_i` for the approximate constraints.
pub fn constraint_values(sub: &Subproblem, x: &Array1<f64>) -> Array1<f64> {
    let inv_ux = (&sub.upp - x).mapv(|v| 1.0 / v);
    let inv_xl = (x - &sub.low).mapv(|v| 1.0 / v);
    sub.p.dot(&inv_ux) + sub.q.dot(&inv_xl) - &sub.b
}

/// Objective of the subproblem at `(x, y, z)`, without the GCMMA constant `r0`.
pub fn subproblem_objective(sub: &Subproblem, x: &Array1<f64>, y: &Array1<f64>, z: f64) -> f64 {
    let mut f = sub.a0 * z;
    for j in 0..sub.n() {
        f += sub.p0[j] / (sub.upp[j] - x[j]) + sub.q0[j] / (x[j] - sub.low[j]);
    }
    f + (&sub.c * y + 0.5 * &sub.d * y * y).sum()
}

/// Artificial variables `(y, z)` that minimize the objective for the given `x`.
pub fn optimal_artificial(sub: &Subproblem, x: &Array1<f64>) -> (Array1<f64>, f64) {
    let g = constraint_values(sub, x);
    let cost = |z: f64| -> (f64, Array1<f64>) {
        let y = Array1::from_iter(
            g.iter()
                .zip(sub.a.iter())
                .map(|(&gi, &ai)| (gi - ai * z).max(0.0)),
        );
        let c = sub.a0 * z + (&sub.c * &y + 0.5 * &sub.d * &y * &y).sum();
        (c, y)
    };
    if !sub.a.iter().any(|&ai| ai > 0.0) {
        return (g.mapv(|v| v.max(0.0)), 0.0);
    }
    let mut cand = vec![0.0];
    for i in 0..g.len() {
        if sub.a[i] > 0.0 && g[i] > 0.0 {
            cand.push(g[i] / sub.a[i]);
        }
    }
    if sub.d.iter().any(|&di| di > 0.0) {
        let zmax = cand.iter().cloned().fold(0.0_f64, f64::max);
        let mut lo = 0.0;
        let mut hi = zmax;
        let phi = (5.0_f64.sqrt() - 1.0) / 2.0;
        for _ in 0..200 {
            let z1 = hi - phi * (hi - lo);
            let z2 = lo + phi * (hi - lo);
            if cost(z1).0 < cost(z2).0 {
                hi = z2;
            } else {
                lo = z1;
            }
        }
        cand.push(0.5 * (lo + hi));
    }
    let mut best_z = 0.0;
    let mut best_c = f64::INFINITY;
    let mut best_y = Array1::zeros(g.len());
    for &zz in &cand {
        let (c, y) = cost(zz);
        if c < best_c {
            best_c = c;
            best_z = zz;
            best_y = y;
        }
    }
    (best_y, best_z)
}

/// Exact-penalty objective at `x` (upper bound of the optimal value).
pub fn exact_penalty_objective(sub: &Subproblem, x: &Array1<f64>) -> f64 {
    let (y, z) = optimal_artificial(sub, x);
    subproblem_objective(sub, x, &y, z)
}

/// Dual feasibility of `lam`.
pub fn dual_feasible(sub: &Subproblem, lam: &Array1<f64>, rtol: f64) -> bool {
    for i in 0..sub.m() {
        if lam[i] < -rtol * 1.0_f64.max(sub.c[i]) {
            return false;
        }
        if sub.d[i] == 0.0 && lam[i] > sub.c[i] * (1.0 + rtol) {
            return false;
        }
    }
    sub.a.dot(lam) <= sub.a0 * (1.0 + rtol) + rtol
}

/// Dual value `W(lam)`.
pub fn dual_value(sub: &Subproblem, lam: &Array1<f64>, check: bool) -> f64 {
    if check && !dual_feasible(sub, lam, 1e-12) {
        panic!("dual_value: lam is not dual feasible");
    }
    let lam_clip = Array1::from_iter(lam.iter().zip(sub.d.iter()).zip(sub.c.iter()).map(
        |((&l, &d), &c)| {
            let mx = if d > 0.0 { f64::INFINITY } else { c };
            l.clamp(0.0, mx)
        },
    ));
    dual_function(sub, &lam_clip).0
}

/// Gap `f_pen(x) - W(lam)` as `(absolute, relative)`.
pub fn duality_gap(sub: &Subproblem, x: &Array1<f64>, lam: &Array1<f64>) -> (f64, f64) {
    let f = exact_penalty_objective(sub, x);
    let g = f - dual_value(sub, lam, true);
    (g, g / 1.0_f64.max(f.abs()))
}

/// Gap of the returned primal point `(x, y, z)`.
pub fn duality_gap_returned(
    sub: &Subproblem,
    x: &Array1<f64>,
    y: &Array1<f64>,
    z: f64,
    lam: &Array1<f64>,
) -> (f64, f64, f64) {
    let g = constraint_values(sub, x) - &sub.a * z;
    let viol = g
        .iter()
        .zip(y.iter())
        .map(|(&gi, &yi)| (gi - yi).max(0.0))
        .fold(0.0_f64, f64::max);
    let f = subproblem_objective(sub, x, y, z);
    let gap = f - dual_value(sub, lam, true);
    (gap, gap / 1.0_f64.max(f.abs()), viol)
}

/// Residuals of the KKT conditions of the subproblem.
#[allow(clippy::too_many_arguments)]
pub fn kkt_residuals(
    sub: &Subproblem,
    x: &Array1<f64>,
    y: &Array1<f64>,
    z: f64,
    lam: &Array1<f64>,
    xsi: &Array1<f64>,
    eta: &Array1<f64>,
    mu: &Array1<f64>,
    zet: f64,
    s: &Array1<f64>,
) -> HashMap<&'static str, KktGroup> {
    let p = &sub.p0 + &sub.p.t().dot(lam);
    let q = &sub.q0 + &sub.q.t().dot(lam);
    let mut dpsi = Array1::<f64>::zeros(sub.n());
    for j in 0..sub.n() {
        dpsi[j] = p[j] / (sub.upp[j] - x[j]).powi(2) - q[j] / (x[j] - sub.low[j]).powi(2);
    }
    let g = constraint_values(sub, x) - &sub.a * z;
    let mut out = HashMap::new();
    out.insert(
        "stat_x",
        KktGroup::Vec(&dpsi - xsi + eta),
    );
    out.insert(
        "stat_y",
        KktGroup::Vec(&sub.c + &sub.d * y - mu - lam),
    );
    out.insert(
        "stat_z",
        KktGroup::Scalar(sub.a0 - zet - sub.a.dot(lam)),
    );
    out.insert("feas", KktGroup::Vec(&g - y + s));
    out.insert(
        "viol",
        KktGroup::Vec(Array1::from_iter(
            g.iter().zip(y.iter()).map(|(&gi, &yi)| (gi - yi).max(0.0)),
        )),
    );
    out.insert(
        "box",
        KktGroup::Pair(
            Array1::from_iter(
                sub.alfa
                    .iter()
                    .zip(x.iter())
                    .map(|(&a, &xj)| (a - xj).max(0.0)),
            ),
            Array1::from_iter(
                x.iter()
                    .zip(sub.beta.iter())
                    .map(|(&xj, &b)| (xj - b).max(0.0)),
            ),
        ),
    );
    out.insert(
        "dual",
        KktGroup::Multi(vec![
            lam.mapv(|v| (-v).max(0.0)),
            xsi.mapv(|v| (-v).max(0.0)),
            eta.mapv(|v| (-v).max(0.0)),
            mu.mapv(|v| (-v).max(0.0)),
            Array1::from(vec![(-zet).max(0.0)]),
            s.mapv(|v| (-v).max(0.0)),
        ]),
    );
    out.insert(
        "comp",
        KktGroup::Multi(vec![
            xsi * (x - &sub.alfa),
            eta * (&sub.beta - x),
            mu * y,
            Array1::from(vec![zet * z]),
            lam * s,
        ]),
    );
    out.insert(
        "proj_x",
        KktGroup::Vec(Array1::from_iter((0..sub.n()).map(|j| {
            let clipped = (x[j] - dpsi[j]).clamp(sub.alfa[j], sub.beta[j]);
            (x[j] - clipped).abs()
        }))),
    );
    out
}

/// One group of KKT residual values.
#[derive(Clone, Debug)]
pub enum KktGroup {
    /// Scalar residual.
    Scalar(f64),
    /// Vector residual.
    Vec(Array1<f64>),
    /// Pair of vector residuals.
    Pair(Array1<f64>, Array1<f64>),
    /// Several vector residuals.
    Multi(Vec<Array1<f64>>),
}

impl KktGroup {
    fn max_abs(&self) -> f64 {
        match self {
            KktGroup::Scalar(v) => v.abs(),
            KktGroup::Vec(v) => v.iter().map(|x| x.abs()).fold(0.0_f64, f64::max),
            KktGroup::Pair(a, b) => a
                .iter()
                .chain(b.iter())
                .map(|x| x.abs())
                .fold(0.0_f64, f64::max),
            KktGroup::Multi(vs) => vs
                .iter()
                .flat_map(|v| v.iter())
                .map(|x| x.abs())
                .fold(0.0_f64, f64::max),
        }
    }
}

/// Largest absolute value of each group of residuals.
pub fn kkt_summary(res: &HashMap<&'static str, KktGroup>) -> HashMap<&'static str, f64> {
    res.iter().map(|(&k, v)| (k, v.max_abs())).collect()
}

/// KKT summary plus duality gaps.
#[allow(clippy::too_many_arguments)]
pub fn check_solution(
    sub: &Subproblem,
    x: &Array1<f64>,
    y: &Array1<f64>,
    z: f64,
    lam: &Array1<f64>,
    xsi: &Array1<f64>,
    eta: &Array1<f64>,
    mu: &Array1<f64>,
    zet: f64,
    s: &Array1<f64>,
) -> HashMap<&'static str, f64> {
    let mut out = kkt_summary(&kkt_residuals(sub, x, y, z, lam, xsi, eta, mu, zet, s));
    out.insert("f_pen", exact_penalty_objective(sub, x));
    let (gap, gap_rel) = duality_gap(sub, x, lam);
    out.insert("gap", gap);
    out.insert("gap_rel", gap_rel);
    out.insert("F_returned", subproblem_objective(sub, x, y, z));
    let (gr, grr, _) = duality_gap_returned(sub, x, y, z, lam);
    out.insert("gap_returned", gr);
    out.insert("gap_returned_rel", grr);
    out
}
