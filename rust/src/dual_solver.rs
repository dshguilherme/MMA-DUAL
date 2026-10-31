//! Solution of the MMA/GCMMA subproblem by the dual method (Svanberg 1987, §5).
//!
//! Port of `src/mmadual/dual_solver.py`.

use crate::approximations::Subproblem;
use faer::linalg::solvers::SelfAdjointEigen;
use faer::{Mat, Side};
use ndarray::{Array1, Array2};

const LM_MU_MIN: f64 = 1e-12;
const LM_MU_MAX: f64 = 1e30;
const ARMIJO: f64 = 1e-4;
const ROUNDING_W: f64 = 1e-14;
const STAGNATION_W: f64 = 1e-13;
/// Relative distance to a bound below which a variable/multiplier counts as bounded.
pub const BOUND_TOL: f64 = 1e-12;
const ARC_HALVINGS: usize = 60;

/// Stopping / convergence report of [`solve_dual`].
#[derive(Clone, Debug)]
pub struct DualInfo {
    /// Projected-gradient tolerance met (and certificate holds when `a != 0`).
    pub converged: bool,
    /// `'tolerance' | 'rounding' | 'max_newton' | 'no_progress' | 'certification'`.
    pub reason: String,
    /// Accepted Newton/LM steps.
    pub newton_steps: usize,
    /// Dual-function evaluations.
    pub evaluations: usize,
    /// Largest projected gradient relative to constraint scales.
    pub residual: f64,
    /// Per-constraint scales `max(1, |b_i|, g_i(x(0)))`.
    pub scale: Array1<f64>,
    /// Half-space multiplier (0 when `a = 0`).
    pub z: f64,
    /// Whether the half-space is active.
    pub halfspace_active: bool,
    /// `F(x,y,z) - W(λ)` of the returned point.
    pub gap_returned: f64,
    /// Largest gap consistent with the tolerance.
    pub gap_allowed: f64,
    /// Whether `gap_returned <= gap_allowed` (and feasibility for `a != 0`).
    pub certified: bool,
}

/// Result of [`solve_dual`].
#[derive(Clone, Debug)]
pub struct SolveDualResult {
    /// Design variables.
    pub x: Array1<f64>,
    /// Artificial variables y (before recovery adjustment for `d = 0`).
    pub y: Array1<f64>,
    /// Artificial variable z.
    pub z: f64,
    /// Dual multipliers λ.
    pub lam: Array1<f64>,
    /// Iteration report.
    pub info: DualInfo,
}

/// Basic consistency of the subproblem.
pub fn check_subproblem(sub: &Subproblem) -> Result<(), String> {
    if sub.a.iter().any(|&ai| ai < 0.0) {
        return Err(
            "mmadual: the coefficients a_i must be >= 0 (Svanberg's form; the subproblem is not bounded otherwise)"
                .into(),
        );
    }
    if sub.a0 < 0.0 {
        return Err(
            "mmadual: a0 must be >= 0 (the subproblem is unbounded in z otherwise)".into(),
        );
    }
    if sub
        .low
        .iter()
        .zip(sub.alfa.iter())
        .any(|(&l, &a)| l >= a)
        || sub
            .beta
            .iter()
            .zip(sub.upp.iter())
            .any(|(&b, &u)| b >= u)
    {
        return Err("mmadual: the asymptotes must satisfy low < alfa and beta < upp".into());
    }
    Ok(())
}

/// Minimizer `x(λ)` of the Lagrangian in x, and coefficients `p(λ)`, `q(λ)`.
pub fn primal_x(sub: &Subproblem, lam: &Array1<f64>) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    // p = p0 + lam @ P  →  p0 + P^T lam
    let p = &sub.p0 + &sub.p.t().dot(lam);
    let q = &sub.q0 + &sub.q.t().dot(lam);
    let n = sub.n();
    let mut x = Array1::<f64>::zeros(n);
    for j in 0..n {
        let sp = p[j].sqrt();
        let sq = q[j].sqrt();
        let xj = (sp * sub.low[j] + sq * sub.upp[j]) / (sp + sq);
        x[j] = xj.clamp(sub.alfa[j], sub.beta[j]);
    }
    (x, p, q)
}

/// Minimizer `y(λ)` of the Lagrangian in y.
pub fn primal_y(sub: &Subproblem, lam: &Array1<f64>) -> Array1<f64> {
    Array1::from_iter(sub.d.iter().zip(lam.iter()).zip(sub.c.iter()).map(
        |((&d, &l), &c)| {
            if d > 0.0 {
                ((l - c) / d).max(0.0)
            } else {
                0.0
            }
        },
    ))
}

/// `W(λ)`, its gradient, and primal quantities at `λ` (`z = 0`).
pub fn dual_function(
    sub: &Subproblem,
    lam: &Array1<f64>,
) -> (f64, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let (x, p, q) = primal_x(sub, lam);
    let y = primal_y(sub, lam);
    let ux = &sub.upp - &x;
    let xl = &x - &sub.low;
    let mut w = 0.0;
    for j in 0..sub.n() {
        w += p[j] / ux[j] + q[j] / xl[j];
    }
    w -= lam.dot(&sub.b);
    w += (&sub.c * &y + 0.5 * &sub.d * &y * &y).sum();
    w -= lam.dot(&y);
    let inv_ux = ux.mapv(|v| 1.0 / v);
    let inv_xl = xl.mapv(|v| 1.0 / v);
    let grad = sub.p.dot(&inv_ux) + sub.q.dot(&inv_xl) - &y - &sub.b;
    (w, grad, x, y, p, q)
}

/// Analytic Hessian of W at `λ` (negative semidefinite).
pub fn dual_hessian(
    sub: &Subproblem,
    lam: &Array1<f64>,
    x: &Array1<f64>,
    p: &Array1<f64>,
    q: &Array1<f64>,
) -> Array2<f64> {
    let m = sub.m();
    let ux = &sub.upp - x;
    let xl = x - &sub.low;
    let width = &sub.beta - &sub.alfa;
    let free: Vec<bool> = (0..sub.n())
        .map(|j| {
            x[j] > sub.alfa[j] + BOUND_TOL * width[j] && x[j] < sub.beta[j] - BOUND_TOL * width[j]
        })
        .collect();
    let n_free = free.iter().filter(|&&f| f).count();
    let mut h = Array2::<f64>::zeros((m, m));
    if n_free > 0 {
        let mut g = Array2::<f64>::zeros((m, n_free));
        let mut dcol = Array1::<f64>::zeros(n_free);
        let mut k = 0;
        for j in 0..sub.n() {
            if !free[j] {
                continue;
            }
            for i in 0..m {
                g[[i, k]] = sub.p[[i, j]] / (ux[j] * ux[j]) - sub.q[[i, j]] / (xl[j] * xl[j]);
            }
            dcol[k] = 2.0 * p[j] / ux[j].powi(3) + 2.0 * q[j] / xl[j].powi(3);
            k += 1;
        }
        // H = -(G / D) @ G.T
        for k in 0..n_free {
            let inv_d = 1.0 / dcol[k];
            for i in 0..m {
                let gi = g[[i, k]] * inv_d;
                for r in 0..m {
                    h[[i, r]] -= gi * g[[r, k]];
                }
            }
        }
    }
    for i in 0..m {
        if sub.d[i] > 0.0 && lam[i] > sub.c[i] {
            h[[i, i]] -= 1.0 / sub.d[i];
        }
    }
    h
}

fn eigh_abs(hl: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    let n = hl.nrows();
    if n == 0 {
        return (Array1::zeros(0), Array2::zeros((0, 0)));
    }
    let mat = Mat::<f64>::from_fn(n, n, |i, j| hl[[i, j]]);
    let dec = SelfAdjointEigen::new(mat.as_ref(), Side::Lower)
        .expect("self-adjoint eigendecomposition");
    let mut w = Array1::<f64>::zeros(n);
    let mut v = Array2::<f64>::zeros((n, n));
    let sdiag = dec.S();
    let u = dec.U();
    for k in 0..n {
        w[k] = sdiag[k].abs();
        for i in 0..n {
            v[[i, k]] = u[(i, k)];
        }
    }
    (w, v)
}

fn scaled_spectrum(hl: &Array2<f64>) -> (Vec<bool>, Option<Array1<f64>>, Option<Array1<f64>>, Option<Array2<f64>>) {
    let n = hl.nrows();
    let diag: Vec<f64> = (0..n).map(|i| hl[[i, i]].abs()).collect();
    let curved: Vec<bool> = diag.iter().map(|&d| d > 0.0).collect();
    if !curved.iter().any(|&c| c) {
        return (curved, None, None, None);
    }
    let idx: Vec<usize> = curved
        .iter()
        .enumerate()
        .filter_map(|(i, &c)| c.then_some(i))
        .collect();
    let nc = idx.len();
    let mut s = Array1::<f64>::zeros(nc);
    let mut block = Array2::<f64>::zeros((nc, nc));
    for (a, &ia) in idx.iter().enumerate() {
        s[a] = 1.0 / diag[ia].sqrt();
        for (b, &ib) in idx.iter().enumerate() {
            block[[a, b]] = s[a] * hl[[ia, ib]] * s[b];
        }
    }
    let (w, v) = eigh_abs(&block);
    (curved, Some(s), Some(w), Some(v))
}

fn lm_step(
    gl: &Array1<f64>,
    curved: &[bool],
    s: Option<&Array1<f64>>,
    w: Option<&Array1<f64>>,
    v: Option<&Array2<f64>>,
    mu: f64,
) -> Array1<f64> {
    let n = gl.len();
    let mut step = Array1::<f64>::zeros(n);
    for i in 0..n {
        if !curved[i] {
            step[i] = gl[i] / (1.0 + mu);
        }
    }
    if let (Some(s), Some(w), Some(v)) = (s, w, v) {
        let idx: Vec<usize> = curved
            .iter()
            .enumerate()
            .filter_map(|(i, &c)| c.then_some(i))
            .collect();
        let nc = idx.len();
        let mut gs = Array1::<f64>::zeros(nc);
        for (a, &ia) in idx.iter().enumerate() {
            gs[a] = s[a] * gl[ia];
        }
        // V.T @ gs
        let mut vt_gs = Array1::<f64>::zeros(nc);
        for k in 0..nc {
            let mut acc = 0.0;
            for a in 0..nc {
                acc += v[[a, k]] * gs[a];
            }
            vt_gs[k] = acc;
        }
        let wmax = w.iter().cloned().fold(0.0_f64, f64::max).max(1e-300);
        for k in 0..nc {
            vt_gs[k] /= w[k] + mu * wmax;
        }
        // S * (V @ vt_gs)
        for (a, &ia) in idx.iter().enumerate() {
            let mut acc = 0.0;
            for k in 0..nc {
                acc += v[[a, k]] * vt_gs[k];
            }
            step[ia] = s[a] * acc;
        }
    }
    step
}

/// Projection of `v` onto `{0 ≤ λ ≤ lam_max, aᵀλ ≤ a0}`.
pub fn project_dual(v: &Array1<f64>, a: &Array1<f64>, a0: f64, lam_max: &Array1<f64>) -> Array1<f64> {
    let lam = Array1::from_iter(
        v.iter()
            .zip(lam_max.iter())
            .map(|(&vi, &mx)| vi.clamp(0.0, mx)),
    );
    if a.dot(&lam) <= a0 || !a.iter().any(|&ai| ai > 0.0) {
        return lam;
    }
    let mut t_lo = 0.0;
    let mut t_hi = 1.0;
    loop {
        let trial = Array1::from_iter(
            v.iter()
                .zip(a.iter())
                .zip(lam_max.iter())
                .map(|((&vi, &ai), &mx)| (vi - t_hi * ai).clamp(0.0, mx)),
        );
        if a.dot(&trial) <= a0 {
            break;
        }
        t_hi *= 2.0;
        if t_hi > 1e300 {
            break;
        }
    }
    for _ in 0..200 {
        let t = 0.5 * (t_lo + t_hi);
        let trial = Array1::from_iter(
            v.iter()
                .zip(a.iter())
                .zip(lam_max.iter())
                .map(|((&vi, &ai), &mx)| (vi - t * ai).clamp(0.0, mx)),
        );
        if a.dot(&trial) > a0 {
            t_lo = t;
        } else {
            t_hi = t;
        }
        if t_hi - t_lo <= 1e-15 * t_hi.max(1.0) {
            break;
        }
    }
    Array1::from_iter(
        v.iter()
            .zip(a.iter())
            .zip(lam_max.iter())
            .map(|((&vi, &ai), &mx)| (vi - t_hi * ai).clamp(0.0, mx)),
    )
}

fn projected_gradient(
    grad: &Array1<f64>,
    lam: &Array1<f64>,
    lam_max: &Array1<f64>,
    z: f64,
    a: &Array1<f64>,
) -> (Array1<f64>, Vec<bool>) {
    let g = grad - z * a;
    let mut active = vec![false; g.len()];
    let mut out = Array1::<f64>::zeros(g.len());
    for i in 0..g.len() {
        let act = (lam[i] <= 0.0 && g[i] < 0.0) || (lam[i] >= lam_max[i] && g[i] > 0.0);
        active[i] = act;
        out[i] = if act { 0.0 } else { g[i] };
    }
    (out, active)
}

fn halfspace_multiplier(
    grad: &Array1<f64>,
    lower: &[bool],
    upper: &[bool],
    a: &Array1<f64>,
) -> f64 {
    if !a.iter().any(|&ai| ai > 0.0) {
        return 0.0;
    }
    let dr2 = |z: f64| -> f64 {
        let mut sum = 0.0;
        for i in 0..grad.len() {
            let gi = grad[i] - z * a[i];
            let contrib = if lower[i] {
                gi > 0.0
            } else if upper[i] {
                gi < 0.0
            } else {
                true
            };
            if contrib {
                sum += a[i] * gi;
            }
        }
        -2.0 * sum
    };
    if dr2(0.0) >= 0.0 {
        return 0.0;
    }
    let mut z_hi: f64 = 1.0;
    for i in 0..grad.len() {
        if a[i] > 0.0 {
            z_hi = f64::max(z_hi, grad[i] / a[i] + 1.0);
        }
    }
    while dr2(z_hi) < 0.0 {
        z_hi *= 2.0;
    }
    let mut z_lo = 0.0;
    for _ in 0..200 {
        let z = 0.5 * (z_lo + z_hi);
        if dr2(z) < 0.0 {
            z_lo = z;
        } else {
            z_hi = z;
        }
        if z_hi - z_lo <= 1e-15 * f64::max(z_hi, 1.0) {
            break;
        }
    }
    0.5 * (z_lo + z_hi)
}

fn certificate(
    sub: &Subproblem,
    lam: &Array1<f64>,
    w: f64,
    grad: &Array1<f64>,
    x: &Array1<f64>,
    y: &Array1<f64>,
    z: f64,
    scale: &Array1<f64>,
    tol: f64,
    eps: &Array1<f64>,
    eps_plane: f64,
) -> (f64, f64, f64) {
    let g = grad + y;
    let u = &g - &sub.a * z;
    let y_ret = Array1::from_iter(
        sub.d
            .iter()
            .zip(y.iter())
            .zip(u.iter())
            .map(|((&d, &yi), &ui)| if d > 0.0 { yi } else { ui.max(0.0) }),
    );
    let viol = u
        .iter()
        .zip(y_ret.iter())
        .map(|(&ui, &yi)| (ui - yi).max(0.0))
        .fold(0.0_f64, f64::max);
    let mut f = sub.a0 * z;
    for j in 0..sub.n() {
        f += sub.p0[j] / (sub.upp[j] - x[j]) + sub.q0[j] / (x[j] - sub.low[j]);
    }
    f += (&sub.c * &y_ret + 0.5 * &sub.d * &y_ret * &y_ret).sum();
    let gap = f - w;
    let mut allowed = ROUNDING_W * (f.abs() + w.abs()) + eps_plane * z;
    for i in 0..sub.m() {
        allowed += tol * (sub.c[i] + sub.d[i] * y_ret[i]).max(lam[i]) * scale[i];
        allowed += eps[i] * u[i].abs();
    }
    (gap, allowed, viol)
}

fn flat_step(
    g: &Array1<f64>,
    lam: &Array1<f64>,
    c: &Array1<f64>,
    d: &Array1<f64>,
    mu: f64,
) -> Array1<f64> {
    let damp = 1.0_f64.max(mu);
    Array1::from_iter(g.iter().zip(lam.iter()).zip(c.iter()).zip(d.iter()).map(
        |(((&gi, &li), &ci), &di)| {
            let dist = if gi > 0.0 {
                if li < ci {
                    ci - li
                } else {
                    di * gi.abs()
                }
            } else {
                li
            };
            gi.signum() * dist / damp
        },
    ))
}

fn lam_max_vec(sub: &Subproblem) -> Array1<f64> {
    Array1::from_iter(
        sub.d
            .iter()
            .zip(sub.c.iter())
            .map(|(&d, &c)| if d > 0.0 { f64::INFINITY } else { c }),
    )
}

fn nearly_equal(a: &Array1<f64>, b: &Array1<f64>) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(&x, &y)| x == y)
}

fn solve_box(sub: &Subproblem, tol: f64, max_newton: usize, warn: bool) -> SolveDualResult {
    let m = sub.m();
    let lam_max = lam_max_vec(sub);
    let mut lam = Array1::<f64>::zeros(m);
    let (mut w, mut grad, mut x, mut y, mut p, mut q) = dual_function(sub, &lam);
    let scale = Array1::from_iter(
        sub.b
            .iter()
            .zip(grad.iter())
            .map(|(&bi, &gi)| 1.0_f64.max(bi.abs()).max(gi + bi)),
    );
    let mut info = DualInfo {
        converged: false,
        reason: "max_newton".into(),
        newton_steps: 0,
        evaluations: 1,
        residual: f64::INFINITY,
        scale: scale.clone(),
        z: 0.0,
        halfspace_active: false,
        gap_returned: 0.0,
        gap_allowed: 0.0,
        certified: false,
    };
    let mut mu = LM_MU_MIN;
    let mut stagnant = 0usize;
    let mut last_trial: Option<Array1<f64>> = None;
    let mut w_new = w;
    let mut grad_new = grad.clone();
    let mut x_new = x.clone();
    let mut y_new = y.clone();
    let mut p_new = p.clone();
    let mut q_new = q.clone();
    #[allow(unused_assignments)]
    let mut lam_new = Array1::<f64>::zeros(m);

    for it in 0..=max_newton {
        let (gp, active) = projected_gradient(&grad, &lam, &lam_max, 0.0, &sub.a);
        let free: Vec<bool> = active.iter().map(|&a| !a).collect();
        info.residual = gp
            .iter()
            .zip(scale.iter())
            .map(|(&g, &s)| (g / s).abs())
            .fold(0.0_f64, f64::max);
        if info.residual <= tol {
            info.converged = true;
            info.reason = "tolerance".into();
            break;
        }
        if it == max_newton {
            break;
        }
        let free_idx: Vec<usize> = free
            .iter()
            .enumerate()
            .filter_map(|(i, &f)| f.then_some(i))
            .collect();
        let nf = free_idx.len();
        let h_full = dual_hessian(sub, &lam, &x, &p, &q);
        let mut h = Array2::<f64>::zeros((nf, nf));
        let mut g_free = Array1::<f64>::zeros(nf);
        for (a, &ia) in free_idx.iter().enumerate() {
            g_free[a] = grad[ia];
            for (b, &ib) in free_idx.iter().enumerate() {
                h[[a, b]] = h_full[[ia, ib]];
            }
        }
        let (curved, s_opt, w_opt, v_opt) = scaled_spectrum(&h);
        loop {
            let mut step = Array1::<f64>::zeros(m);
            let mut step_free = lm_step(
                &g_free,
                &curved,
                s_opt.as_ref(),
                w_opt.as_ref(),
                v_opt.as_ref(),
                mu,
            );
            if curved.iter().any(|&c| !c) {
                let mut g_flat = Array1::<f64>::zeros(nf);
                let mut lam_flat = Array1::<f64>::zeros(nf);
                let mut c_flat = Array1::<f64>::zeros(nf);
                let mut d_flat = Array1::<f64>::zeros(nf);
                for (a, &ia) in free_idx.iter().enumerate() {
                    if !curved[a] {
                        g_flat[a] = g_free[a];
                        lam_flat[a] = lam[ia];
                        c_flat[a] = sub.c[ia];
                        d_flat[a] = sub.d[ia];
                    }
                }
                let flat = flat_step(&g_flat, &lam_flat, &c_flat, &d_flat, mu);
                for (a, _) in free_idx.iter().enumerate() {
                    if !curved[a] {
                        step_free[a] = flat[a];
                    }
                }
            }
            for (a, &ia) in free_idx.iter().enumerate() {
                step[ia] = step_free[a];
            }
            if !step.iter().all(|v| v.is_finite()) || step.dot(&gp) <= 0.0 {
                step = gp.clone();
            }
            lam_new = Array1::from_iter(
                lam.iter()
                    .zip(step.iter())
                    .zip(lam_max.iter())
                    .map(|((&l, &s), &mx)| (l + s).clamp(0.0, mx)),
            );
            let evaluate = last_trial
                .as_ref()
                .map(|t| !nearly_equal(&lam_new, t))
                .unwrap_or(true);
            if evaluate {
                last_trial = Some(lam_new.clone());
                let out = dual_function(sub, &lam_new);
                w_new = out.0;
                grad_new = out.1;
                x_new = out.2;
                y_new = out.3;
                p_new = out.4;
                q_new = out.5;
                info.evaluations += 1;
                let pred = grad.dot(&(&lam_new - &lam));
                if w_new - w >= ARMIJO * pred - ROUNDING_W * w.abs() || nearly_equal(&lam_new, &lam)
                {
                    break;
                }
            }
            if mu >= LM_MU_MAX {
                lam_new = lam.clone();
                break;
            }
            mu *= 10.0;
        }
        if nearly_equal(&lam_new, &lam) {
            info.reason = "no_progress".into();
            break;
        }
        stagnant = if w_new - w <= STAGNATION_W * w.abs() && mu <= LM_MU_MIN {
            stagnant + 1
        } else {
            0
        };
        mu = (mu / 100.0).max(LM_MU_MIN);
        lam = lam_new.clone();
        w = w_new;
        grad = grad_new.clone();
        x = x_new.clone();
        y = y_new.clone();
        p = p_new.clone();
        q = q_new.clone();
        last_trial = None;
        info.newton_steps += 1;
        if stagnant >= 2 {
            info.reason = "rounding".into();
            break;
        }
    }
    let eps = Array1::zeros(m);
    let (gap, allowed, _) = certificate(sub, &lam, w, &grad, &x, &y, 0.0, &scale, tol, &eps, 0.0);
    info.gap_returned = gap;
    info.gap_allowed = allowed;
    info.certified = gap <= allowed;
    if !info.converged {
        let (gp, _) = projected_gradient(&grad, &lam, &lam_max, 0.0, &sub.a);
        info.residual = gp
            .iter()
            .zip(scale.iter())
            .map(|(&g, &s)| (g / s).abs())
            .fold(0.0_f64, f64::max);
        if warn && info.residual > tol.sqrt() {
            eprintln!(
                "mmadual: dual iteration stopped ({}) after {} steps with projected gradient {:.1e} (tol = {:.1e}); the subproblem solution may be less accurate than requested",
                info.reason, info.newton_steps, info.residual, tol
            );
        }
    }
    SolveDualResult {
        x,
        y,
        z: 0.0,
        lam,
        info,
    }
}

fn snap(
    mut lam: Array1<f64>,
    lam_max: &Array1<f64>,
    eps: &Array1<f64>,
    a: &Array1<f64>,
    a0: f64,
) -> Array1<f64> {
    for i in 0..lam.len() {
        if lam[i] <= eps[i] {
            lam[i] = 0.0;
        }
    }
    let mut trial = lam.clone();
    let mut any_up = false;
    for i in 0..lam.len() {
        if lam[i] >= lam_max[i] - eps[i] {
            trial[i] = lam_max[i];
            any_up = true;
        }
    }
    if any_up && a.dot(&trial) <= a0 {
        lam = trial;
    }
    lam
}

fn ratio(
    lam: &Array1<f64>,
    s: &Array1<f64>,
    lam_max: &Array1<f64>,
    fixed: &[bool],
    hold_plane: bool,
    a: &Array1<f64>,
    a0: f64,
) -> f64 {
    let mut alpha = f64::INFINITY;
    for i in 0..lam.len() {
        if fixed[i] {
            continue;
        }
        if s[i] < 0.0 {
            alpha = alpha.min(lam[i] / (-s[i]));
        }
        if s[i] > 0.0 && lam_max[i].is_finite() {
            alpha = alpha.min((lam_max[i] - lam[i]) / s[i]);
        }
    }
    if !hold_plane {
        let a_s = a.dot(s);
        if a_s > 0.0 {
            alpha = alpha.min((a0 - a.dot(lam)).max(0.0) / a_s);
        }
    }
    alpha
}

fn to_first_bound(
    lam: &Array1<f64>,
    s: &Array1<f64>,
    lam_max: &Array1<f64>,
    fixed: &[bool],
    hold_plane: bool,
    a: &Array1<f64>,
    a0: f64,
) -> Array1<f64> {
    let alpha = ratio(lam, s, lam_max, fixed, hold_plane, a, a0);
    if alpha.is_finite() && alpha > 0.0 {
        s * alpha
    } else {
        s.clone()
    }
}

/// Orthonormal basis of `{u : afᵀ u = 0}` as columns of an `(k, k-1)` matrix (SVD of a row).
fn nullspace_row(af: &Array1<f64>) -> Array2<f64> {
    let k = af.len();
    if k <= 1 {
        return Array2::zeros((k, 0));
    }
    // Householder reflector of af to e0 direction, then take columns 1..k-1 of Q.
    let mut v = af.to_owned();
    let norm = v.dot(&v).sqrt();
    if norm == 0.0 {
        let mut n = Array2::<f64>::zeros((k, k - 1));
        for j in 0..k - 1 {
            n[[j + 1, j]] = 1.0;
        }
        return n;
    }
    v /= norm;
    if v[0] >= 0.0 {
        v[0] += 1.0;
    } else {
        v[0] -= 1.0;
    }
    let beta = 2.0 / v.dot(&v);
    // Q = I - beta v v^T; columns 1.. are the nullspace (first maps to ±e0)
    let mut q = Array2::<f64>::eye(k);
    for i in 0..k {
        for j in 0..k {
            q[[i, j]] -= beta * v[i] * v[j];
        }
    }
    // Orient so Q[:,0] is parallel to af (optional); take remaining columns.
    let mut n = Array2::<f64>::zeros((k, k - 1));
    for j in 0..k - 1 {
        for i in 0..k {
            n[[i, j]] = q[[i, j + 1]];
        }
    }
    n
}

fn tangent_direction(
    h: &Array2<f64>,
    grad: &Array1<f64>,
    lam: &Array1<f64>,
    lam_max: &Array1<f64>,
    free: &[bool],
    a: &Array1<f64>,
    a0: f64,
    on_plane: bool,
    mu: f64,
) -> Array1<f64> {
    let m = grad.len();
    let mut s = Array1::<f64>::zeros(m);
    let free_idx: Vec<usize> = free
        .iter()
        .enumerate()
        .filter_map(|(i, &f)| f.then_some(i))
        .collect();
    if free_idx.is_empty() {
        return s;
    }
    let nf = free_idx.len();
    let mut hf = Array2::<f64>::zeros((nf, nf));
    let mut gf = Array1::<f64>::zeros(nf);
    let mut af = Array1::<f64>::zeros(nf);
    for (a_i, &ia) in free_idx.iter().enumerate() {
        gf[a_i] = grad[ia];
        af[a_i] = a[ia];
        for (b_i, &ib) in free_idx.iter().enumerate() {
            hf[[a_i, b_i]] = h[[ia, ib]];
        }
    }
    let n_basis = if on_plane && af.iter().any(|&v| v > 0.0) {
        Some(nullspace_row(&af))
    } else {
        None
    };
    let (hf_r, gf_r) = if let Some(ref n) = n_basis {
        let nr = n.ncols();
        let mut hr = Array2::<f64>::zeros((nr, nr));
        let mut gr = Array1::<f64>::zeros(nr);
        // gf_r = N.T @ gf; Hf_r = N.T @ Hf @ N
        for p in 0..nr {
            for i in 0..nf {
                gr[p] += n[[i, p]] * gf[i];
            }
        }
        let mut hn = Array2::<f64>::zeros((nf, nr));
        for i in 0..nf {
            for p in 0..nr {
                for j in 0..nf {
                    hn[[i, p]] += hf[[i, j]] * n[[j, p]];
                }
            }
        }
        for p in 0..nr {
            for q in 0..nr {
                for i in 0..nf {
                    hr[[p, q]] += n[[i, p]] * hn[[i, q]];
                }
            }
        }
        (hr, gr)
    } else {
        (hf, gf.clone())
    };
    let (curved, s_opt, w_opt, v_opt) = scaled_spectrum(&hf_r);
    let mut sf = lm_step(
        &gf_r,
        &curved,
        s_opt.as_ref(),
        w_opt.as_ref(),
        v_opt.as_ref(),
        mu,
    );
    if curved.iter().any(|&c| !c) {
        let mut flat = Array1::<f64>::zeros(gf_r.len());
        for i in 0..gf_r.len() {
            if !curved[i] {
                flat[i] = gf_r[i];
            }
        }
        let mut f = Array1::<f64>::zeros(m);
        if let Some(ref n) = n_basis {
            let mut f_free = Array1::<f64>::zeros(nf);
            for i in 0..nf {
                for p in 0..n.ncols() {
                    f_free[i] += n[[i, p]] * flat[p];
                }
            }
            for (a_i, &ia) in free_idx.iter().enumerate() {
                f[ia] = f_free[a_i];
            }
        } else {
            for (a_i, &ia) in free_idx.iter().enumerate() {
                f[ia] = flat[a_i];
            }
        }
        let fixed: Vec<bool> = free.iter().map(|&f| !f).collect();
        let alpha = ratio(lam, &f, lam_max, &fixed, on_plane, a, a0);
        let scale = if alpha.is_finite() && alpha > 0.0 {
            alpha
        } else {
            1.0
        } / 1.0_f64.max(mu);
        for i in 0..sf.len() {
            if !curved[i] {
                sf[i] = gf_r[i] * scale;
            }
        }
    }
    if let Some(ref n) = n_basis {
        let mut s_free = Array1::<f64>::zeros(nf);
        for i in 0..nf {
            for p in 0..n.ncols() {
                s_free[i] += n[[i, p]] * sf[p];
            }
        }
        for (a_i, &ia) in free_idx.iter().enumerate() {
            s[ia] = s_free[a_i];
        }
    } else {
        for (a_i, &ia) in free_idx.iter().enumerate() {
            s[ia] = sf[a_i];
        }
    }
    s
}

fn feasible_direction<F>(
    mut direction: F,
    lam: &Array1<f64>,
    lam_max: &Array1<f64>,
    gp: &Array1<f64>,
    active: &[bool],
    lower: &[bool],
    upper: &[bool],
    plane_active: bool,
    on_plane: bool,
    a: &Array1<f64>,
    a0: f64,
) -> (Array1<f64>, Vec<bool>, bool)
where
    F: FnMut(&[bool], bool) -> Array1<f64>,
{
    let mut fixed = active.to_vec();
    let mut hold_plane = plane_active;
    loop {
        let free: Vec<bool> = fixed.iter().map(|&f| !f).collect();
        let mut s = direction(&free, hold_plane);
        if !s.iter().all(|v| v.is_finite()) || s.dot(gp) <= 0.0 {
            let gstep = Array1::from_iter(
                free
                    .iter()
                    .zip(gp.iter())
                    .map(|(&f, &g)| if f { g } else { 0.0 }),
            );
            s = to_first_bound(lam, &gstep, lam_max, &fixed, hold_plane, a, a0);
        }
        let mut blocked_any = false;
        for i in 0..s.len() {
            if free[i] && ((lower[i] && s[i] < 0.0) || (upper[i] && s[i] > 0.0)) {
                fixed[i] = true;
                blocked_any = true;
            }
        }
        let block_plane = (!hold_plane) && on_plane && a.dot(&s) > 0.0;
        if !blocked_any && !block_plane {
            return (s, fixed, hold_plane);
        }
        hold_plane = hold_plane || block_plane;
        if fixed.iter().all(|&f| f) {
            return (Array1::zeros(s.len()), fixed, hold_plane);
        }
    }
}

fn solve_polytope(sub: &Subproblem, tol: f64, max_newton: usize, warn: bool) -> SolveDualResult {
    let m = sub.m();
    let lam_max = lam_max_vec(sub);
    let eps = sub.c.mapv(|c| BOUND_TOL * 1.0_f64.max(c));
    let eps_plane = BOUND_TOL * 1.0_f64.max(sub.a0);
    let mut lam = Array1::<f64>::zeros(m);
    let (mut w, mut grad, mut x, mut y, mut p, mut q) = dual_function(sub, &lam);
    let scale = Array1::from_iter(
        sub.b
            .iter()
            .zip(grad.iter())
            .map(|(&bi, &gi)| 1.0_f64.max(bi.abs()).max(gi + bi)),
    );
    let mut info = DualInfo {
        converged: false,
        reason: "max_newton".into(),
        newton_steps: 0,
        evaluations: 1,
        residual: f64::INFINITY,
        scale: scale.clone(),
        z: 0.0,
        halfspace_active: false,
        gap_returned: 0.0,
        gap_allowed: 0.0,
        certified: false,
    };
    let mut mu = LM_MU_MIN;
    let mut stagnant = 0usize;
    #[allow(unused_assignments)]
    let mut z = 0.0;
    let mut last_trial: Option<Array1<f64>> = None;
    let mut w_new = w;
    let mut grad_new = grad.clone();
    let mut x_new = x.clone();
    let mut y_new = y.clone();
    let mut p_new = p.clone();
    let mut q_new = q.clone();
    #[allow(unused_assignments)]
    let mut lam_new = Array1::<f64>::zeros(m);

    let trial = |lam: &Array1<f64>, s: &Array1<f64>, alpha: f64| -> Array1<f64> {
        let v = lam + &(s * alpha);
        snap(project_dual(&v, &sub.a, sub.a0, &lam_max), &lam_max, &eps, &sub.a, sub.a0)
    };

    for it in 0..=max_newton {
        let lower: Vec<bool> = lam.iter().zip(eps.iter()).map(|(&l, &e)| l <= e).collect();
        let upper: Vec<bool> = lam
            .iter()
            .zip(lam_max.iter())
            .zip(eps.iter())
            .map(|((&l, &mx), &e)| l >= mx - e)
            .collect();
        let on_plane = sub.a0 - sub.a.dot(&lam) <= eps_plane;
        z = if on_plane {
            halfspace_multiplier(&grad, &lower, &upper, &sub.a)
        } else {
            0.0
        };
        let plane_active = z > 0.0;
        let u = &grad - z * &sub.a;
        let mut active = vec![false; m];
        let mut gp = Array1::<f64>::zeros(m);
        for i in 0..m {
            let act = (lower[i] && u[i] < 0.0) || (upper[i] && u[i] > 0.0);
            active[i] = act;
            gp[i] = if act { 0.0 } else { u[i] };
        }
        let r = project_dual(&(&lam + &grad), &sub.a, sub.a0, &lam_max) - &lam;
        info.residual = gp
            .iter()
            .zip(scale.iter())
            .map(|(&g, &s)| (g / s).abs())
            .chain(r.iter().zip(scale.iter()).map(|(&g, &s)| (g / s).abs()))
            .fold(0.0_f64, f64::max);
        info.z = z;
        info.halfspace_active = plane_active;
        if info.residual <= tol {
            info.reason = "tolerance".into();
            break;
        }
        if it == max_newton {
            break;
        }
        let h = dual_hessian(sub, &lam, &x, &p, &q);
        let mut accepted = false;
        loop {
            let (s, fixed, hold_plane) = feasible_direction(
                |free, hold| {
                    tangent_direction(&h, &grad, &lam, &lam_max, free, &sub.a, sub.a0, hold, mu)
                },
                &lam,
                &lam_max,
                &gp,
                &active,
                &lower,
                &upper,
                plane_active,
                on_plane,
                &sub.a,
                sub.a0,
            );
            let alpha = if s.iter().any(|&v| v != 0.0) {
                ratio(&lam, &s, &lam_max, &fixed, hold_plane, &sub.a, sub.a0).min(1.0)
            } else {
                0.0
            };
            if alpha > 0.0 {
                lam_new = trial(&lam, &s, alpha);
                let evaluate = last_trial
                    .as_ref()
                    .map(|t| !nearly_equal(&lam_new, t))
                    .unwrap_or(true);
                if evaluate {
                    last_trial = Some(lam_new.clone());
                    let out = dual_function(sub, &lam_new);
                    w_new = out.0;
                    grad_new = out.1;
                    x_new = out.2;
                    y_new = out.3;
                    p_new = out.4;
                    q_new = out.5;
                    info.evaluations += 1;
                    let pred = grad.dot(&(&lam_new - &lam));
                    if pred > 0.0 && w_new - w >= ARMIJO * pred - ROUNDING_W * w.abs() {
                        accepted = true;
                        break;
                    }
                }
            }
            if mu >= LM_MU_MAX || fixed.iter().all(|&f| f) {
                break;
            }
            mu *= 10.0;
        }
        if !accepted {
            let mut gpt = gp.clone();
            let free_a: Vec<usize> = active
                .iter()
                .enumerate()
                .filter_map(|(i, &a)| (!a).then_some(i))
                .collect();
            if plane_active && free_a.iter().any(|&i| sub.a[i] > 0.0) {
                let mut af_dot_af = 0.0;
                let mut af_dot_uf = 0.0;
                for &i in &free_a {
                    af_dot_af += sub.a[i] * sub.a[i];
                    af_dot_uf += sub.a[i] * gp[i];
                }
                if af_dot_af > 0.0 {
                    let coeff = af_dot_uf / af_dot_af;
                    for &i in &free_a {
                        gpt[i] = gp[i] - sub.a[i] * coeff;
                    }
                }
            }
            let (s, fixed, hold_plane) = feasible_direction(
                |free, _hold| {
                    Array1::from_iter(
                        free
                            .iter()
                            .zip(gpt.iter())
                            .map(|(&f, &g)| if f { g } else { 0.0 }),
                    )
                },
                &lam,
                &lam_max,
                &gp,
                &active,
                &lower,
                &upper,
                plane_active,
                on_plane,
                &sub.a,
                sub.a0,
            );
            let mut alpha = if s.iter().any(|&v| v != 0.0) {
                ratio(&lam, &s, &lam_max, &fixed, hold_plane, &sub.a, sub.a0)
            } else {
                0.0
            };
            if !alpha.is_finite() {
                alpha = 1.0;
            }
            for _ in 0..ARC_HALVINGS {
                if alpha <= 0.0 {
                    break;
                }
                lam_new = trial(&lam, &s, alpha);
                let out = dual_function(sub, &lam_new);
                w_new = out.0;
                grad_new = out.1;
                x_new = out.2;
                y_new = out.3;
                p_new = out.4;
                q_new = out.5;
                info.evaluations += 1;
                let pred = grad.dot(&(&lam_new - &lam));
                if pred > 0.0 && w_new - w >= ARMIJO * pred - ROUNDING_W * w.abs() {
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }
        }
        if !accepted {
            info.reason = "no_progress".into();
            break;
        }
        stagnant = if w_new - w <= STAGNATION_W * w.abs() && mu <= LM_MU_MIN {
            stagnant + 1
        } else {
            0
        };
        mu = (mu / 100.0).max(LM_MU_MIN);
        lam = lam_new.clone();
        w = w_new;
        grad = grad_new.clone();
        x = x_new.clone();
        y = y_new.clone();
        p = p_new.clone();
        q = q_new.clone();
        last_trial = None;
        info.newton_steps += 1;
        if stagnant >= 2 {
            info.reason = "rounding".into();
            break;
        }
    }

    let lower: Vec<bool> = lam.iter().zip(eps.iter()).map(|(&l, &e)| l <= e).collect();
    let upper: Vec<bool> = lam
        .iter()
        .zip(lam_max.iter())
        .zip(eps.iter())
        .map(|((&l, &mx), &e)| l >= mx - e)
        .collect();
    let on_plane = sub.a0 - sub.a.dot(&lam) <= eps_plane;
    z = if on_plane {
        halfspace_multiplier(&grad, &lower, &upper, &sub.a)
    } else {
        0.0
    };
    let u = &grad - z * &sub.a;
    let gp = Array1::from_iter((0..m).map(|i| {
        if (lower[i] && u[i] < 0.0) || (upper[i] && u[i] > 0.0) {
            0.0
        } else {
            u[i]
        }
    }));
    let r = project_dual(&(&lam + &grad), &sub.a, sub.a0, &lam_max) - &lam;
    info.residual = gp
        .iter()
        .zip(scale.iter())
        .map(|(&g, &s)| (g / s).abs())
        .chain(r.iter().zip(scale.iter()).map(|(&g, &s)| (g / s).abs()))
        .fold(0.0_f64, f64::max);
    info.z = z;
    info.halfspace_active = z > 0.0;
    let (gap, allowed, viol) =
        certificate(sub, &lam, w, &grad, &x, &y, z, &scale, tol, &eps, eps_plane);
    info.gap_returned = gap;
    info.gap_allowed = allowed;
    info.certified = gap <= allowed && viol <= tol * scale.iter().cloned().fold(0.0_f64, f64::max);
    if info.reason == "tolerance" {
        if info.certified {
            info.converged = true;
        } else {
            info.reason = "certification".into();
        }
    }
    if !info.converged
        && warn
        && (info.residual > tol.sqrt() || info.reason == "certification")
    {
        eprintln!(
            "mmadual: dual iteration stopped ({}) after {} steps with projected gradient {:.1e} (tol = {:.1e}) and returned-point gap {:.1e} (allowed {:.1e}); the subproblem solution may be less accurate than requested",
            info.reason, info.newton_steps, info.residual, tol, gap, allowed
        );
    }
    SolveDualResult {
        x,
        y,
        z,
        lam,
        info,
    }
}

/// Maximize the dual function; returns the primal/dual point and an [`DualInfo`] report.
pub fn solve_dual(sub: &Subproblem, tol: f64, max_newton: usize, warn: bool) -> SolveDualResult {
    check_subproblem(sub).expect("invalid subproblem");
    if sub.a.iter().any(|&ai| ai > 0.0) {
        solve_polytope(sub, tol, max_newton, warn)
    } else {
        solve_box(sub, tol, max_newton, warn)
    }
}

/// Multipliers of the bounds and slacks for mmapy-compatible return tuples.
pub fn recover_multipliers(
    sub: &Subproblem,
    x: &Array1<f64>,
    y: &Array1<f64>,
    z: f64,
    lam: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>, f64, Array1<f64>, Array1<f64>) {
    let p = &sub.p0 + &sub.p.t().dot(lam);
    let q = &sub.q0 + &sub.q.t().dot(lam);
    let mut xsi = Array1::<f64>::zeros(sub.n());
    let mut eta = Array1::<f64>::zeros(sub.n());
    for j in 0..sub.n() {
        let dpsi = p[j] / (sub.upp[j] - x[j]).powi(2) - q[j] / (x[j] - sub.low[j]).powi(2);
        xsi[j] = dpsi.max(0.0);
        eta[j] = (-dpsi).max(0.0);
    }
    let inv_ux = (&sub.upp - x).mapv(|v| 1.0 / v);
    let inv_xl = (x - &sub.low).mapv(|v| 1.0 / v);
    let g = sub.p.dot(&inv_ux) + sub.q.dot(&inv_xl) - &sub.a * z - &sub.b;
    let y_out = Array1::from_iter(
        sub.d
            .iter()
            .zip(y.iter())
            .zip(g.iter())
            .map(|((&d, &yi), &gi)| if d > 0.0 { yi } else { gi.max(0.0) }),
    );
    let mu = Array1::from_iter(
        sub.c
            .iter()
            .zip(sub.d.iter())
            .zip(y_out.iter())
            .zip(lam.iter())
            .map(|(((&c, &d), &yi), &li)| (c + d * yi - li).max(0.0)),
    );
    let zet = (sub.a0 - sub.a.dot(lam)).max(0.0);
    let s = Array1::from_iter(
        y_out
            .iter()
            .zip(g.iter())
            .map(|(&yi, &gi)| (yi - gi).max(0.0)),
    );
    (xsi, eta, mu, zet, s, y_out)
}

/// Dual solver with the 0.1.x calling convention (coefficient arrays in).
#[allow(clippy::too_many_arguments)]
pub fn subsolv_dual(
    low: &[f64],
    upp: &[f64],
    alfa: &[f64],
    beta: &[f64],
    p0: &[f64],
    q0: &[f64],
    p: Array2<f64>,
    q: Array2<f64>,
    a0: f64,
    a: &[f64],
    b: &[f64],
    c: &[f64],
    d: &[f64],
    tol: f64,
    max_newton: usize,
) -> SolveDualResult {
    let m = b.len();
    let n = low.len();
    assert_eq!(p.shape(), &[m, n]);
    assert_eq!(q.shape(), &[m, n]);
    let sub = Subproblem {
        low: Array1::from(low.to_vec()),
        upp: Array1::from(upp.to_vec()),
        alfa: Array1::from(alfa.to_vec()),
        beta: Array1::from(beta.to_vec()),
        p0: Array1::from(p0.to_vec()),
        q0: Array1::from(q0.to_vec()),
        p,
        q,
        b: Array1::from(b.to_vec()),
        a0,
        a: Array1::from(a.to_vec()),
        c: Array1::from(c.to_vec()),
        d: Array1::from(d.to_vec()),
        r0: None,
        r: None,
    };
    solve_dual(&sub, tol, max_newton, true)
}
