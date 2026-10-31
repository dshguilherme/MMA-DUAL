//! GCMMA (Svanberg 2002): [`gcmmasub`] and outer/inner helpers.
//!
//! Port of `src/mmadual/gcmma.py`.

use crate::approximations::{
    approximation_values, gcmma_subproblem, Subproblem, XMAMI_MIN,
};
use crate::dual_solver::{recover_multipliers, solve_dual, DualInfo};
use ndarray::{Array1, Array2};

const RAACOF_MIN: f64 = 1e-12;

/// Result of one GCMMA subproblem solve.
#[derive(Clone, Debug)]
pub struct GcmmaResult {
    /// New design point.
    pub x: Array1<f64>,
    /// Artificial variables y.
    pub y: Array1<f64>,
    /// Artificial variable z.
    pub z: f64,
    /// Constraint multipliers λ.
    pub lam: Array1<f64>,
    /// Lower bound multipliers ξ.
    pub xsi: Array1<f64>,
    /// Upper bound multipliers η.
    pub eta: Array1<f64>,
    /// Multipliers of y ≥ 0.
    pub mu: Array1<f64>,
    /// Multiplier of z ≥ 0.
    pub zet: f64,
    /// Constraint slacks.
    pub s: Array1<f64>,
    /// Approximation of f0 at `x`.
    pub f0app: f64,
    /// Approximations of fi at `x`.
    pub fapp: Array1<f64>,
    /// Dual iteration info (when requested).
    pub info: Option<DualInfo>,
}

/// Options for [`gcmmasub`].
#[derive(Clone, Debug)]
pub struct GcmmaOptions {
    /// Fraction of asymptote distance kept as move room.
    pub albefa: f64,
    /// Optional move limit fraction of `(xmax - xmin)`.
    pub move_limit: Option<f64>,
    /// Dual stopping tolerance.
    pub tol_dual: f64,
    /// Maximum dual Newton steps.
    pub max_newton: usize,
    /// Attach [`DualInfo`] to the result.
    pub return_info: bool,
    /// Emit dual-solver warnings on stderr.
    pub warn: bool,
}

impl Default for GcmmaOptions {
    fn default() -> Self {
        Self {
            albefa: 0.1,
            move_limit: None,
            tol_dual: 1e-10,
            max_newton: 100,
            return_info: false,
            warn: true,
        }
    }
}

/// GCMMA subproblem at `xval` with current asymptotes and conservativeness parameters.
#[allow(clippy::too_many_arguments)]
pub fn gcmmasub(
    _m: usize,
    _n: usize,
    _iter: usize,
    _epsimin: f64,
    xval: &[f64],
    xmin: &[f64],
    xmax: &[f64],
    low: &[f64],
    upp: &[f64],
    raa0: f64,
    raa: &[f64],
    f0val: f64,
    df0dx: &[f64],
    fval: &[f64],
    dfdx: &Array2<f64>,
    a0: f64,
    a: &[f64],
    c: &[f64],
    d: &[f64],
    opts: &GcmmaOptions,
) -> GcmmaResult {
    let sub: Subproblem = gcmma_subproblem(
        xval,
        xmin,
        xmax,
        low,
        upp,
        raa0,
        raa,
        f0val,
        df0dx,
        fval,
        dfdx,
        a0,
        a,
        c,
        d,
        opts.albefa,
        opts.move_limit,
    );
    let sol = solve_dual(&sub, opts.tol_dual, opts.max_newton, opts.warn);
    let (xsi, eta, mu, zet, s, y) =
        recover_multipliers(&sub, &sol.x, &sol.y, sol.z, &sol.lam);
    let (f0s, gs) = approximation_values(
        &sol.x, &sub.low, &sub.upp, &sub.p0, &sub.q0, &sub.p, &sub.q,
    );
    let r0 = sub.r0.expect("gcmma subproblem has r0");
    let r = sub.r.as_ref().expect("gcmma subproblem has r");
    GcmmaResult {
        x: sol.x,
        y,
        z: sol.z,
        lam: sol.lam,
        xsi,
        eta,
        mu,
        zet,
        s,
        f0app: r0 + f0s,
        fapp: r + gs,
        info: opts.return_info.then_some(sol.info),
    }
}

/// Options for [`asymp`].
#[derive(Clone, Debug)]
pub struct AsympOptions {
    /// Initial asymptote distance factor.
    pub asyinit: f64,
    /// Asymptote contraction factor.
    pub asydecr: f64,
    /// Asymptote expansion factor.
    pub asyincr: f64,
    /// Minimum asymptote distance factor.
    pub asymin: f64,
    /// Maximum asymptote distance factor.
    pub asymax: f64,
}

impl Default for AsympOptions {
    fn default() -> Self {
        Self {
            asyinit: 0.5,
            asydecr: 0.7,
            asyincr: 1.2,
            asymin: 0.01,
            asymax: 10.0,
        }
    }
}

/// Start of an outer GCMMA iteration: initial conservativeness and asymptotes.
#[allow(clippy::too_many_arguments)]
pub fn asymp(
    outeriter: usize,
    n: usize,
    xval: &[f64],
    xold1: &[f64],
    xold2: &[f64],
    xmin: &[f64],
    xmax: &[f64],
    low: &[f64],
    upp: &[f64],
    _raa0: f64,
    _raa: &[f64],
    raa0eps: f64,
    raaeps: &[f64],
    df0dx: &[f64],
    dfdx: &Array2<f64>,
    opts: &AsympOptions,
) -> (Array1<f64>, Array1<f64>, f64, Array1<f64>) {
    let xval = Array1::from(xval.to_vec());
    let xold1 = Array1::from(xold1.to_vec());
    let xold2 = Array1::from(xold2.to_vec());
    let xmin = Array1::from(xmin.to_vec());
    let xmax = Array1::from(xmax.to_vec());
    let low = Array1::from(low.to_vec());
    let upp = Array1::from(upp.to_vec());
    let df0dx = Array1::from(df0dx.to_vec());
    let xmami = (&xmax - &xmin).mapv(|s| s.max(XMAMI_MIN));
    let raa0_new = raa0eps.max((0.1 / n as f64) * df0dx.mapv(f64::abs).dot(&xmami));
    let m = dfdx.nrows();
    let mut raa_new = Array1::<f64>::zeros(m);
    for i in 0..m {
        let mut acc = 0.0;
        for j in 0..n {
            acc += dfdx[[i, j]].abs() * xmami[j];
        }
        raa_new[i] = raaeps[i].max((0.1 / n as f64) * acc);
    }
    let (low_new, upp_new) = if outeriter <= 2 {
        (
            &xval - opts.asyinit * &xmami,
            &xval + opts.asyinit * &xmami,
        )
    } else {
        let trend = (&xval - &xold1) * (&xold1 - &xold2);
        let factor = trend.mapv(|t| {
            if t > 0.0 {
                opts.asyincr
            } else if t < 0.0 {
                opts.asydecr
            } else {
                1.0
            }
        });
        let mut low_new = &xval - &factor * (&xold1 - &low);
        let mut upp_new = &xval + &factor * (&upp - &xold1);
        for j in 0..n {
            let lo = xval[j] - opts.asymax * xmami[j];
            let hi = xval[j] - opts.asymin * xmami[j];
            low_new[j] = low_new[j].clamp(lo.min(hi), lo.max(hi));
            let lo = xval[j] + opts.asymin * xmami[j];
            let hi = xval[j] + opts.asymax * xmami[j];
            upp_new[j] = upp_new[j].clamp(lo.min(hi), lo.max(hi));
        }
        (low_new, upp_new)
    };
    (low_new, upp_new, raa0_new, raa_new)
}

/// Conservativeness test of the inner GCMMA iteration: `true` if approximations bound the true values.
pub fn concheck(
    _m: usize,
    epsimin: f64,
    f0app: f64,
    f0valnew: f64,
    fapp: &[f64],
    fvalnew: &[f64],
) -> bool {
    let ok0 = f0app + epsimin >= f0valnew;
    let ok = fapp
        .iter()
        .zip(fvalnew.iter())
        .all(|(&fa, &fv)| fa + epsimin >= fv);
    ok0 && ok
}

/// Update of `raa0` and `raa` in the inner GCMMA iteration.
#[allow(clippy::too_many_arguments)]
pub fn raaupdate(
    xmma: &[f64],
    xval: &[f64],
    xmin: &[f64],
    xmax: &[f64],
    low: &[f64],
    upp: &[f64],
    f0valnew: f64,
    fvalnew: &[f64],
    f0app: f64,
    fapp: &[f64],
    raa0: f64,
    raa: &[f64],
    _raa0eps: f64,
    _raaeps: &[f64],
    epsimin: f64,
) -> (f64, Array1<f64>) {
    let n = xmma.len();
    let mut raacof = 0.0;
    for j in 0..n {
        let xmami = (xmax[j] - xmin[j]).max(XMAMI_MIN);
        let dx = xmma[j] - xval[j];
        raacof += dx * dx / ((upp[j] - xmma[j]) * (xmma[j] - low[j])) * (upp[j] - low[j]) / xmami;
    }
    raacof = raacof.max(RAACOF_MIN);
    let mut raa0_new = raa0;
    if f0valnew > f0app + 0.5 * epsimin {
        raa0_new = (1.1 * (raa0_new + (f0valnew - f0app) / raacof)).min(10.0 * raa0_new);
    }
    let mut raa_new = Array1::from(raa.to_vec());
    for i in 0..fvalnew.len() {
        if fvalnew[i] > fapp[i] + 0.5 * epsimin {
            raa_new[i] =
                (1.1 * (raa_new[i] + (fvalnew[i] - fapp[i]) / raacof)).min(10.0 * raa_new[i]);
        }
    }
    (raa0_new, raa_new)
}

/// Residuals of the KKT conditions of the original problem.
#[allow(clippy::too_many_arguments)]
pub fn kktcheck(
    m: usize,
    n: usize,
    x: &[f64],
    y: &[f64],
    z: f64,
    lam: &[f64],
    xsi: &[f64],
    eta: &[f64],
    mu: &[f64],
    zet: f64,
    s: &[f64],
    xmin: &[f64],
    xmax: &[f64],
    df0dx: &[f64],
    fval: &[f64],
    dfdx: &Array2<f64>,
    a0: f64,
    a: &[f64],
    c: &[f64],
    d: &[f64],
) -> (Array1<f64>, f64, f64) {
    assert_eq!(dfdx.shape(), &[m, n]);
    let mut res = Vec::with_capacity(3 * n + 4 * m + 2);
    for j in 0..n {
        let mut v = df0dx[j] - xsi[j] + eta[j];
        for i in 0..m {
            v += dfdx[[i, j]] * lam[i];
        }
        res.push(v);
    }
    for i in 0..m {
        res.push(c[i] + d[i] * y[i] - mu[i] - lam[i]);
    }
    let mut az = 0.0;
    for i in 0..m {
        az += a[i] * lam[i];
    }
    res.push(a0 - zet - az);
    for i in 0..m {
        res.push(fval[i] - a[i] * z - y[i] + s[i]);
    }
    for j in 0..n {
        res.push(xsi[j] * (x[j] - xmin[j]));
    }
    for j in 0..n {
        res.push(eta[j] * (xmax[j] - x[j]));
    }
    for i in 0..m {
        res.push(mu[i] * y[i]);
    }
    res.push(zet * z);
    for i in 0..m {
        res.push(lam[i] * s[i]);
    }
    let residual = Array1::from(res);
    let norm = residual.dot(&residual).sqrt();
    let maxabs = residual.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    (residual, norm, maxabs)
}
