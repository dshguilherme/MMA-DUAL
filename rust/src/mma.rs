//! MMA outer iteration: public [`mmasub`].
//!
//! Port of `src/mmadual/mma.py`.

use crate::approximations::{mma_subproblem, Subproblem};
use crate::dual_solver::{recover_multipliers, solve_dual, DualInfo};
use ndarray::{Array1, Array2};

/// Result of one MMA iteration ([`mmasub`]).
#[derive(Clone, Debug)]
pub struct MmaResult {
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
    /// Updated lower asymptotes.
    pub low: Array1<f64>,
    /// Updated upper asymptotes.
    pub upp: Array1<f64>,
    /// Dual iteration info (when requested).
    pub info: Option<DualInfo>,
}

/// Options for [`mmasub`] beyond the positional MMA arguments.
#[derive(Clone, Debug)]
pub struct MmaOptions {
    /// Move limit fraction of `(xmax - xmin)`.
    pub move_limit: f64,
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
    /// Regularization `raa0`.
    pub raa0: f64,
    /// Fraction of asymptote distance kept as move room.
    pub albefa: f64,
    /// Dual stopping tolerance.
    pub tol_dual: f64,
    /// Maximum dual Newton steps.
    pub max_newton: usize,
    /// Attach [`DualInfo`] to the result.
    pub return_info: bool,
    /// Emit dual-solver warnings on stderr.
    pub warn: bool,
}

impl Default for MmaOptions {
    fn default() -> Self {
        Self {
            move_limit: 0.5,
            asyinit: 0.5,
            asydecr: 0.7,
            asyincr: 1.2,
            asymin: 0.01,
            asymax: 10.0,
            raa0: 1e-5,
            albefa: 0.1,
            tol_dual: 1e-10,
            max_newton: 100,
            return_info: false,
            warn: true,
        }
    }
}

/// One MMA iteration: build the subproblem at `xval` and solve it by the dual method.
#[allow(clippy::too_many_arguments)]
pub fn mmasub(
    _m: usize,
    _n: usize,
    iter: usize,
    xval: &[f64],
    xmin: &[f64],
    xmax: &[f64],
    xold1: &[f64],
    xold2: &[f64],
    _f0val: f64,
    df0dx: &[f64],
    fval: &[f64],
    dfdx: &Array2<f64>,
    low: &[f64],
    upp: &[f64],
    a0: f64,
    a: &[f64],
    c: &[f64],
    d: &[f64],
    opts: &MmaOptions,
) -> MmaResult {
    let sub: Subproblem = mma_subproblem(
        iter,
        xval,
        xmin,
        xmax,
        xold1,
        xold2,
        df0dx,
        fval,
        dfdx,
        low,
        upp,
        a0,
        a,
        c,
        d,
        opts.move_limit,
        opts.asyinit,
        opts.asydecr,
        opts.asyincr,
        opts.asymin,
        opts.asymax,
        opts.raa0,
        opts.albefa,
    );
    let sol = solve_dual(&sub, opts.tol_dual, opts.max_newton, opts.warn);
    let (xsi, eta, mu, zet, s, y) =
        recover_multipliers(&sub, &sol.x, &sol.y, sol.z, &sol.lam);
    MmaResult {
        x: sol.x,
        y,
        z: sol.z,
        lam: sol.lam,
        xsi,
        eta,
        mu,
        zet,
        s,
        low: sub.low,
        upp: sub.upp,
        info: opts.return_info.then_some(sol.info),
    }
}
