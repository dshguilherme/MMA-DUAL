//! Construction of the separable convex MMA and GCMMA subproblems.
//!
//! Port of `src/mmadual/approximations.py`.

use ndarray::{Array1, Array2, Axis};

/// Lower bound of `xmax - xmin` used in regularization (Svanberg's MATLAB codes).
pub const XMAMI_MIN: f64 = 1e-5;
/// Fraction of `|df/dx|` added to both p and q so every term is strictly convex.
pub const CONVEX_REG: f64 = 1e-3;

/// Separable convex MMA/GCMMA subproblem coefficients.
///
/// Design variables have length `n`; constraints have length `m`. `P` and `Q` are `(m, n)`.
#[derive(Clone, Debug)]
pub struct Subproblem {
    /// Moving asymptote L.
    pub low: Array1<f64>,
    /// Moving asymptote U.
    pub upp: Array1<f64>,
    /// Lower move/asymptote bound α.
    pub alfa: Array1<f64>,
    /// Upper move/asymptote bound β.
    pub beta: Array1<f64>,
    /// Objective coefficients p0.
    pub p0: Array1<f64>,
    /// Objective coefficients q0.
    pub q0: Array1<f64>,
    /// Constraint coefficients P (`m × n`).
    pub p: Array2<f64>,
    /// Constraint coefficients Q (`m × n`).
    pub q: Array2<f64>,
    /// Right-hand sides b.
    pub b: Array1<f64>,
    /// Coefficient a0 of z.
    pub a0: f64,
    /// Coefficients a of z in the constraints.
    pub a: Array1<f64>,
    /// Linear costs c of y.
    pub c: Array1<f64>,
    /// Quadratic costs d of y.
    pub d: Array1<f64>,
    /// GCMMA objective constant r0 (None for MMA).
    pub r0: Option<f64>,
    /// GCMMA constraint constants r (None for MMA).
    pub r: Option<Array1<f64>>,
}

impl Subproblem {
    /// Number of design variables.
    pub fn n(&self) -> usize {
        self.p0.len()
    }

    /// Number of constraints.
    pub fn m(&self) -> usize {
        self.b.len()
    }
}

fn as_1d(v: &[f64], n: usize) -> Array1<f64> {
    if v.len() == 1 && n != 1 {
        Array1::from_elem(n, v[0])
    } else {
        assert_eq!(v.len(), n, "expected length {n}, got {}", v.len());
        Array1::from(v.to_vec())
    }
}

/// Asymptotes L (low) and U (upp) of the MMA at outer iteration `iter` (1-based).
pub fn moving_asymptotes(
    iter: usize,
    xval: &Array1<f64>,
    xold1: &Array1<f64>,
    xold2: &Array1<f64>,
    xmin: &Array1<f64>,
    xmax: &Array1<f64>,
    low: &Array1<f64>,
    upp: &Array1<f64>,
    asyinit: f64,
    asydecr: f64,
    asyincr: f64,
    asymin: f64,
    asymax: f64,
) -> (Array1<f64>, Array1<f64>) {
    let xmami = xmax - xmin;
    if iter <= 2 {
        return (xval - asyinit * &xmami, xval + asyinit * &xmami);
    }
    let trend = (xval - xold1) * (xold1 - xold2);
    let factor = trend.mapv(|t| {
        if t > 0.0 {
            asyincr
        } else if t < 0.0 {
            asydecr
        } else {
            1.0
        }
    });
    let mut low_new = xval - &factor * (xold1 - low);
    let mut upp_new = xval + &factor * (upp - xold1);
    for j in 0..xval.len() {
        let lo = xval[j] - asymax * xmami[j];
        let hi = xval[j] - asymin * xmami[j];
        low_new[j] = low_new[j].clamp(lo.min(hi), lo.max(hi));
        let lo = xval[j] + asymin * xmami[j];
        let hi = xval[j] + asymax * xmami[j];
        upp_new[j] = upp_new[j].clamp(lo.min(hi), lo.max(hi));
    }
    (low_new, upp_new)
}

/// Bounds α ≤ x ≤ β of the subproblem.
pub fn variable_bounds(
    xval: &Array1<f64>,
    xmin: &Array1<f64>,
    xmax: &Array1<f64>,
    low: &Array1<f64>,
    upp: &Array1<f64>,
    albefa: f64,
    move_limit: Option<f64>,
) -> (Array1<f64>, Array1<f64>) {
    let mut alfa = low + albefa * (xval - low);
    let mut beta = upp - albefa * (upp - xval);
    if let Some(mv) = move_limit {
        let span = xmax - xmin;
        alfa = Array1::from_iter(
            alfa.iter()
                .zip(xval.iter())
                .zip(span.iter())
                .map(|((a, x), s)| (*a).max(*x - mv * *s)),
        );
        beta = Array1::from_iter(
            beta.iter()
                .zip(xval.iter())
                .zip(span.iter())
                .map(|((b, x), s)| (*b).min(*x + mv * *s)),
        );
    }
    let alfa = Array1::from_iter(alfa.iter().zip(xmin.iter()).map(|(a, x)| (*a).max(*x)));
    let beta = Array1::from_iter(beta.iter().zip(xmax.iter()).map(|(b, x)| (*b).min(*x)));
    (alfa, beta)
}

/// Coefficients p0, q0 and P, Q of the separable approximations at `xval`.
pub fn separable_coefficients(
    xval: &Array1<f64>,
    xmin: &Array1<f64>,
    xmax: &Array1<f64>,
    low: &Array1<f64>,
    upp: &Array1<f64>,
    df0dx: &Array1<f64>,
    dfdx: &Array2<f64>,
    raa0: f64,
    raa: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>, Array2<f64>, Array2<f64>) {
    let n = xval.len();
    let m = dfdx.nrows();
    let xmami_inv = (xmax - xmin).mapv(|s| 1.0 / s.max(XMAMI_MIN));
    let ux2 = (upp - xval).mapv(|v| v * v);
    let xl2 = (xval - low).mapv(|v| v * v);
    let pos0 = df0dx.mapv(|v| v.max(0.0));
    let neg0 = df0dx.mapv(|v| (-v).max(0.0));
    let reg0 = CONVEX_REG * (&pos0 + &neg0) + raa0 * &xmami_inv;
    let p0 = (&pos0 + &reg0) * &ux2;
    let q0 = (&neg0 + &reg0) * &xl2;

    let mut p = Array2::<f64>::zeros((m, n));
    let mut q = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let pos = dfdx[[i, j]].max(0.0);
            let neg = (-dfdx[[i, j]]).max(0.0);
            let reg = CONVEX_REG * (pos + neg) + raa[i] * xmami_inv[j];
            p[[i, j]] = (pos + reg) * ux2[j];
            q[[i, j]] = (neg + reg) * xl2[j];
        }
    }
    (p0, q0, p, q)
}

/// Values at `x` of the separable terms of the objective and constraints (without constants).
pub fn approximation_values(
    x: &Array1<f64>,
    low: &Array1<f64>,
    upp: &Array1<f64>,
    p0: &Array1<f64>,
    q0: &Array1<f64>,
    p: &Array2<f64>,
    q: &Array2<f64>,
) -> (f64, Array1<f64>) {
    let uxinv = (upp - x).mapv(|v| 1.0 / v);
    let xlinv = (x - low).mapv(|v| 1.0 / v);
    let f0 = p0.dot(&uxinv) + q0.dot(&xlinv);
    let g = p.dot(&uxinv) + q.dot(&xlinv);
    (f0, g)
}

/// MMA subproblem at outer iteration `iter` (Svanberg 1987 / 2007 MATLAB parameters).
#[allow(clippy::too_many_arguments)]
pub fn mma_subproblem(
    iter: usize,
    xval: &[f64],
    xmin: &[f64],
    xmax: &[f64],
    xold1: &[f64],
    xold2: &[f64],
    df0dx: &[f64],
    fval: &[f64],
    dfdx: &Array2<f64>,
    low: &[f64],
    upp: &[f64],
    a0: f64,
    a: &[f64],
    c: &[f64],
    d: &[f64],
    move_limit: f64,
    asyinit: f64,
    asydecr: f64,
    asyincr: f64,
    asymin: f64,
    asymax: f64,
    raa0: f64,
    albefa: f64,
) -> Subproblem {
    let n = xval.len();
    let m = fval.len();
    let xval = as_1d(xval, n);
    let xmin = as_1d(xmin, n);
    let xmax = as_1d(xmax, n);
    let xold1 = as_1d(xold1, n);
    let xold2 = as_1d(xold2, n);
    let mut low = as_1d(low, n);
    let mut upp = as_1d(upp, n);
    let df0dx = as_1d(df0dx, n);
    let fval = as_1d(fval, m);
    assert_eq!(dfdx.shape(), &[m, n]);
    let (low_n, upp_n) = moving_asymptotes(
        iter, &xval, &xold1, &xold2, &xmin, &xmax, &low, &upp, asyinit, asydecr, asyincr, asymin,
        asymax,
    );
    low = low_n;
    upp = upp_n;
    let (alfa, beta) = variable_bounds(&xval, &xmin, &xmax, &low, &upp, albefa, Some(move_limit));
    let raa = Array1::from_elem(m, raa0);
    let (p0, q0, p, q) =
        separable_coefficients(&xval, &xmin, &xmax, &low, &upp, &df0dx, dfdx, raa0, &raa);
    let (_, gval) = approximation_values(&xval, &low, &upp, &p0, &q0, &p, &q);
    let b = &gval - &fval;
    Subproblem {
        low,
        upp,
        alfa,
        beta,
        p0,
        q0,
        p,
        q,
        b,
        a0,
        a: as_1d(a, m),
        c: as_1d(c, m),
        d: as_1d(d, m),
        r0: None,
        r: None,
    }
}

/// GCMMA subproblem (Svanberg 2002) for the current asymptotes and conservativeness parameters.
#[allow(clippy::too_many_arguments)]
pub fn gcmma_subproblem(
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
    albefa: f64,
    move_limit: Option<f64>,
) -> Subproblem {
    let n = xval.len();
    let m = fval.len();
    let xval = as_1d(xval, n);
    let xmin = as_1d(xmin, n);
    let xmax = as_1d(xmax, n);
    let low = as_1d(low, n);
    let upp = as_1d(upp, n);
    let df0dx = as_1d(df0dx, n);
    let fval = as_1d(fval, m);
    let raa = as_1d(raa, m);
    assert_eq!(dfdx.shape(), &[m, n]);
    let (alfa, beta) = variable_bounds(&xval, &xmin, &xmax, &low, &upp, albefa, move_limit);
    let (p0, q0, p, q) =
        separable_coefficients(&xval, &xmin, &xmax, &low, &upp, &df0dx, dfdx, raa0, &raa);
    let (f0s, gs) = approximation_values(&xval, &low, &upp, &p0, &q0, &p, &q);
    let r0 = f0val - f0s;
    let r = &fval - &gs;
    let b = -&r;
    Subproblem {
        low,
        upp,
        alfa,
        beta,
        p0,
        q0,
        p,
        q,
        b,
        a0,
        a: as_1d(a, m),
        c: as_1d(c, m),
        d: as_1d(d, m),
        r0: Some(r0),
        r: Some(r),
    }
}

/// Convenience: build `dfdx` as `(m, n)` from a flat row-major slice.
pub fn dfdx_from_slice(m: usize, n: usize, data: &[f64]) -> Array2<f64> {
    assert_eq!(data.len(), m * n);
    Array2::from_shape_vec((m, n), data.to_vec()).expect("shape")
}

/// Row-major reshape helper used by drop-in APIs.
pub fn ensure_dfdx(m: usize, n: usize, dfdx: Array2<f64>) -> Array2<f64> {
    if dfdx.shape() == [m, n] {
        dfdx
    } else if dfdx.len() == m * n {
        dfdx.into_shape_with_order((m, n)).expect("reshape dfdx")
    } else {
        panic!("dfdx shape {:?} incompatible with m={m}, n={n}", dfdx.shape());
    }
}

/// Sum along axis helper kept for call sites that prefer ndarray idioms.
pub fn sum_axis1(a: &Array2<f64>) -> Array1<f64> {
    a.sum_axis(Axis(1))
}
