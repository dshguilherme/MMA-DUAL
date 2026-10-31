//! MMA coefficient construction smoke tests.

use approx::assert_abs_diff_eq;
use mmadual::approximations::mma_subproblem;
use ndarray::Array2;

#[test]
fn mma_subproblem_shapes_and_positivity() {
    let n = 5;
    let m = 2;
    let xval = vec![5.0; n];
    let xmin = vec![1.0; n];
    let xmax = vec![10.0; n];
    let xold1 = vec![5.0; n];
    let xold2 = vec![5.0; n];
    let df0dx = vec![0.0624; n];
    let fval = vec![0.1, -0.2];
    let mut dfdx = Array2::<f64>::zeros((m, n));
    for j in 0..n {
        dfdx[[0, j]] = -1.0;
        dfdx[[1, j]] = 0.5;
    }
    let low = vec![1.0; n];
    let upp = vec![10.0; n];
    let sub = mma_subproblem(
        1, &xval, &xmin, &xmax, &xold1, &xold2, &df0dx, &fval, &dfdx, &low, &upp, 1.0,
        &[0.0, 0.0], &[1e3, 1e3], &[0.0, 0.0], 0.5, 0.5, 0.7, 1.2, 0.01, 10.0, 1e-5, 0.1,
    );
    assert_eq!(sub.n(), n);
    assert_eq!(sub.m(), m);
    assert_eq!(sub.p.shape(), &[m, n]);
    for j in 0..n {
        assert!(sub.p0[j] > 0.0 && sub.q0[j] > 0.0);
        assert!(sub.low[j] < sub.alfa[j]);
        assert!(sub.beta[j] < sub.upp[j]);
    }
}

#[test]
fn beam_first_iteration_coefficients_finite() {
    let w = [61.0_f64, 37.0, 19.0, 7.0, 1.0];
    let n = 5;
    let x = vec![5.0; n];
    let df0dx = vec![0.0624; n];
    let g = w
        .iter()
        .zip(x.iter())
        .map(|(&wj, &xj)| wj / f64::powi(xj, 3))
        .sum::<f64>()
        - 1.0;
    let mut dg = Array2::<f64>::zeros((1, n));
    for j in 0..n {
        dg[[0, j]] = -3.0 * w[j] / f64::powi(x[j], 4);
    }
    let sub = mma_subproblem(
        1, &x, &[1.0; 5], &[10.0; 5], &x, &x, &df0dx, &[g], &dg, &[1.0; 5], &[10.0; 5], 1.0,
        &[0.0], &[1000.0], &[0.0], 0.5, 0.5, 0.7, 1.2, 0.01, 10.0, 1e-5, 0.1,
    );
    assert!(sub.b[0].is_finite());
    assert_abs_diff_eq!(sub.a0, 1.0, epsilon = 0.0);
}
