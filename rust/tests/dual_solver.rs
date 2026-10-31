//! Dual-solver analytic gradient/Hessian and a≠0 smoke tests.

use approx::assert_relative_eq;
use mmadual::approximations::mma_subproblem;
use mmadual::dual_solver::{dual_function, dual_hessian, primal_x, solve_dual, BOUND_TOL};
use ndarray::{Array1, Array2};

fn random_subproblem(n: usize, m: usize, scale: f64, seed: u64) -> mmadual::Subproblem {
    // simple LCG
    let mut s = seed;
    let mut rand = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        (s >> 33) as f64 / (u32::MAX as f64)
    };
    let xval: Vec<f64> = (0..n).map(|_| 0.05 + 0.9 * rand()).collect();
    let xold1: Vec<f64> = xval
        .iter()
        .map(|&x| (x + 0.05 * (2.0 * rand() - 1.0)).clamp(0.0, 1.0))
        .collect();
    let xold2: Vec<f64> = xold1
        .iter()
        .map(|&x| (x + 0.05 * (2.0 * rand() - 1.0)).clamp(0.0, 1.0))
        .collect();
    let fval: Vec<f64> = (0..m).map(|_| -1.0 + 2.0 * rand()).collect();
    let df0dx: Vec<f64> = (0..n).map(|_| scale * (2.0 * rand() - 1.0)).collect();
    let mut dfdx = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            dfdx[[i, j]] = 2.0 * rand() - 1.0;
        }
    }
    let low: Vec<f64> = xval.iter().map(|&x| x - 0.3).collect();
    let upp: Vec<f64> = xval.iter().map(|&x| x + 0.3).collect();
    let xmin = vec![0.0; n];
    let xmax = vec![1.0; n];
    let a = vec![0.0; m];
    let c = vec![1e3; m];
    let d = vec![0.0; m];
    mma_subproblem(
        5, &xval, &xmin, &xmax, &xold1, &xold2, &df0dx, &fval, &dfdx, &low, &upp, 1.0, &a, &c,
        &d, 0.2, 0.5, 0.7, 1.2, 0.01, 10.0, 1e-5, 0.1,
    )
}

fn interior_point(sub: &mmadual::Subproblem, lam: &Array1<f64>) -> bool {
    let (x, _, _) = primal_x(sub, lam);
    x.iter()
        .zip(sub.alfa.iter())
        .zip(sub.beta.iter())
        .all(|((&xj, &a), &b)| xj > a && xj < b)
}

#[test]
fn gradient_finite_differences() {
    for seed in 0..5u64 {
        let sub = random_subproblem(300, 4, 1.0, seed);
        let mut s = 100 + seed;
        let mut rand = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (s >> 33) as f64 / (u32::MAX as f64)
        };
        for _ in 0..3 {
            let lam = Array1::from_iter((0..4).map(|_| 0.01 + 0.29 * rand()));
            if !interior_point(&sub, &lam) {
                continue;
            }
            let (_w, grad, ..) = dual_function(&sub, &lam);
            let h = 1e-6;
            for i in 0..4 {
                let mut e = Array1::zeros(4);
                e[i] = h;
                let (wp, ..) = dual_function(&sub, &(&lam + &e));
                let (wm, ..) = dual_function(&sub, &(&lam - &e));
                let fd = (wp - wm) / (2.0 * h);
                assert!(
                    (fd - grad[i]).abs() <= 1e-6 * 1.0_f64.max(grad[i].abs()),
                    "gradient component {i}: {} vs FD {fd}",
                    grad[i]
                );
            }
        }
    }
}

#[test]
fn hessian_finite_differences_interior() {
    let sub = random_subproblem(80, 3, 1.0, 7);
    let lam = Array1::from(vec![0.05, 0.08, 0.02]);
    if !interior_point(&sub, &lam) {
        return;
    }
    let (_w, _g, x, _y, p, q) = dual_function(&sub, &lam);
    let h = dual_hessian(&sub, &lam, &x, &p, &q);
    let eps = 1e-6;
    for i in 0..3 {
        for j in 0..3 {
            let mut e = Array1::zeros(3);
            e[j] = eps;
            let (_wp, gp, ..) = dual_function(&sub, &(&lam + &e));
            let (_wm, gm, ..) = dual_function(&sub, &(&lam - &e));
            let fd = (gp[i] - gm[i]) / (2.0 * eps);
            assert_relative_eq!(h[[i, j]], fd, epsilon = 1e-5, max_relative = 1e-4);
        }
    }
}

#[test]
fn solve_box_converges() {
    let sub = random_subproblem(100, 4, 1.0, 3);
    let sol = solve_dual(&sub, 1e-10, 100, false);
    assert!(
        sol.info.converged || sol.info.reason == "rounding",
        "reason={} residual={}",
        sol.info.reason,
        sol.info.residual
    );
    for j in 0..sub.n() {
        assert!(sol.x[j] >= sub.alfa[j] - 1e-14);
        assert!(sol.x[j] <= sub.beta[j] + 1e-14);
    }
}

#[test]
fn solve_polytope_minimax_smoke() {
    let n = 20usize;
    let m = 3usize;
    let xval = vec![0.5; n];
    let xmin = vec![0.0; n];
    let xmax = vec![1.0; n];
    let xold1 = vec![0.48; n];
    let xold2 = vec![0.46; n];
    let df0dx = vec![1.0; n];
    let fval = vec![0.1, -0.2, 0.05];
    let mut dfdx = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            dfdx[[i, j]] = if (i + j) % 2 == 0 { 0.5 } else { -0.3 };
        }
    }
    let low = vec![0.2; n];
    let upp = vec![0.8; n];
    let a = vec![1.0; m];
    let c = vec![1e3; m];
    let d = vec![0.0; m];
    let mut sub = mma_subproblem(
        5, &xval, &xmin, &xmax, &xold1, &xold2, &df0dx, &fval, &dfdx, &low, &upp, 1.0, &a, &c,
        &d, 0.2, 0.5, 0.7, 1.2, 0.01, 10.0, 1e-5, 0.1,
    );
    // force minimax coefficients
    sub.a = Array1::from(a);
    sub.a0 = 1.0;
    let sol = solve_dual(&sub, 1e-8, 100, false);
    assert!(
        sol.info.residual <= 1e-4 || sol.info.converged || sol.info.certified,
        "minimax residual={} reason={}",
        sol.info.residual,
        sol.info.reason
    );
    let _ = BOUND_TOL;
}
