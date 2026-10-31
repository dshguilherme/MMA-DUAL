//! Diagnostics on a solved MMA subproblem.

use mmadual::approximations::mma_subproblem;
use mmadual::diagnostics::{check_solution, duality_gap, exact_penalty_objective};
use mmadual::dual_solver::{recover_multipliers, solve_dual};
use ndarray::Array2;

#[test]
fn beam_subproblem_has_small_gap() {
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
    let sol = solve_dual(&sub, 1e-10, 100, false);
    assert!(sol.info.converged, "reason={}", sol.info.reason);
    let (xsi, eta, mu, zet, s, y) =
        recover_multipliers(&sub, &sol.x, &sol.y, sol.z, &sol.lam);
    let (gap, gap_rel) = duality_gap(&sub, &sol.x, &sol.lam);
    assert!(gap >= -1e-10, "gap={gap}");
    assert!(gap_rel < 1e-6, "gap_rel={gap_rel}");
    let fpen = exact_penalty_objective(&sub, &sol.x);
    assert!(fpen.is_finite());
    let summary = check_solution(
        &sub, &sol.x, &y, sol.z, &sol.lam, &xsi, &eta, &mu, zet, &s,
    );
    assert!(summary["proj_x"] < 1e-6, "proj_x={}", summary["proj_x"]);
    assert!(summary["gap"] < 1e-6, "gap={}", summary["gap"]);
}
