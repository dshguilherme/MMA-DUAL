# Changelog

## Unreleased

- Rust port of the package under `rust/` (crate `mmadual`): dual solver (box and minimax paths),
  MMA/GCMMA interfaces, diagnostics, and parity tests including Svanberg's beam. See `rust/README.md`.

## 0.2.1 (2026-10-31)

Documentation release: `src/mmadual/` is identical to 0.2.0 (tag `v0.2.0`, commit `42b434c`); the version number changes so that the PyPI page carries the new README. The results recorded for 0.2.0 apply unchanged.

- README reorganized for use as a scientific library: purpose, minimal runnable example (Svanberg's beam), interfaces, the `info` dictionary and the diagnostics with their actual field names, stopping and accuracy, limitations, evidence, references.
- Antecedents stated: the dual reduction (Svanberg 1987), the dual interior-point Newton solver with analytical Hessian and diagonal regularization of Aage and Lazarov (2013), and the difference of formulation (linear $z$, $d_i = 0$ allowed, half-space and upper bounds in the dual domain) that the active-set path of this package addresses.
- Corrected statements: Jacobi scaling reduces scale differences, exact invariance of the complete iteration is not claimed; the Levenberg–Marquardt damping modifies the whole regularized block, with a relatively larger effect on the directions of small curvature; `tol_dual` controls a scaled residual and no tolerance value is associated with a guaranteed gap; the reason `'rounding'` is a stagnation test on $W$ and does not by itself show that machine precision was reached; $\lambda_i = c_i$ does not imply a violated constraint (the recovered $y_i$ can be zero); `mmapy.subsolv` has $3n + 4m + 2$ unknowns and, for $m < n$, condenses its Newton system to $(m+1) \times (m+1)$ at $O(nm^2)$, the same leading orders as the dual Hessian; a dual Newton step costs one or more evaluations of the dual function plus the Hessian, not one pass over the variables; exact bound attainment belongs to the analytical minimizer and is available to a dual interior-point solver too.
- Benchmarks: the 0.1.x tables (shared virtual machine, `bench_n_m.py`) replaced by the verified campaign of the manuscript (workstation, one thread, `mmapy` 0.3.1, MMA-DUAL 0.2.0 at `42b434c`), with the complete call, the frozen subproblem and the matched-gap estimate distinguished and the converged references separated from those that reached the iteration limit; the topology optimization of the manuscript ($900 \times 300$ MBB beam) summarized with the $\varepsilon/\xi_j$ offsets. Details in `docs/benchmarks-0.2.0.md`.
- Derivations moved from the README to `docs/method.md`, with the general decomposition of the returned-point gap and the conditions under which it reduces to a sum of complementarity products.

## 0.2.0 (2026-10-29)

Reorganization of the single module `mmadual.py` (0.1.2) into the package `src/mmadual/`, with the
construction of the MMA/GCMMA subproblems written from the published formulations and the GCMMA
helpers added, so that a complete MMA or GCMMA loop runs without `mmapy`.

### Preserved

- `mmasub(m, n, iter, xval, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx, low, upp, a0, a, c, d, move=0.5, asyinit=0.5, asydecr=0.7, asyincr=1.2, asymin=0.01, asymax=10, raa0=1e-5, albefa=0.1, tol_dual=1e-10, max_newton=100)`: same positional arguments, keyword options and returned tuple (11 column vectors; `z` and `zet` as (1, 1) arrays, as in `mmapy`).
- `gcmmasub(m, n, iter, epsimin, xval, xmin, xmax, low, upp, raa0, raa, f0val, df0dx, fval, dfdx, a0, a, c, d, albefa=0.1, move=None, tol_dual=1e-10, max_newton=100)`: same, with `f0app`, `fapp`.
- `subsolv_dual(m, n, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d, tol, max_newton)`: the 0.1.x calling convention of the dual solver, as a wrapper of `solve_dual`.
- The subproblems built: identical to those of 0.1.2 and of `mmapy` 0.3.1 to rounding (relative $10^{-13}$), including the regularizations `raa0`, the factor 0.001, the floor $10^{-5}$ of $x_{\max} - x_{\min}$ and the GCMMA constants $r_0$, $r$ (`tests/test_approximations.py`, against coefficients captured from `mmapy` and stored in `tests/data/reference_mmapy.npz`).
- The dual solver: same algorithm (projected Newton, Jacobi-scaled spectral Levenberg-Marquardt step, per-constraint tolerance, rounding and `max_newton` stops, `RuntimeWarning`) except for the step of the components without curvature (see Changed); the solutions coincide with those of 0.1.2 to $10^{-12}$ on the same subproblems (`tests/test_dual_solver.py::test_same_solution_as_v012`). Timings within the measurement noise of 0.1.2 (`benchmarks/bench_n_m.py ... v012`).
- The tests of 0.1.2 (`test_against_mmapy.py`, `test_edge_cases.py`) and the benchmark `bench_n_m.py` with its `matched` mode; the MBB example.
- GPL-3.0-or-later; the credits to Svanberg and to GCMMA-MMA-Python; the Zenodo concept DOI and the PyPI distribution name `mma-dual`.

### Added

- `asymp`, `concheck`, `raaupdate`, `kktcheck` with the signatures of `mmapy`, written from Svanberg (2002) and the MATLAB codes' rules; outputs identical to `mmapy`'s to rounding. `raaupdate` does not modify its `raa` argument in place (`mmapy`'s does).
- Package interface: `mma_subproblem`, `gcmma_subproblem` (subproblem dict), `solve_dual` (returns an `info` dict: `converged`, `reason`, `newton_steps`, `evaluations`, `residual`, `scale`, `z`, `halfspace_active`), `recover_multipliers`; `return_info=True` in `mmasub`/`gcmmasub`.
- `diagnostics`: `kkt_residuals`, `kkt_summary`, `check_solution`, `exact_penalty_objective`, `optimal_artificial`, `dual_value` (with a dual-feasibility check), `duality_gap` (certifies $x$, artificial variables re-optimized), `duality_gap_returned` (certifies the returned $(x, y, z)$), `constraint_values`.
- Subproblems with $a \ne 0$ ($a_i \ge 0$, minimax form): active-set projected Newton method on the box intersected with the half-space $a^{\mathsf T}\lambda \le a_0$ (`_solve_polytope`): tolerance-based classification of the multipliers at the bounds and of the half-space, Newton/LM direction in the tangent space of the active constraints, step up to the first bound (ratio test) with Armijo acceptance, flat directions scaled to the first bound, $z$ recovered as the multiplier of the half-space determined jointly with the box active set, stationarity measured on the identified active set and by the projection onto the whole domain, and a final certificate of the returned point by its duality gap (`converged` requires it; reason `'certification'` otherwise). Tested against `mmapy` on minimax subproblems, on affine dual functions on the simplex, and against `scipy.optimize.linprog` on the generator of the second review (`tests/test_minimax.py`, `tests/test_minimax_independent_stress.py`; `benchmarks/minimax_stress.py`).
- `info` keys `gap_returned`, `gap_allowed`, `certified` (both paths; for $a = 0$ informative only, the stopping rule of 0.1.x is kept).
- Tests: `test_approximations.py`, `test_dual_solver.py` (finite-difference gradient and Hessian, active-set transitions, identity with 0.1.2), `test_diagnostics.py`, `test_gcmma_loop.py` (complete GCMMA without `mmapy`, iterate-by-iterate agreement with `mmapy`'s helpers), `test_minimax.py`. All automated with pytest.
- `benchmarks/mbb_subproblem_accuracy.py`: accuracy of `mmapy.subsolv` as a function of `epsimin` on a fixed MBB subproblem, against the dual solver.
- `examples/toy_problem.py` runs the MMA and the GCMMA with `mmadual` only (`mmapy` optional, for comparison). `examples/mbb_topopt.py` is importable (command-line parsing moved under `__main__`).
- The `v012` option of `bench_n_m.py` and the split of the `mmasub` time into construction and solution.
- GitHub Actions workflow `tests.yml`: pytest with `mmapy` installed on every push to `main`/`refactor-package` and on pull requests (Python 3.9 and 3.12).

### Changed

- `a != 0` no longer raises `NotImplementedError` (it is solved); `a_i < 0` raises `ValueError`.
- $a = 0$ path: a multiplier without curvature (no free design variable in its constraint, $W$ affine in it) steps to its bound in the direction of the gradient instead of taking the gradient as a step of unit length (`_flat_step`; 0.1.x needed up to hundreds of steps on exactly affine constraints with a small gradient, and stopped at `max_newton` in 3 of 350 synthetic cases, now regression tests). A trial point identical to the one just rejected is not evaluated again. Everything else of the 0.1.x iteration is unchanged: where the flat step is not triggered the results are identical to 0.1.2; where it is, the dual values agree to $10^{-15}$ and the gaps stay within the tolerance band (`docs/verification-0.2.0.md`, Section 3.1).
- Layout: `src/mmadual/` package instead of the single module; `pyproject.toml` uses `packages.find` with `where = ["src"]`. The single module of 0.1.2 is kept unchanged as `benchmarks/mmadual_v012_reference.py`.
- Version 0.2.0 in `pyproject.toml`, `mmadual.__version__` and `CITATION.cff` (the version DOI of 0.2.0 is entered in `CITATION.cff` once Zenodo issues it).

### Not supported / limitations

- $a_i < 0$ (not part of Svanberg's formulation).
- The dual Hessian is dense ($m \times m$) and its assembly costs $O(nm^2)$: the method is meant for $m$ up to a few tens of constraints.
- Exact scale invariance of the whole iteration under a rescaling of the constraints is not claimed: the Newton step on the curved components and the step to the bound of the flat ones are invariant, the floor of 1 in the per-constraint tolerance is not; the solution is invariant to rounding when the path is and to `tol_dual` otherwise (tested).
- Only the MMA/GCMMA forms with the artificial variables $y$, $z$ of Svanberg's codes; no other approximation schemes.
