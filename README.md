# MMA-DUAL

[![PyPI](https://img.shields.io/pypi/v/mma-dual.svg)](https://pypi.org/project/mma-dual/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23741623.svg)](https://doi.org/10.5281/zenodo.23741623)
[![Tests](https://github.com/olavosilva/MMA-DUAL/actions/workflows/tests.yml/badge.svg)](https://github.com/olavosilva/MMA-DUAL/actions/workflows/tests.yml)

MMA-DUAL is a Python package that builds the convex subproblems of the method of moving asymptotes
(MMA, Svanberg 1987) and of its globally convergent version (GCMMA, Svanberg 2002) and solves them by
the **dual method**: the design variables are eliminated analytically for fixed constraint multipliers,
and the concave dual function of the $m$ multipliers is maximized by a projected Newton iteration with
analytical Hessian. It is meant for problems with many design variables and few constraints
($m \ll n$), the usual situation in topology optimization.

The public functions `mmasub`, `gcmmasub`, `asymp`, `concheck`, `raaupdate` and `kktcheck` have the
calling signatures and return values of [GCMMA-MMA-Python](https://github.com/arjendeetman/GCMMA-MMA-Python)
(`mmapy`, Deetman 2025), so an existing MMA or GCMMA loop runs unchanged after `from mmadual import ...`.
The subproblems built are those of `mmapy` to rounding; only the solution of the subproblem differs.

**What the package provides**

- A drop-in replacement for the `mmapy` routines, with `numpy` as the only dependency.
- The subproblem as a plain dictionary of coefficients, a dual solver that takes that dictionary as
  its only input, and the recovery of all the multipliers of `mmapy`'s return tuple.
- A per-call report of the dual iteration (steps, stopping reason, residual) with the duality gap of
  the complete returned primal point, checked against the bound implied by the tolerance; the
  diagnostics add the gap of the returned design variables alone, with the auxiliary variables
  re-optimized, so that the two can be compared.
- Diagnostics (KKT residuals, feasibility of the relaxed and of the unrelaxed constraints,
  complementarity, exact-penalty objective, dual value, gaps) that apply to the solution of any
  subproblem solver, including `mmapy`'s, on the same coefficients.
- Exact bound attainment: design variables sit exactly on their subproblem bounds when the analytical
  minimizer puts them there (an interior-point primal iterate stops at a distance of the order of
  $\varepsilon/\xi_j$ from a bound with multiplier $\xi_j$; see [Limitations](#limitations-and-scope)).

**Where the ideas come from.** The dual reduction, the analytical elimination of the design variables,
Newton iterations on the dual and the regularization of the dual Hessian are classical. Svanberg (1987,
Section 5) describes the dual method; Aage and Lazarov (2013, Sections 3.2–3.3) implement a parallel
dual solver in which the primal variables are eliminated analytically, the dual is solved by an
interior-point Newton iteration with the analytical dual Hessian, and the Hessian is regularized by a
diagonal shift. Their formulation (their Eq. 4) has quadratic penalties in both $y$ and $z$; this package
keeps Svanberg's linear term in $z$ and allows $d_i = 0$, so the dual domain acquires the half-space
$a^{\mathsf T}\lambda \le a_0$ and the upper bounds $\lambda_i \le c_i$, treated here by an active-set
method. What is specific to this package is the combination of that active-set treatment of the
multiplier bounds and of the minimax half-space, the step along locally affine dual directions, the
recovery of the primal point with the two gaps and the certificate, the independent diagnostics, and
the `mmapy`-compatible interfaces; see [docs/method.md](docs/method.md).

## Installation

```
pip install mma-dual
```

Python ≥ 3.9 and `numpy` ≥ 1.20. The tests, examples and benchmarks compare with `mmapy`:
`pip install "mma-dual[test]"` installs `mmapy`, `scipy` and `pytest`. From a clone:
`pip install -e ".[test]"`.

### Rust crate

A faithful Rust port lives in [`rust/`](rust/) (crate name `mmadual`, same GPL-3.0-or-later license).
Build and test with:

```
cd rust && cargo test
```

See [`rust/README.md`](rust/README.md) for the Python→Rust API map. The algorithm and public names
match the Python package; vectors are 1-D (`ndarray`).

## Minimal example

Svanberg's five-variable cantilever beam (minimum weight with a tip-deflection limit), solved by an
MMA loop that uses only this package:

```python
import numpy as np
from mmadual import mmasub

w = np.array([61, 37, 19, 7, 1], dtype=float)
m, n = 1, 5
x = 5.0*np.ones((n, 1)); xold1, xold2 = x.copy(), x.copy()
xmin, xmax = np.ones((n, 1)), 10.0*np.ones((n, 1))
low, upp = xmin.copy(), xmax.copy()
a0, a, c, d = 1.0, np.zeros((m, 1)), 1000.0*np.ones((m, 1)), np.zeros((m, 1))
for k in range(1, 31):
    f0, df0 = 0.0624*x.sum(), 0.0624*np.ones((n, 1))
    g, dg = np.array([[np.sum(w/x.ravel()**3) - 1.0]]), (-3.0*w/x.ravel()**4).reshape(1, n)
    xnew, y, z, lam, xsi, eta, mu, zet, s, low, upp, info = mmasub(
        m, n, k, x, xmin, xmax, xold1, xold2, f0, df0, g, dg, low, upp, a0, a, c, d,
        move=0.5, return_info=True)
    xold2, xold1, x = xold1, x, xnew
    if np.abs(x - xold1).max() < 1e-6:
        break
print(f"{k} iterations, x = {np.round(x.ravel(), 3)}, f0 = {0.0624*x.sum():.4f}, "
      f"dual steps {info['newton_steps']}, reason {info['reason']}, gap_returned {info['gap_returned']:.1e}")
```

Output: `11 iterations, x = [6.016 5.309 4.494 3.501 2.153], f0 = 1.3400, dual steps 8, reason tolerance, gap_returned 4.4e-09`
(Svanberg's reported optimum is 6.016, 5.309, 4.494, 3.502, 2.153). `examples/toy_problem.py` runs the
same problem with the MMA and with the GCMMA, including the inner iterations, and
`examples/mbb_topopt.py` runs a minimum-compliance topology optimization with `mmapy` and with `mmadual`.

## Interfaces

### Drop-in routines

```python
from mmadual import mmasub, gcmmasub, asymp, concheck, raaupdate, kktcheck

xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = mmasub(
    m, n, iter, xval, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx,
    low, upp, a0, a, c, d, move=0.5)
```

Arguments and returned tuple are those of `mmapy.mmasub` (column vectors; `z` and `zet` as (1, 1)
arrays); `move`, `asyinit`, `asydecr`, `asyincr`, `asymin`, `asymax`, `raa0` and `albefa` have the same
defaults. Options beyond `mmapy`:

| option | default | effect |
|---|---|---|
| `tol_dual` | `1e-10` | stopping tolerance of the dual iteration, relative to the scale of each constraint (see [Stopping](#stopping-accuracy-and-what-converged-means)) |
| `max_newton` | `100` | maximum number of accepted Newton steps |
| `return_info` | `False` | append the `info` dictionary of the dual iteration to the returned tuple |
| `gcmmasub(..., move=None)` | | optional move limit in the GCMMA subproblem, absent in `mmapy` |

`asymp`, `concheck`, `raaupdate` and `kktcheck` reproduce the outputs of `mmapy`'s helpers to rounding
(`raaupdate` does not modify its `raa` argument in place; `mmapy`'s does). The GCMMA loop of
`examples/toy_problem.py` shows their use.

### Subproblem, solver and recovery

```python
from mmadual import mma_subproblem, gcmma_subproblem, solve_dual, recover_multipliers, diagnostics

sub = mma_subproblem(iter, xval, xmin, xmax, xold1, xold2, df0dx, fval, dfdx, low, upp, a0, a, c, d, move=0.5)
x, y, z, lam, info = solve_dual(sub, tol=1e-10, max_newton=100)
xsi, eta, mu, zet, s, y = recover_multipliers(sub, x, y, z, lam)
check = diagnostics.check_solution(sub, x, y, z, lam, xsi, eta, mu, zet, s)
```

`sub` is a dictionary with the coefficients `low, upp, alfa, beta, p0, q0, P, Q, b, a0, a, c, d` (and
`r0, r` for the GCMMA), 1-D arrays, independent of how it was built. `solve_dual` and `diagnostics` take
it as the only description of the subproblem, so a subproblem built by other code, or solved by another
solver (`mmapy.subsolv` on the same coefficients, for instance), can be checked with the same functions.
`subsolv_dual(m, n, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d, tol, max_newton)` keeps the
0.1.x calling convention as a wrapper of `solve_dual`.

### The `info` dictionary

| key | meaning |
|---|---|
| `converged` | the stopping tolerance was met; for $a \ne 0$, also the certificate below holds |
| `reason` | `'tolerance'`, `'rounding'`, `'max_newton'`, `'no_progress'` or, for $a \ne 0$ only, `'certification'` |
| `newton_steps`, `evaluations` | accepted Newton steps and evaluations of the dual function |
| `residual` | largest projected gradient at exit, relative to the constraint scales |
| `scale` | the per-constraint scales $\kappa_i = \max(1, \lvert b_i\rvert, g_i(x(0)))$ |
| `gap_returned` | $F(x, y, z) - W(\lambda)$ of the point that `mmasub` returns, with $y$ as `recover_multipliers` returns it |
| `gap_allowed` | the largest `gap_returned` consistent with the tolerance (sum of complementarity products bounded by `tol` times the scales, plus classification and rounding allowances) |
| `certified` | `gap_returned <= gap_allowed` (and, for $a \ne 0$, violation of the relaxed constraints within tolerance) |
| `z`, `halfspace_active` | multiplier of the half-space $a^{\mathsf T}\lambda \le a_0$ and whether it is active ($a \ne 0$; else `0`, `False`) |

### Diagnostics

`diagnostics.check_solution(sub, x, y, z, lam, xsi, eta, mu, zet, s)` returns one dictionary with the
stationarity residuals in $x$, $y$ and $z$ computed with the returned multipliers, a projected-gradient
stationarity measure in $x$ that does not use them, the violation of the relaxed constraints
$g_i(x) - a_i z - y_i \le b_i$, the complementarity products, the dual feasibility of $\lambda$, and:

- `f_pen`: the exact-penalty objective $F_{\rm pen}(x)$, the subproblem objective with the auxiliary
  variables chosen optimally for the returned $x$ (`exact_penalty_objective`, `optimal_artificial`);
- `gap`, `gap_rel`: $\mathcal G_x = F_{\rm pen}(x) - W(\lambda)$, the gap that certifies the design
  variables and the dual value (`duality_gap`);
- `F_returned`, `gap_returned`, `gap_returned_rel`: $F(x, y, z)$ and
  $\mathcal G_{\rm ret} = F(x, y, z) - W(\lambda)$ for the auxiliary variables actually returned
  (`subproblem_objective`, `duality_gap_returned`); for a primal-feasible returned point and a
  dual-feasible $\lambda$, $0 \le \mathcal G_x \le \mathcal G_{\rm ret}$, and the difference is the excess
  cost of the returned $y$ and $z$.

`constraint_values(sub, x)` gives $g_i(x) - b_i$, so the violation of the unrelaxed constraints
$g_i(x) \le b_i$ can be computed separately from that of the relaxed ones; the two differ whenever
$y_i > 0$. `dual_value(sub, lam)` checks that $\lambda$ is dual feasible before evaluating $W$.

## Stopping, accuracy and what `converged` means

The dual iteration stops for one of these reasons, reported in `info['reason']`:

- `'tolerance'`: the projected gradient of $W$ is below `tol_dual` $\kappa_i$ for every constraint,
  with $\kappa_i = \max(1, |b_i|, g_i(x(0)))$. The gradient of $W$ is the residual of the relaxed
  constraints at the primal minimizer, so this is a feasibility and complementarity residual of the
  subproblem *with* its auxiliary variables, scaled per constraint.
- `'rounding'`: two consecutive undamped Newton steps increased $W$ by less than $10^{-13}|W|$. This
  is a stagnation test on the value of $W$, taken before the residual is tested again; it does not by
  itself show that the residual or the gap reached the precision of the machine. In the recorded
  runs it occurs on nearly redundant constraints, where $W$ is flat along $m - 1$ directions, and
  occasionally one step before the tolerance would have been met.
- `'max_newton'`, `'no_progress'`: the step limit, or a trial point identical to the current one.
- `'certification'` ($a \ne 0$ only): the tolerance was met but the returned point fails the
  certificate.

`tol_dual` controls a scaled residual, not the gap. The complementarity products $\lambda_i s_i$ and
$\mu_i y_i$ are the residual times $\lambda_i$ or $\mu_i \le c_i$, so with the usual $c_i = 10^3$ and
constraint scales $\kappa_i$ of order $10^2$ to $10^5$ a residual at the tolerance can leave a gap of
$10^{-5}$ to $10^{-2}$ in absolute terms. The gap obtained at a given tolerance also depends on where
the last Newton step lands: once the active set is identified the iteration is a Newton method on a
smooth piece of $W$, and in the recorded runs the gap drops by one to three orders of magnitude per
step, so it can fall by several orders between one tolerance and the next, or stay on a plateau until
a further step is triggered. For these reasons no tolerance value is associated here
with a guaranteed accuracy. The recommendation is to read `gap_returned` (or `gap` from the
diagnostics), compare it with what the application needs, and tighten `tol_dual` when necessary;
one or two more Newton steps usually suffice.

What `converged` and `certified` mean on the two paths of the solver:

- $a = 0$ (the usual MMA/GCMMA form): `converged` means `reason == 'tolerance'`; the certificate is
  evaluated and reported but does not control termination.
- $a \ne 0$ (minimax form): `converged` requires the tolerance on both stationarity measures (on the
  identified active set and by projection onto the whole dual domain) and `certified`, which also
  checks that the returned point violates the relaxed constraints by at most `tol` times the largest
  scale.

A `RuntimeWarning` is issued when the iteration stops without `converged` and the residual is above
$\sqrt{\text{tol}}$ (both paths), and, on the $a \ne 0$ path only, when the certificate fails; on the
$a = 0$ path a failed certificate is reported in `certified` without a warning.

Feasibility of the returned point: the design variables always satisfy $\alpha_j \le x_j \le \beta_j$;
the relaxed constraints hold to the residual above when $d_i = 0$, where
$y_i = \max(0, g_i(x) - b_i - a_i z)$ is recovered from the returned $x$, and to the dual gradient
residual when $d_i > 0$, where $y_i(\lambda)$ is returned. A multiplier at its upper bound,
$\lambda_i = c_i$, means that the constraint may be violated at $x$ with the violation carried by
$y_i$; the recovered $y_i$ can also be zero, in which case the constraint is active or satisfied.

## Limitations and scope

- The dual Hessian is dense, $m \times m$, assembled in $O(nm^2)$ operations and decomposed in
  $O(m^3)$; the package is meant for $m$ up to a few tens of constraints. For $m < n$,
  `mmapy.subsolv` also condenses its Newton system to an $m \times m$ (strictly, $(m+1) \times (m+1)$)
  system assembled in $O(nm^2)$ and solved in $O(m^3)$, so the two methods have the same leading
  orders; the measured differences come from the constants and the number of iterations of the two
  implementations, not from different asymptotic costs.
- $a_i \ge 0$ only (Svanberg's formulation); $a_i < 0$ raises `ValueError`. Only the MMA/GCMMA forms
  with the auxiliary variables $y$ and $z$ of Svanberg's codes.
- Jacobi scaling of the Hessian reduces the effect of differences of scale between the constraints;
  exact scale invariance of the complete iteration is not claimed (the floor of 1 in $\kappa_i$ and
  the rounding allowances are not invariant). The Levenberg–Marquardt damping modifies the whole
  regularized block, with a relatively larger effect on the directions of small curvature.
- Exact bound attainment follows from the analytical minimization of the Lagrangian in $x$; it is
  available to any dual solver that evaluates that minimizer, including a dual interior-point method
  such as that of Aage and Lazarov (2013), whose barrier acts on the multipliers only. It is not a
  property exclusive to this package, and a design variable at zero density in a subproblem does not
  by itself guarantee a better optimum of the original problem.
- The cost of the subproblem is one part of the cost of an optimization. In the recorded topology
  optimization (below) the analysis dominated the iteration, and the gain in the complete iteration
  was 1.4 to 1.7 against an 11 to 18 gain in the subproblem; with another analysis the balance changes.

## Evidence

All the numbers below come from the verification and benchmark campaign of the accompanying
manuscript, with MMA-DUAL 0.2.0 (tag `v0.2.0`, commit `42b434c`; the solver of 0.2.1 is identical, the release
only updates the documentation) and `mmapy` 0.3.1. The scripts, per-run results, frozen subproblems and
environment metadata are deposited as a reproduction archive on Zenodo
([10.5281/zenodo.23842200](https://doi.org/10.5281/zenodo.23842200)), and
[docs/benchmarks-0.2.0.md](docs/benchmarks-0.2.0.md) gives the protocol and the complete tables. The earlier timing tables of this README (version 0.1.x, a shared virtual machine,
`benchmarks/bench_n_m.py`) are superseded and are not mixed with these.

**Verification** ([docs/verification-0.2.0.md](docs/verification-0.2.0.md) and the manuscript):
subproblem coefficients equal to those captured from `mmapy` to $3 \times 10^{-16}$ relative;
analytical gradient and Hessian against central differences ($1.1 \times 10^{-9}$ and
$2.2 \times 10^{-10}$ relative at the best step, with the expected jumps of the Hessian where a
variable leaves a bound); KKT residuals and both gaps on twelve representative subproblems, including
violated constraints, $d_i > 0$, constraint scales from $10^{-6}$ to $10^6$, identical constraints,
$m > n$, and minimax forms; 2100 near-affine minimax subproblems against `scipy.optimize.linprog`
(dual values to $2.8 \times 10^{-14}$), 350 affine $a = 0$ cases, 400 curved minimax cases; complete
MMA and GCMMA loops on Svanberg's beam with iterate-by-iterate agreement with `mmapy`.

**Subproblem benchmark** (AMD Ryzen 9 5950X, Windows 11, Python 3.12.9, NumPy 2.5.4 and SciPy 1.18.1
with OpenBLAS restricted to one thread, verified with `threadpoolctl`; random MMA subproblems at
iteration 5 with $c_i = 10^3$, $d_i = 0$, move limit 0.2; ten independent subproblems per regular
configuration and five per poorly scaled one; medians of the paired ratios). "Complete call" is the
time of `mmasub` of each package, including the construction of the approximation; "frozen" is
`mmapy.subsolv` against `solve_dual` on the same coefficients; "matched" is the ratio when the dual
solver is rerun with the loosest tolerance whose gap does not exceed that of `mmapy.subsolv`, the
MMA-DUAL time being estimated as construction plus solution time measured separately (it excludes
the multiplier recovery that `mmasub` performs).

| objective scale | $n$ | $m$ | `mmapy` complete call [s] | MMA-DUAL complete call [s] | ratio, complete call | ratio, frozen | ratio, matched gap | `mmapy` reference |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1 000 | 4 | 0.015 | 0.0026 | 5.6 | 5.8 | 5.9 | converged |
| 1 | 1 000 | 16 | 0.014 | 0.0037 | 3.6 | 3.9 | 3.8 | converged |
| 1 | 10 000 | 4 | 0.049 | 0.0088 | 5.5 | 5.5 | 6.0 | converged |
| 1 | 10 000 | 16 | 0.087 | 0.028 | 3.2 | 4.0 | 3.1 | converged |
| 1 | 10 000 | 64 | 0.28 | 0.11 | 2.4 | 3.1 | 2.4 | converged |
| 1 | 100 000 | 4 | 1.23 | 0.107 | 11.2 | 12.4 | 11.2 | converged |
| 1 | 100 000 | 16 | 1.30 | 0.316 | 4.1 | 5.3 | 3.8 | converged |
| 1 | 100 000 | 64 | 3.83 | 1.33 | 2.9 | 3.7 | 2.8 | converged |
| $10^{-3}$ | 10 000 | 4 | 2.22 | 0.0079 | 280 | 279 | 300 | iteration limit |
| $10^{-3}$ | 10 000 | 16 | 3.52 | 0.029 | 119 | 151 | 117 | iteration limit |
| $10^{-3}$ | 100 000 | 4 | 34.8 | 0.105 | 332 | 367 | 340 | iteration limit |
| $10^{-3}$ | 100 000 | 16 | 62.3 | 0.335 | 187 | 218 | 213 | iteration limit |
| $10^{-3}$ | 1 000 | 64 | 0.45 | 0.011 | 41 | 46 | 43 | converged |

In the eight regularly scaled configurations the complete-call ratios are between 2.4 and 11.2,
decreasing with $m$, and the ratios at matched gap are within 15 % of them. In the four poorly scaled
configurations with $m \le 16$ the `mmapy` reference stopped at its iteration limit with gaps of
$2 \times 10^{-3}$ to $0.15$ relative (stationarity $2 \times 10^{-3}$ to $4 \times 10^{-3}$), so the
ratios of hundreds measure the cost of an iteration limit against a converged solve, not a gain at
equal accuracy; at $\epsilon_{\min} = 10^{-9}$ and $10^{-11}$ `mmapy.subsolv` converges in the
$m = 16$ configuration at $n = 10^4$ but not in the $m = 4$ one. The dual solver took 8 to 11 Newton
steps in every configuration. At its default tolerance its gap was the smaller one in eleven of the
thirteen configurations and the larger one at $n = 10^5$ with $m = 16$ and 64 ($7 \times 10^{-7}$ and
$1 \times 10^{-8}$ relative, against $8 \times 10^{-9}$ and $1 \times 10^{-9}$ for `mmapy`), the
consequence of the per-constraint scaling of the tolerance discussed above; tolerances of $10^{-12}$
to $10^{-14}$ brought the median gap below $10^{-11}$ at a cost of 0 to 16 % of the solution time.

**Topology optimization** (MBB beam of the 88-line code, $900 \times 300$ elements, $n = 270\,000$,
filter radius 10, volume fraction 0.5, PyPardiso on one thread, same workstation;
`examples/mbb_topopt.py` with the diagnostics of the manuscript's script). With the default parameters
of both packages the two runs follow different trajectories: the interior-point solutions never return
a design variable at a bound and leave the void densities at about 0.05, and the `mmapy` run stops, by
the design-change criterion, at a design with compliance 226.2 against 205.6 for the MMA-DUAL run.
On the subproblem of iteration 70 solved on identical coefficients, the bound multipliers of the void
elements have a median of $8.5 \times 10^{-7}$ (of the order of $\lambda/(0.5 n)$ for the volume
constraint), and the offsets $x_j - \alpha_j$ of the interior-point solution agree with
$\varepsilon/\xi_j$ to 2.3 % at $\epsilon_{\min} = 10^{-9}$ and $10^{-11}$ and in order of magnitude at
$10^{-7}$. With $\epsilon_{\min} = 10^{-9}$ the `mmapy` run gives a similar topology and a compliance
within 0.08 % of the MMA-DUAL one, at 1.7 times the core time per iteration (analysis 4.2 s,
`mmasub` 3.6 s against 0.19 s). The MMA-DUAL run with tolerance $10^{-12}$ reproduces the run with
$10^{-10}$ to $10^{-6}$ in the compliance.

## Tests, examples and benchmarks

```
pip install -e ".[test]"
pytest                                                 # 74 tests, about a minute; needs mmapy and scipy
python examples/toy_problem.py                         # Svanberg's beam, MMA and GCMMA, no mmapy needed
python examples/mbb_topopt.py 300 100 500 3.0 0.05     # MBB beam, mmapy against mmadual, time per component
python benchmarks/bench_n_m.py [1e-3] [matched] [v012] # quick timing comparison on random subproblems
python benchmarks/mbb_subproblem_accuracy.py           # accuracy of mmapy.subsolv against epsimin on one MBB subproblem
python benchmarks/minimax_stress.py                    # minimax subproblems against scipy.optimize.linprog
```

The tests cover the construction of the subproblems and the GCMMA helpers against coefficients captured
from `mmapy` 0.3.1 (`tests/data/reference_mmapy.npz`); the gradient and Hessian of the dual function
against finite differences, including a change of the active set; the agreement of the solver with
version 0.1.2; KKT residuals and both gaps on edge cases (violated constraints, $d > 0$, extreme
scales, redundant constraints, singular Hessians, rescaled constraints); the minimax path against
`mmapy` and against `scipy.optimize.linprog`; complete GCMMA iterations without `mmapy`; and, when
`mmapy` is installed, the exact-penalty objective of the dual solution against the interior-point one.
`scipy.linalg.LinAlgWarning` raised inside `mmapy`'s routine on degenerate reference runs is filtered in
`pyproject.toml`, not hidden (`pytest -W default::scipy.linalg.LinAlgWarning`).

## Package layout

```
src/mmadual/
  approximations.py   moving asymptotes, bounds alfa/beta, coefficients p0, q0, P, Q, b (MMA) and r0, r (GCMMA)
  dual_solver.py      dual function, gradient, Hessian; projected Newton with LM damping (a = 0 and a != 0 paths);
                      recovery of the multipliers; certificate of the returned point
  mma.py              mmasub
  gcmma.py            gcmmasub, asymp, concheck, raaupdate, kktcheck
  diagnostics.py      KKT residuals, exact-penalty objective, dual value, both duality gaps
docs/                 method.md (formulation, dual function, derivatives, the iteration),
                      verification-0.2.0.md (equivalence with 0.1.2 and mmapy), benchmarks-0.2.0.md (campaign protocol and tables)
tests/                pytest suite; tests/data/ holds coefficients captured from mmapy 0.3.1
benchmarks/           bench_n_m.py, mbb_subproblem_accuracy.py, minimax_stress.py,
                      mmadual_v012_reference.py (the single-module version 0.1.2, kept as reference)
examples/             toy_problem.py, mbb_topopt.py
```

## Origin and license

The dual method for the MMA subproblem is described by Svanberg (1987, Section 5); the approximations
and their parameters (asymptote rules, the regularization $0.001\,|\partial f/\partial x| + r_{aa0}/(x_{\max} - x_{\min})$,
the GCMMA update of the conservativeness parameters) are those of Svanberg's published MATLAB codes
(2007) and of the GCMMA paper (2002). Versions 0.1.x of this package were derived from GCMMA-MMA-Python
by Arjen Deetman, the Python translation of those codes, from which the construction of the
subproblems was taken verbatim and the solution procedure replaced by the dual solver. From version
0.2.0 the construction is written from the published formulations and reproduces, to rounding, the
subproblems of `mmapy`, which remains the reference of the tests and benchmarks. The dual solver was
written for this package; its ingredients have the antecedents cited above. The package is distributed
under the GNU General Public License v3 or later, the license of its ancestors (see `LICENSE`).

## References

- Svanberg, K. (1987). The method of moving asymptotes: a new method for structural optimization.
  *International Journal for Numerical Methods in Engineering* 24, 359–373.
  [10.1002/nme.1620240207](https://doi.org/10.1002/nme.1620240207)
- Svanberg, K. (2002). A class of globally convergent optimization methods based on conservative convex
  separable approximations. *SIAM Journal on Optimization* 12, 555–573.
  [10.1137/S1052623499362822](https://doi.org/10.1137/S1052623499362822)
- Svanberg, K. (2007). MMA and GCMMA: two methods for nonlinear optimization. Technical note, KTH.
- Aage, N., Lazarov, B. S. (2013). Parallel framework for topology optimization using the method of
  moving asymptotes. *Structural and Multidisciplinary Optimization* 47, 493–505.
  [10.1007/s00158-012-0869-2](https://doi.org/10.1007/s00158-012-0869-2)
- Deetman, A. (2025). GCMMA-MMA-Python: Python implementation of the Method of Moving Asymptotes,
  version 0.3.1. [10.5281/zenodo.15459165](https://doi.org/10.5281/zenodo.15459165)
- Andreassen, E., Clausen, A., Schevenels, M., Lazarov, B. S., Sigmund, O. (2011). Efficient topology
  optimization in MATLAB using 88 lines of code. *Structural and Multidisciplinary Optimization* 43, 1–16.
  [10.1007/s00158-010-0594-7](https://doi.org/10.1007/s00158-010-0594-7)

## Citing

This repository is archived on Zenodo: the concept DOI [10.5281/zenodo.23741623](https://doi.org/10.5281/zenodo.23741623)
always resolves to the latest version, and each release has its own DOI (0.2.0:
[10.5281/zenodo.23785895](https://doi.org/10.5281/zenodo.23785895); the DOI of 0.2.1 is listed on the
Zenodo page once the release is archived, and `CITATION.cff` is updated with it; use the "Cite this
repository" button on GitHub). The reproduction archive of the benchmarks has its own DOI, given in
[Evidence](#evidence). Please cite the version used, together
with GCMMA-MMA-Python, from which earlier versions of MMA-DUAL were derived, and the papers by Svanberg
that define the methods:

```bibtex
@software{silva2026mmadual,
  author  = {Silva, Olavo M.},
  title   = {{MMA-DUAL}: {MMA} and {GCMMA} subproblems solved by the dual method},
  version = {0.2.1},
  year    = {2026},
  doi     = {10.5281/zenodo.23842751},
  url     = {https://github.com/olavosilva/MMA-DUAL},
  note    = {GPL-3.0}
}

@software{deetman2025gcmmammapython,
  author  = {Deetman, Arjen},
  title   = {{GCMMA-MMA-Python}: {Python} implementation of the Method of Moving Asymptotes},
  version = {0.3.1},
  year    = {2025},
  doi     = {10.5281/zenodo.15459165},
  url     = {https://github.com/arjendeetman/GCMMA-MMA-Python}
}

@article{svanberg1987mma,
  author  = {Svanberg, Krister},
  title   = {The method of moving asymptotes - a new method for structural optimization},
  journal = {International Journal for Numerical Methods in Engineering},
  volume  = {24},
  number  = {2},
  pages   = {359--373},
  year    = {1987},
  doi     = {10.1002/nme.1620240207}
}

@article{svanberg2002gcmma,
  author  = {Svanberg, Krister},
  title   = {A class of globally convergent optimization methods based on conservative convex separable approximations},
  journal = {SIAM Journal on Optimization},
  volume  = {12},
  number  = {2},
  pages   = {555--573},
  year    = {2002},
  doi     = {10.1137/S1052623499362822}
}
```

The DOI of GCMMA-MMA-Python above is the one of version 0.3.1; the concept DOI that always resolves to
the latest version is 10.5281/zenodo.13197565.
