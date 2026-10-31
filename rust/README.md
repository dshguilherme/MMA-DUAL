# mmadual (Rust)

Faithful Rust port of the Python package [`mmadual`](../README.md) (MMA-DUAL 0.2.1): MMA and GCMMA
subproblems solved by the dual method, GPL-3.0-or-later.

The public names match the Python API (`mmasub`, `gcmmasub`, `solve_dual`, `asymp`, `concheck`,
`raaupdate`, `kktcheck`, plus `diagnostics`). Vectors are 1-D (`ndarray::Array1`); there is no
column-`(n,1)` convention.

## Build and test

```bash
cd rust
cargo test
```

Requires Rust ≥ 1.84 (for `faer` 0.24). Dependencies: `ndarray`, `faer` (pure-Rust dense
eigendecomposition of the dual Hessian).

## Python → Rust map

| Python | Rust |
|---|---|
| `mma_subproblem(...)` → `dict` | [`mma_subproblem`](src/approximations.rs) → [`Subproblem`](src/approximations.rs) |
| `solve_dual(sub, tol, max_newton)` | [`solve_dual`](src/dual_solver.rs) → [`SolveDualResult`](src/dual_solver.rs) |
| `mmasub(..., return_info=...)` | [`mmasub`](src/mma.rs) with [`MmaOptions`](src/mma.rs) |
| `gcmmasub` / `asymp` / `concheck` / `raaupdate` / `kktcheck` | [`gcmma`](src/gcmma.rs) |
| `diagnostics.*` | [`diagnostics`](src/diagnostics.rs) |

Keyword `move` becomes `move_limit` in Rust options (reserved keyword).

## Parity

Integration tests lock Svanberg's five-variable beam (MMA and GCMMA) to the published design
`x ≈ [6.016, 5.309, 4.494, 3.501, 2.153]`, `f0 ≈ 1.3400`, plus finite-difference checks of the dual
gradient/Hessian and a minimax (`a ≠ 0`) smoke test.

## License

GPL-3.0-or-later, same as the Python package. Copyright of the original work belongs to
Olavo M. Silva; this directory is a derivative port. See the root [LICENSE](../LICENSE) and
[CITATION.cff](../CITATION.cff).
