//! MMA and GCMMA subproblems solved by the dual method.
//!
//! Faithful Rust port of the Python package `mmadual` (GPL-3.0-or-later).
//! Public names match the Python API: [`mmasub`], [`gcmmasub`], [`solve_dual`], etc.

#![deny(missing_docs)]

pub mod approximations;
pub mod diagnostics;
pub mod dual_solver;
pub mod gcmma;
pub mod mma;

pub use approximations::{gcmma_subproblem, mma_subproblem, Subproblem};
pub use dual_solver::{recover_multipliers, solve_dual, DualInfo, SolveDualResult};
pub use gcmma::{asymp, concheck, gcmmasub, kktcheck, raaupdate, GcmmaResult};
pub use mma::{mmasub, MmaResult};

/// Crate version, aligned with the Python package.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
