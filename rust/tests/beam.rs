//! Svanberg's five-variable cantilever beam (README example).

use approx::assert_abs_diff_eq;
use mmadual::mma::{mmasub, MmaOptions};
use ndarray::Array2;

#[test]
fn svanberg_beam_mma() {
    let w = [61.0, 37.0, 19.0, 7.0, 1.0];
    let m = 1usize;
    let n = 5usize;
    let mut x = vec![5.0; n];
    let mut xold1 = x.clone();
    let mut xold2 = x.clone();
    let xmin = vec![1.0; n];
    let xmax = vec![10.0; n];
    let mut low = xmin.clone();
    let mut upp = xmax.clone();
    let a0 = 1.0;
    let a = [0.0];
    let c = [1000.0];
    let d = [0.0];
    let opts = MmaOptions {
        move_limit: 0.5,
        return_info: true,
        warn: false,
        ..MmaOptions::default()
    };
    let mut k_done = 0usize;
    for k in 1..=30 {
        let f0 = 0.0624 * x.iter().sum::<f64>();
        let df0dx = vec![0.0624; n];
        let g = w.iter().zip(x.iter()).map(|(&wj, &xj)| wj / xj.powi(3)).sum::<f64>() - 1.0;
        let mut dg = Array2::<f64>::zeros((m, n));
        for j in 0..n {
            dg[[0, j]] = -3.0 * w[j] / x[j].powi(4);
        }
        let res = mmasub(
            m, n, k, &x, &xmin, &xmax, &xold1, &xold2, f0, &df0dx, &[g], &dg, &low, &upp, a0,
            &a, &c, &d, &opts,
        );
        xold2 = xold1;
        xold1 = x;
        x = res.x.to_vec();
        low = res.low.to_vec();
        upp = res.upp.to_vec();
        k_done = k;
        let dx = x
            .iter()
            .zip(xold1.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        if dx < 1e-6 {
            let info = res.info.as_ref().expect("info");
            assert!(info.newton_steps > 0 || info.reason == "tolerance");
            break;
        }
    }
    let expected = [6.016, 5.309, 4.494, 3.501, 2.153];
    for j in 0..n {
        assert_abs_diff_eq!(x[j], expected[j], epsilon = 5e-3);
    }
    let f0 = 0.0624 * x.iter().sum::<f64>();
    assert_abs_diff_eq!(f0, 1.3400, epsilon = 1e-3);
    assert!(k_done <= 15, "expected ~11 iterations, got {k_done}");
}
