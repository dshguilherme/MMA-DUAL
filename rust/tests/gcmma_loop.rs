//! GCMMA loop on Svanberg's beam (conservativeness inner iterations).

use approx::assert_abs_diff_eq;
use mmadual::gcmma::{asymp, concheck, gcmmasub, raaupdate, AsympOptions, GcmmaOptions};
use ndarray::Array2;

fn beam_values(x: &[f64]) -> (f64, Vec<f64>, Array2<f64>, f64, Vec<f64>) {
    let w = [61.0, 37.0, 19.0, 7.0, 1.0];
    let n = x.len();
    let f0 = 0.0624 * x.iter().sum::<f64>();
    let df0dx = vec![0.0624; n];
    let g = w.iter().zip(x.iter()).map(|(&wj, &xj)| wj / xj.powi(3)).sum::<f64>() - 1.0;
    let mut dg = Array2::<f64>::zeros((1, n));
    for j in 0..n {
        dg[[0, j]] = -3.0 * w[j] / x[j].powi(4);
    }
    (f0, df0dx, dg, g, vec![g])
}

#[test]
fn svanberg_beam_gcmma() {
    let m = 1usize;
    let n = 5usize;
    let mut x = vec![5.0; n];
    let mut xold1 = x.clone();
    let mut xold2 = x.clone();
    let xmin = vec![1.0; n];
    let xmax = vec![10.0; n];
    let mut low = xmin.clone();
    let mut upp = xmax.clone();
    let mut raa0 = 1e-5;
    let mut raa = vec![1e-5];
    let raa0eps = 1e-5;
    let raaeps = vec![1e-5];
    let a0 = 1.0;
    let a = [0.0];
    let c = [1000.0];
    let d = [0.0];
    let epsimin = 1e-7;
    let opts = GcmmaOptions {
        warn: false,
        ..GcmmaOptions::default()
    };
    let asymp_opts = AsympOptions::default();

    for k in 1..=40 {
        let (f0, df0dx, dg, _g, fval) = beam_values(&x);
        let (low_n, upp_n, raa0_n, raa_n) = asymp(
            k, n, &x, &xold1, &xold2, &xmin, &xmax, &low, &upp, raa0, &raa, raa0eps, &raaeps,
            &df0dx, &dg, &asymp_opts,
        );
        low = low_n.to_vec();
        upp = upp_n.to_vec();
        raa0 = raa0_n;
        raa = raa_n.to_vec();

        let mut res = gcmmasub(
            m, n, k, epsimin, &x, &xmin, &xmax, &low, &upp, raa0, &raa, f0, &df0dx, &fval, &dg,
            a0, &a, &c, &d, &opts,
        );
        let mut xnew = res.x.to_vec();
        let (mut f0new, _, _, _, mut fnew) = beam_values(&xnew);
        let mut inner = 0;
        while !concheck(m, epsimin, res.f0app, f0new, res.fapp.as_slice().unwrap(), &fnew)
            && inner < 15
        {
            let (raa0_u, raa_u) = raaupdate(
                &xnew, &x, &xmin, &xmax, &low, &upp, f0new, &fnew, res.f0app,
                res.fapp.as_slice().unwrap(), raa0, &raa, raa0eps, &raaeps, epsimin,
            );
            raa0 = raa0_u;
            raa = raa_u.to_vec();
            res = gcmmasub(
                m, n, k, epsimin, &x, &xmin, &xmax, &low, &upp, raa0, &raa, f0, &df0dx, &fval,
                &dg, a0, &a, &c, &d, &opts,
            );
            xnew = res.x.to_vec();
            let vals = beam_values(&xnew);
            f0new = vals.0;
            fnew = vals.4;
            inner += 1;
        }
        xold2 = xold1;
        xold1 = x;
        x = xnew;
        let dx = x
            .iter()
            .zip(xold1.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        if dx < 1e-6 {
            break;
        }
    }
    let expected = [6.016, 5.309, 4.494, 3.501, 2.153];
    for j in 0..n {
        assert_abs_diff_eq!(x[j], expected[j], epsilon = 2e-2);
    }
    let f0 = 0.0624 * x.iter().sum::<f64>();
    assert_abs_diff_eq!(f0, 1.3400, epsilon = 5e-3);
}
