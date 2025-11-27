use std::autodiff::*;
use std::f64;

// Наивная реализация: log(1 + exp(x))
fn softplus_naive(x: f64) -> f64 {
    (1.0 + x.exp()).ln()
}

// Численно устойчивая реализация: log(1 + exp(x))
fn softplus_stable(x: f64) -> f64 {
    if x > 0.0 {
        // 1 + exp(x) = exp(x) * (1 + exp(-x))
        // => log(1 + exp(x)) = x + log(1 + exp(-x))
        x + (-x).exp().ln_1p()
    } else {
        // при x <= 0 переполнения нет, считаем напрямую, но через log1p
        x.exp().ln_1p()
    }
}

// === Обёртки для forward-mode autodiff ===

#[autodiff_forward(d_softplus_naive, Dual, Dual)]
fn ad_softplus_naive(x: f64) -> f64 {
    softplus_naive(x)
}

#[autodiff_forward(d_softplus_stable, Dual, Dual)]
fn ad_softplus_stable(x: f64) -> f64 {
    softplus_stable(x)
}

// Аналитическая производная: σ(x) = exp(x) / (1 + exp(x))
fn softplus_deriv_analytic(x: f64) -> f64 {
    let ex = x.exp();
    ex / (1.0 + ex)
}

pub fn run() {
    let xs = [-100.0_f64, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0];

    println!(
        "{:>8} | {:>18} | {:>18} | {:>18}",
        "x", "naive f", "stable f", "analytic f"
    );
    println!("{}", "-".repeat(70));
    for &x in &xs {
        let f_n = softplus_naive(x);
        let f_s = softplus_stable(x);
        let f_a = (1.0 + x.exp()).ln(); // "идеальная" формула в double

        println!(
            "{:8.1} | {:18.10e} | {:18.10e} | {:18.10e}",
            x, f_n, f_s, f_a
        );
    }

    println!();
    println!(
        "{:>8} | {:>18} | {:>18} | {:>18}",
        "x", "d naive", "d stable", "d analytic"
    );
    println!("{}", "-".repeat(70));
    for &x in &xs {
        // AD-градиенты
        let (_y_n, dy_n) = ad_softplus_naive(x);
        let (_y_s, dy_s) = ad_softplus_stable(x);

        // Аналитическая производная
        let dy_a = softplus_deriv_analytic(x);

        println!(
            "{:8.1} | {:18.10e} | {:18.10e} | {:18.10e}",
            x, dy_n, dy_s, dy_a
        );
    }
}
