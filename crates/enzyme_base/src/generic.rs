use std::autodiff::*;

/// P-норма (p > 0) для вектора длиной `N`.
#[autodiff_forward(d_lp_norm, Dual, Dual)]
fn lp_norm<const N: usize, const P: i32>(v: [f64; N]) -> f64 {
    debug_assert!(P > 0);
    let sum: f64 = v.iter().map(|&x| x.powi(P)).sum();
    sum.powf(1.0 / P as f64)
}

pub fn run() {
    println!("Обобщённые p-нормы для любых длины/степени");
    let v = [3.0, 4.0, 12.0];
    let (val, grad_x) = d_lp_norm::<3, 3>(v, [1.0, 0.0, 0.0]);
    let (_, grad_y) = d_lp_norm::<3, 3>(v, [0.0, 1.0, 0.0]);
    let (_, grad_z) = d_lp_norm::<3, 3>(v, [0.0, 0.0, 1.0]);
    println!("||v||_3 = {val:.4}, ∂/∂x = {grad_x:.4}, ∂/∂y = {grad_y:.4}, ∂/∂z = {grad_z:.4}");
}
