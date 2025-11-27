use std::autodiff::*;

#[autodiff_forward(d_norm, Dual, Dual)]
fn norm(v: [f64; 2]) -> f64 {
    let s: f64 = v.iter().map(|&x| x.powi(2)).sum();
    s.sqrt()
}

#[autodiff_forward(d_norm_raw, Dual, Const, Dual)]
fn norm_raw(v: *const f64, len: usize) -> f64 {
    let v = unsafe { std::slice::from_raw_parts(v, len) };
    let s: f64 = v.iter().map(|&x| x.powi(2)).sum();
    s.sqrt()
}

pub fn run() {
    println!("Норма");
    let v = [3.0, 4.0];
    let (val, grad_x) = d_norm(v, [1.0, 0.0]); // val=25, grad_x=6 (по x)
    let (_, grad_y) = d_norm(v, [0.0, 1.0]); // grad_y=8 (по y)
    println!("обычная: {val}, {grad_x}");

    unsafe {
        let (_, grad_x) = d_norm_raw(v.as_ptr(), [1.0, 0.0].as_ptr(), 2); // grad_y=8 (по y)
        let (_, grad_y) = d_norm_raw(v.as_ptr(), [0.0, 1.0].as_ptr(), 2); // grad_y=8 (по y)
        println!("через raw: {val}, {grad_x}");
    }
}
