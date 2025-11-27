use std::autodiff::*;
#[autodiff_forward(d_f_good, Dual, Dual)]
fn f_good(x: f32) -> f32 {
    x
}

#[autodiff_forward(d_f_wrong, Dual, Dual)]
fn f_wrong(x: f32) -> f32 {
    if x == 0.0 { 0.0 } else { x }
}

pub fn run() {
    let (v, dx) = d_f_good(0.0, 1.0);
    println!("Верно: v={v} dx={dx}");
    let (v, dx) = d_f_wrong(0.0, 1.0);
    println!("Ошибка: v={v} dx={dx}");
}
