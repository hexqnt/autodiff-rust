use std::autodiff::*;

#[autodiff_reverse(df, Active, Active, Active)]
fn f(x: f32, y: f32) -> f32 {
    (x.powi(2) + y.powi(2)).sqrt()
}

#[autodiff_forward(dff, Dual, Dual, Dual)]
fn ff(x: f32, y: f32) -> f32 {
    (x.powi(2) + y.powi(2)).sqrt()
}

pub fn run() {
    println!("Скалярные функции");
    let (x, y) = (5.0, 7.0);
    let (z, bx, by) = df(x, y, 1.0);
    println!("{}, {}", z, bx);
    let (bx, by) = dff(x, y, 1.0, 0.0);
    println!("{}, {}", z, bx);
}
