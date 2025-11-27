use std::autodiff::*;

#[autodiff_forward(d_relu, Dual, Dual)]
fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

#[autodiff_forward(d_abs, Dual, Dual)]
fn abs(x: f32) -> f32 {
    if x >= 0.0 { x } else { -x }
}

pub fn run() {
    let (v, dx) = d_relu(0.0, 1.0);
    println!("В точке 0 для relu субградиент [0,1], а получаем  {dx}");
    let (v, dx) = d_abs(0.0, 1.0);
    println!("В точке 0 для abs субградиент [-1,1], а получаем {dx}");
}
