use std::autodiff::*;

#[autodiff_forward(d_round, Dual, Dual)]
fn f_round(x: f64) -> f64 {
    x.round()
}

#[autodiff_forward(d_floor, Dual, Dual)]
fn f_floor(x: f64) -> f64 {
    x.floor()
}

#[autodiff_forward(d_ceil, Dual, Dual)]
fn f_ceil(x: f64) -> f64 {
    x.ceil()
}

pub fn run() {
    // Точки “внутри интервала” — градиент будет 0
    let x1 = 1.51_f64;
    let (y_round_1, dy_round_1) = d_round(x1, 1.0);
    let (y_floor_1, dy_floor_1) = d_floor(x1, 1.0);
    let (y_ceil_1, dy_ceil_1) = d_ceil(x1, 1.0);

    println!("x = {x1}");
    println!("round(x) = {y_round_1}, d/dx round = {dy_round_1}");
    println!("floor(x) = {y_floor_1}, d/dx floor = {dy_floor_1}");
    println!("ceil(x)  = {y_ceil_1},  d/dx ceil  = {dy_ceil_1}");
    println!();

    // Точка “на границе” — формально нет производной, но AD
    // всё равно вернёт какой-то путь-зависимый результат
    let x2 = 1.0_f64;
    let (y_round_2, dy_round_2) = d_round(x2, 1.0);
    let (y_floor_2, dy_floor_2) = d_floor(x2, 1.0);
    let (y_ceil_2, dy_ceil_2) = d_ceil(x2, 1.0);

    println!("x = {x2}");
    println!("round(x) = {y_round_2}, d/dx round = {dy_round_2}");
    println!("floor(x) = {y_floor_2}, d/dx floor = {dy_floor_2}");
    println!("ceil(x)  = {y_ceil_2},  d/dx ceil  = {dy_ceil_2}");
}
