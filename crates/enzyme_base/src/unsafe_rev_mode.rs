use std::autodiff::*;

// #[autodiff_reverse(d_dup_example, Duplicated, Active)] // Active для return, Duplicated для arg (срез)
// fn example_dup(v: &[f64]) -> f64 {
//     v.iter().map(|&x| x.powi(2)).sum()
// }

// pub fn run() {
//     let mut data = vec![1.0, 2.0, 3.1, 2.333, 44.2, -1.0]; // Mutable Vec для adj
//     let slice = data.as_slice(); // Срез от Vec (immutable для primal)

//     let mut adj = vec![0.0; data.len()]; // Shadow для адъюнктов (градиентов)
//     let mut adj_slice = adj.as_mut_slice(); // &mut срез для Duplicated

//     let val = d_dup_example(slice, adj_slice, 1.0); // seed=1.0; возвращает primal value, обновляет adj_slice

//     println!("Value: {:?}", val); // 1^2 + 2^2 = 5.0
//     println!("Gradient: {:?}", adj); // [2.0, 4.0] (2*x для каждого)
// }
// Прямой код
#[autodiff_forward(d_dot, Dual, Const, DualOnly)]
fn dot(x: &[f32], w: &[f32]) -> f32 {
    x.iter().zip(w).map(|(xi, wi)| xi * wi).sum()
}

// Forward-дифференциал:
//  - x: Dual      — считаем производную по x
//  - w: Const     — по w производную не считаем
//  - ret: DualOnly — вернуть только d(dot)/dx, без самого dot
// fn dot(x: &[f32], w: &[f32]) -> f32;

pub fn run() {
    let x = [1.0, 2.0, 3.0];
    let w = [0.5, -1.0, 4.0];

    // seed по x: dx_i = 1.0 -> directional derivative по вектору (1,1,1)
    let dx = [1.0, 1.0, 1.0];

    // С учётом DualOnly на выходе сигнатура d_dot будет:
    //   fn d_dot(
    //       x:  &[f32], dx: &[f32],
    //       w:  &[f32],
    //   ) -> f32
    //
    // Возвращаемое значение — одно число: directional derivative.
    let d_dot_dx = d_dot(&x, &dx, &w);

    println!("d(dot)/dx along (1,1,1) = {d_dot_dx}");
}
