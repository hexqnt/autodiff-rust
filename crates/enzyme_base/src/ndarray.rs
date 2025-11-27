use std::autodiff::*;
use ndarray::prelude::*;

#[autodiff_reverse(d_dup_example,  Duplicated,Active)]
fn example_dup<'a>(v: ArrayView1<'a, f64>) -> f64 {
    v.iter().map(|&x| x.powi(2)).sum()
}

pub fn run_ndarray() {
    let mut data = Array1::from_vec(vec![1.0, 2.0, 3.1, 2.333, 44.2, -1.0]);  // Mutable Array1 для adj
    let view = data.view();  // Immutable view для primal

    let mut adj = Array1::zeros(data.len());  // Shadow для адъюнктов
    let mut adj_view = adj.view_mut();  // Mutable view для Duplicated

    // Вызов: view (primal), seed, adj_view (shadow) — ошибка компиляции из-за activities для view
    let val = d_dup_example(view, adj_view, 1.0);

    println!("Value: {:?}", val);
    println!("Gradient: {:?}", adj);  // Должно быть [2.0, 4.0, 6.2, ...]
}