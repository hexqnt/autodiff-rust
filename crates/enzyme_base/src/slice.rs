use std::autodiff::*;

#[autodiff_reverse(d_dup_example, Duplicated, Active)] // Active для return, Duplicated для arg (срез)
fn example_dup(v: &[f64]) -> f64 {
    v.iter().map(|&x| x.powi(2)).sum()
}

pub fn run() {
    let mut data = vec![1.0, 2.0, 3.1, 2.333, 44.2, -1.0]; // Mutable Vec для adj
    let slice = data.as_slice(); // Срез от Vec (immutable для primal)

    let mut adj = vec![0.0; data.len()]; // Shadow для адъюнктов (градиентов)
    let mut adj_slice = adj.as_mut_slice(); // &mut срез для Duplicated

    let val = d_dup_example(slice, adj_slice, 1.0); // seed=1.0; возвращает primal value, обновляет adj_slice

    println!("Value: {:?}", val); // 1^2 + 2^2 = 5.0
    println!("Gradient: {:?}", adj); // [2.0, 4.0] (2*x для каждого)
}
