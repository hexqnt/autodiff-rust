mod dual_numbers;
mod naive_dual_numbers;

use dual_numbers::{Dual, variables};
use naive_dual_numbers::NaiveDual;

fn naive_forward_pass() {
    let f = |x: NaiveDual, y: NaiveDual| x * y + x.sin() * y;

    let x0 = 2.0;
    let y0 = 1.0;

    // Дифференциал по x: x — переменная, y — константа.
    let x_var = NaiveDual::variable(x0);
    let y_const = NaiveDual::constant(y0);
    let result_dx = f(x_var, y_const);

    // Дифференциал по y: x — константа, y — переменная.
    let x_const = NaiveDual::constant(x0);
    let y_var = NaiveDual::variable(y0);
    let result_dy = f(x_const, y_var);
}

fn dual_forward_pass() {
    let f = |x: Dual<2>, y: Dual<2>| x * y + x.sin() * y;
    let [x, y] = variables([2.0, 1.0]);

    // Дифференциал сразу по y и x
    let result = f(x, y);
}

fn main() {
    naive_forward_pass();
    dual_forward_pass();
}
