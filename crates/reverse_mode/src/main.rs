/// Небольшая лента значений, которые понадобятся на обратном проходе.
#[derive(Debug, Clone, Copy)]
pub struct Tape {
    x: f64,
    branch_taken: bool,
    cos_x: f64,
}

/// Прямой проход: считаем значение f(x, y) и запоминаем всё нужное для reverse pass.
pub fn primal_with_tape(x: f64, y: f64) -> (f64, Tape) {
    let product = x * y;
    let branch_taken = x > y;
    let sin_term = if branch_taken { x.sin() } else { 0.0 };
    let cos_x = if branch_taken { x.cos() } else { 0.0 };
    let value = product + sin_term;

    let tape = Tape {
        x,
        branch_taken,
        cos_x,
    };

    (value, tape)
}

/// Обратный проход: получаем дифференциалы по x и y из seed_df.
pub fn reverse_from_tape(t: &Tape, seed_df: f64, y: f64) -> (f64, f64) {
    // Сумма передаёт градиент обоим слагаемым как есть.
    let d_product = seed_df;
    let mut dx = d_product * y;
    let dy = d_product * t.x;

    if t.branch_taken {
        let d_sin = seed_df;
        dx += d_sin * t.cos_x;
    }

    (dx, dy)
}

fn main() {
    let x = 2.0;
    let y = 1.0;
    let (value, tape) = primal_with_tape(x, y);
    let (dx, dy) = reverse_from_tape(&tape, 1.0, y);

    println!("f({x}, {y}) = {value}");
    println!("df/dx = {dx}, df/dy = {dy}");
}

#[cfg(test)]
mod tests {
    use super::{primal_with_tape, reverse_from_tape};

    #[test]
    fn trig_branch_is_used_when_x_greater_than_y() {
        let x = 2.0;
        let y = 1.0;
        let (value, tape) = primal_with_tape(x, y);
        assert!((value - (x * y + x.sin())).abs() < 1e-12);

        let (dx, dy) = reverse_from_tape(&tape, 1.0, y);
        assert!((dx - (y + x.cos())).abs() < 1e-12);
        assert!((dy - x).abs() < 1e-12);
    }

    #[test]
    fn trig_branch_skipped_when_x_not_greater_than_y() {
        let x = 0.5;
        let y = 1.0;
        let (value, tape) = primal_with_tape(x, y);
        assert!((value - x * y).abs() < 1e-12);

        let (dx, dy) = reverse_from_tape(&tape, 1.0, y);
        assert!((dx - y).abs() < 1e-12);
        assert!((dy - x).abs() < 1e-12);
    }
}
