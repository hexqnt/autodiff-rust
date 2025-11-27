use std::autodiff::*;

/// Простая модель: f(x) = (a * x + b)^2
struct Model {
    a: f32,
    b: f32,
}

/// Enzyme пока не умеет ставить #[autodiff] прямо на методы, поэтому делаем
/// свободную функцию и тонкие обёртки в impl Model.
#[autodiff_forward(model_loss_grad, Const, Dual, Dual)]
fn model_loss(model: &Model, x: f32) -> f32 {
    let y = model.a * x + model.b;
    y * y
}

impl Model {
    fn loss(&self, x: f32) -> f32 {
        model_loss(self, x)
    }

    fn loss_grad(&self, x: f32, dloss: f32) -> (f32, f32) {
        model_loss_grad(self, x, dloss)
    }
}

pub fn run() {
    let m = Model { a: 2.0, b: 1.0 };
    let x = 3.0;

    // Сигнатура сгенерированного метода:
    // fn loss_grad(&self, x: f32, dloss: f32) -> (f32, f32)
    //
    // Первый возвращаемый элемент — значение primal (loss),
    // второй — производная по единственному Active-аргументу (x).
    // dloss = 1.0 — "seed" для d(loss)/d(output).
    let (loss, dloss_dx) = m.loss_grad(x, 1.0);

    println!("loss = {loss}, dloss/dx = {dloss_dx}");
}
