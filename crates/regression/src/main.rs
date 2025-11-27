#![feature(autodiff)]

mod data;
mod plot;

use argmin::core::{CostFunction, Error, Executor, Gradient};
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use std::autodiff::*;
use std::cell::RefCell;
use std::f32::consts::PI;

/// Модель:
///   T_model(t) = a * sin(2π (t + phase) / period) + b
/// t — номер точки (в днях), a — амплитуда, b — средний уровень, period — периодичность, phase — сдвиг (в днях).
fn model(i: f32, a: f32, b: f32, p: f32, c: f32) -> f32 {
    let arg = 2.0 * PI * (i + c) / p;
    a * arg.sin() + b
}

#[autodiff_reverse(d_sse, Const, Active, Active, Active, Active, Active)]
fn sse_loss(temps: &[f32], a: f32, b: f32, p: f32, c: f32) -> f32 {
    temps
        .iter()
        .enumerate()
        .map(|(i, &temp)| {
            let r = model(i as f32, a, b, p, c) - temp;
            r * r
        })
        .sum()
}

struct RegressionProblem<'a> {
    temps: &'a [f32],
    cache: RefCell<Option<(Vec<f32>, f32, Vec<f32>)>>,
}

impl<'a> CostFunction for RegressionProblem<'a> {
    type Param = Vec<f32>;
    type Output = f32;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        if let Some((p, loss, _)) = self.cache.borrow().as_ref() {
            if p == param {
                return Ok(*loss);
            }
        }

        let [a, b, p, phase]: [f32; 4] = param
            .as_slice()
            .try_into()
            .map_err(|_| Error::msg("ожидаю [a, b, period, phase] длиной 4"))?;
        let (loss, da, db, dp, dphase) = d_sse(self.temps, a, b, p, phase, 1.0);
        let grad = vec![da, db, dp, dphase];
        self.cache
            .replace(Some((param.clone(), loss, grad.clone())));
        Ok(loss)
    }
}

impl<'a> Gradient for RegressionProblem<'a> {
    type Param = Vec<f32>;
    type Gradient = Vec<f32>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error> {
        if let Some((p, _, g)) = self.cache.borrow().as_ref() {
            if p == param {
                return Ok(g.clone());
            }
        }

        let [a, b, p, c]: [f32; 4] = param
            .as_slice()
            .try_into()
            .map_err(|_| Error::msg("ожидаю [a, b, period, phase] длиной 4"))?;
        let (loss, da, db, dp, dc) = d_sse(self.temps, a, b, p, c, 1.0);
        let grad = vec![da, db, dp, dc];
        self.cache
            .replace(Some((param.clone(), loss, grad.clone())));
        Ok(grad)
    }
}

fn main() -> Result<(), Error> {
    // Дневные температуры Петербурга.
    let temps = data::SPB_TEMP.as_slice();

    // Минимальный пример: argmin + steepest descent с line search.
    let problem = RegressionProblem {
        temps,
        cache: RefCell::new(None),
    };
    // Стартуем фазу в 0, оптимизатор найдёт сдвиг по данным.
    let init = vec![1.0_f32, 1.0_f32, 360.0_f32, 0.0_f32];
    let solver = SteepestDescent::new(MoreThuenteLineSearch::new());

    let res = Executor::new(problem, solver)
        .configure(|state| state.param(init).max_iters(50_000))
        .run()?;

    let state = res.state();
    let best = state
        .best_param
        .as_ref()
        .unwrap_or_else(|| state.param.as_ref().expect("optimizer produced param"));

    println!(
        "best: loss={:.4}, a ≈ {:.3}, b ≈ {:.3}, period ≈ {:.3}, phase ≈ {:.3}",
        state.cost, best[0], best[1], best[2], best[3]
    );

    // Визуализация: отдельные точки и точки+модель (fit.svg).
    plot::save_plot_with_model(temps, |t| model(t, best[0], best[1], best[2], best[3]))?;
    println!("saved plots: points.svg and fit.svg");

    Ok(())
}
