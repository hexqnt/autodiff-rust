use crate::{data, plot};
use argmin::core::{CostFunction, Error, Executor, Gradient};
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use std::autodiff::*;
use std::cell::RefCell;
use std::f32::consts::PI;

const IDX_A: usize = 0;
const IDX_B: usize = 1;
const IDX_P: usize = 2;
const IDX_C: usize = 3;

/// Модель:
/// T_model(t) = a * sin(2 * PI * (t + c) / p) + b
/// t — номер точки (в днях), a — амплитуда, b — средний уровень, p — период, c — сдвиг (в днях).
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

struct Cache {
    params: Vec<f32>,
    loss: f32,
    grad: Vec<f32>,
}

struct RegressionProblem<'a> {
    temps: &'a [f32],
    cache: RefCell<Option<Cache>>,
}

impl<'a> CostFunction for RegressionProblem<'a> {
    type Param = Vec<f32>;
    type Output = f32;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        if let Some(cache) = self.cache.borrow().as_ref()
            && cache.params.as_slice() == param {
                return Ok(cache.loss);
            }

        let [a, b, p, c]: [f32; 4] = param
            .as_slice()
            .try_into()
            .map_err(|_| Error::msg("ожидаю [a, b, p, c] длиной 4"))?;
        let (loss, da, db, dp, dc) = d_sse(self.temps, a, b, p, c, 1.0);
        let grad = vec![da, db, dp, dc];
        self.cache.replace(Some(Cache {
            params: param.clone(),
            loss,
            grad: grad.clone(),
        }));
        Ok(loss)
    }
}

impl<'a> Gradient for RegressionProblem<'a> {
    type Param = Vec<f32>;
    type Gradient = Vec<f32>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error> {
        if let Some(cache) = self.cache.borrow().as_ref()
            && cache.params.as_slice() == param {
                return Ok(cache.grad.clone());
            }

        let [a, b, p, c]: [f32; 4] = param
            .as_slice()
            .try_into()
            .map_err(|_| Error::msg("ожидаю [a, b, p, c] длиной 4"))?;
        let (loss, da, db, dp, dc) = d_sse(self.temps, a, b, p, c, 1.0);
        let grad = vec![da, db, dp, dc];
        self.cache.replace(Some(Cache {
            params: param.clone(),
            loss,
            grad: grad.clone(),
        }));
        Ok(grad)
    }
}

pub fn run() -> Result<(), Error> {
    // Дневные температуры Петербурга.
    let temps = data::SPB_TEMP.as_slice();

    // Минимальный пример: argmin + steepest descent с line search.
    let problem = RegressionProblem {
        temps,
        cache: RefCell::new(None),
    };

    let mean = temps.iter().sum::<f32>() / temps.len() as f32;

    // Стартуем фазу в 0, и предварительно оценкой среднего, оптимизатор найдёт сдвиг по данным.
    let init = vec![1.0_f32, mean, 360.0_f32, 0.0_f32];
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
        "loss ={:.4}, a ≈ {:.3}, b ≈ {:.3}, p ≈ {:.3}, c ≈ {:.3}",
        state.cost,
        best[IDX_A],
        best[IDX_B],
        best[IDX_P],
        best[IDX_C]
    );

    // Визуализация: отдельные точки и точки+модель (fit.svg).
    plot::save_plot_with_model(temps, |t| {
        model(t, best[IDX_A], best[IDX_B], best[IDX_P], best[IDX_C])
    })?;
    let out_dir = std::env::current_dir().map_err(|e| Error::msg(e.to_string()))?;
    let points = out_dir.join(plot::POINTS_SVG);
    let fit = out_dir.join(plot::FIT_SVG);
    println!(
        "графики сохранены: {} и {}",
        points.display(),
        fit.display()
    );

    Ok(())
}
