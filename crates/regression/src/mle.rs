use crate::{data, plot};
use argmin::core::{CostFunction, Error, Executor, Gradient};
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::condition::ArmijoCondition;
use argmin::solver::linesearch::BacktrackingLineSearch;
use argmin::solver::quasinewton::LBFGS;
use std::autodiff::*;
use std::cell::RefCell;
use std::f32::consts::PI;

const P_MIN: f32 = 300.0;
const P_MAX: f32 = 400.0;

// Индексы параметров модели.
const IDX_A1S: usize = 0;
const IDX_A1C: usize = 1;
const IDX_A2S: usize = 2;
const IDX_A2C: usize = 3;
const IDX_B: usize = 4;
const IDX_B_TREND: usize = 5;
const IDX_P: usize = 6;
const IDX_V0: usize = 7;
const IDX_V1S: usize = 8;
const IDX_V1C: usize = 9;
const IDX_V_TREND: usize = 10;
const PARAM_LEN: usize = 11;

#[derive(Clone, Copy)]
struct TimeNorm {
    center: f32,
    scale: f32,
}

impl TimeNorm {
    fn new(len: usize, p: f32) -> Self {
        let center = (len as f32 - 1.0) * 0.5;
        let scale = p.abs().max(1.0);
        Self { center, scale }
    }

    fn normalize(self, t: f32) -> f32 {
        (t - self.center) / self.scale
    }
}
/// ================================================
///  Модель
/// ================================================
/// Среднее задаётся гармониками в форме Фурье.
fn mean_model(
    i: f32,
    a1s: f32,
    a1c: f32,
    a2s: f32,
    a2c: f32,
    b: f32,
    p: f32,
) -> f32 {
    let arg1 = 2.0 * PI * i / p;
    let arg2 = 4.0 * PI * i / p;
    let (s1, c1) = arg1.sin_cos();
    let (s2, c2) = arg2.sin_cos();
    b + a1s * s1 + a1c * c1 + a2s * s2 + a2c * c2
}

/// ================================================
///  Модель
/// ================================================
/// Лог-дисперсия задаётся гармоникой в форме Фурье.
fn log_sigma2_raw(i: f32, v0: f32, v1s: f32, v1c: f32, p: f32) -> f32 {
    let arg = 2.0 * PI * i / p;
    let (s1, c1) = arg.sin_cos();
    v0 + v1s * s1 + v1c * c1
}

/// ================================================
///  Модель
/// ================================================
/// Автоматическое дифференцирование функции потерь. Возвращает значение функции потерь и градиент.
#[autodiff_reverse(
    d_nll,
    Const,
    Duplicated,
    Active
)]
fn nll_loss(temps: &[f32], params: &[f32]) -> f32 {
    let params: &[f32; PARAM_LEN] = params
        .try_into()
        .expect("ожидаю 11 параметров для nll_loss");
    let [a1s, a1c, a2s, a2c, b, b_trend, p, v0, v1s, v1c, v_trend] = *params;
    // Нормируем время, чтобы тренд имел адекватный масштаб.
    let time_norm = TimeNorm::new(temps.len(), p);
    let data_loss: f32 = temps
        .iter()
        .enumerate()
        .map(|(i, &temp)| {
            let i = i as f32;
            let t_norm = time_norm.normalize(i);
            let mu = mean_model(i, a1s, a1c, a2s, a2c, b, p) + b_trend * t_norm;
            let log_var = log_sigma2_raw(i, v0, v1s, v1c, p) + v_trend * t_norm;
            let var = log_var.exp();
            let r = mu - temp;
            log_var + (r * r) / var
        })
        .sum();
    data_loss
}

struct Cache {
    params: Vec<f32>,
    loss: f32,
    grad: Vec<f32>,
}

struct MleProblem<'a> {
    temps: &'a [f32],
    cache: RefCell<Option<Cache>>,
}

impl<'a> CostFunction for MleProblem<'a> {
    type Param = Vec<f32>;
    type Output = f32;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        if let Some(cache) = self.cache.borrow().as_ref()
            && cache.params.as_slice() == param {
                return Ok(cache.loss);
            }

        if param.len() != PARAM_LEN {
            return Err(Error::msg(
                "ожидаю [a1s, a1c, a2s, a2c, b, b_trend, p, v0, v1s, v1c, v_trend] длиной 11",
            ));
        }
        let mut grad = vec![0.0_f32; param.len()];
        let loss = d_nll(self.temps, param.as_slice(), grad.as_mut_slice(), 1.0);
        self.cache.replace(Some(Cache {
            params: param.clone(),
            loss,
            grad: grad.clone(),
        }));
        Ok(loss)
    }

}

impl<'a> Gradient for MleProblem<'a> {
    type Param = Vec<f32>;
    type Gradient = Vec<f32>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error> {
        if let Some(cache) = self.cache.borrow().as_ref()
            && cache.params.as_slice() == param {
                return Ok(cache.grad.clone());
            }

        if param.len() != PARAM_LEN {
            return Err(Error::msg(
                "ожидаю [a1s, a1c, a2s, a2c, b, b_trend, p, v0, v1s, v1c, v_trend] длиной 11",
            ));
        }
        let mut grad = vec![0.0_f32; param.len()];
        let loss = d_nll(self.temps, param.as_slice(), grad.as_mut_slice(), 1.0);
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

    let problem = MleProblem {
        temps,
        cache: RefCell::new(None),
    };
    // Среднее и дисперсия для инициализации.
    let mean = temps.iter().sum::<f32>() / temps.len() as f32;
    let var = temps
        .iter()
        .map(|&x| {
            let d = x - mean;
            d * d
        })
        .sum::<f32>()
        / temps.len() as f32;
    let v0_init = var.max(1e-6).ln();

    // Инициализация параметров модели c некоторыми эвристиками.
    let init = vec![
        0.0_f32,  // a1s
        0.0_f32,   // a1c
        0.0_f32,   // a2s
        0.0_f32,   // a2c
        mean,      // b
        0.0_f32,   // b_trend
        360.0_f32, // p
        v0_init,   // v0
        0.5_f32,   // v1s
        0.5_f32,   // v1c
        0.0_f32,   // v_trend
    ];

    //  Настройка стратегии шага оптимизатора.
    let linesearch = BacktrackingLineSearch::new(ArmijoCondition::new(1e-4)?);

    // Создание и запуск оптимизатора.

    // Почему LBFGS выбивает по памяти при компиляции. Используем SteepestDescent.
    // let solver = LBFGS::new(linesearch, 10);
    let solver = SteepestDescent::new(linesearch);



    let res = Executor::new(problem, solver)
        .configure(|state| state.param(init).max_iters(200))
        .run()?;

    let state = res.state();

    // Лучшие зафиченные параметры.
    let best = state
        .best_param
        .as_ref()
        .unwrap_or_else(|| state.param.as_ref().expect("optimizer produced param"));

    // Вывод результатов.
    let p = best[IDX_P].clamp(P_MIN, P_MAX);
    let b_trend = best[IDX_B_TREND];
    let v_trend = best[IDX_V_TREND];
    let amp1 = (best[IDX_A1S] * best[IDX_A1S] + best[IDX_A1C] * best[IDX_A1C]).sqrt();
    let amp2 = (best[IDX_A2S] * best[IDX_A2S] + best[IDX_A2C] * best[IDX_A2C]).sqrt();
    let mut min_mu = f32::INFINITY;
    let mut max_mu = f32::NEG_INFINITY;
    let mut min_log_var = f32::INFINITY;
    let mut max_log_var = f32::NEG_INFINITY;
    let samples = temps.len().min(720);
    let time_norm = TimeNorm::new(temps.len(), p);
    let t_max = (temps.len() - 1) as f32;
    for idx in 0..=samples {
        let t = t_max * (idx as f32) / (samples as f32);
        let t_norm = time_norm.normalize(t);
        let mu = mean_model(
            t,
            best[IDX_A1S],
            best[IDX_A1C],
            best[IDX_A2S],
            best[IDX_A2C],
            best[IDX_B],
            p,
        )
            + b_trend * t_norm;
        let log_var = log_sigma2_raw(t, best[IDX_V0], best[IDX_V1S], best[IDX_V1C], p)
            + v_trend * t_norm;
        if mu < min_mu {
            min_mu = mu;
        }
        if mu > max_mu {
            max_mu = mu;
        }
        if log_var < min_log_var {
            min_log_var = log_var;
        }
        if log_var > max_log_var {
            max_log_var = log_var;
        }
    }
    let sigma_min = (0.5 * min_log_var).exp();
    let sigma_max = (0.5 * max_log_var).exp();
    println!("оценка MLE:");
    println!("  loss: {:.4}", state.cost);
    println!("  mean (Фурье, k=1..2):");
    println!("    b (средний уровень): {:.3}", best[IDX_B]);
    println!("    p (период, дни): {:.3}", p);
    println!(
        "    k=1: a1s={:.3}, a1c={:.3} (ампл.~{:.3})",
        best[IDX_A1S], best[IDX_A1C], amp1
    );
    println!(
        "    k=2: a2s={:.3}, a2c={:.3} (ампл.~{:.3})",
        best[IDX_A2S], best[IDX_A2C], amp2
    );
    println!("    b_trend тренд (на период): {:.4}", b_trend);
    println!("  лог-дисперсия:");
    println!(
        "    v0={:.3}, v1s={:.3}, v1c={:.3}",
        best[IDX_V0], best[IDX_V1S], best[IDX_V1C]
    );
    println!("    v_trend тренд (на период): {:.4}", v_trend);
    println!(
        "  производные оценки: диапазон среднего~[{:.2}, {:.2}] (дельта~{:.2}), диапазон sigma~[{:.2}, {:.2}] C",
        min_mu,
        max_mu,
        max_mu - min_mu,
        sigma_min,
        sigma_max
    );

    // Визуализация: отдельные точки и точки+модель (fit.svg).
    plot::save_plot_with_model(temps, |t| {
        let t_norm = time_norm.normalize(t);
        mean_model(
            t,
            best[IDX_A1S],
            best[IDX_A1C],
            best[IDX_A2S],
            best[IDX_A2C],
            best[IDX_B],
            p,
        ) + b_trend * t_norm
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
