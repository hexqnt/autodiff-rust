use std::autodiff::*;
use std::cmp::max_by_key;

// #[autodiff_re(d_floor, Dual, Dual)]
pub fn argmax(xs: &[f64]) -> Option<usize> {
    xs.iter()
        .enumerate()
        .filter(|(_, x)| !x.is_nan())
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
}
