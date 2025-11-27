#![feature(autodiff)]
mod branch_equal_zero;
mod non_smoth_func;
mod round_floor_ceil;

fn main() {
    branch_equal_zero::run();
    non_smoth_func::run();
    round_floor_ceil::run();
}
