#![feature(autodiff)]
mod array;
mod generic;
mod method;
mod scalar;
mod slice;
mod unsafe_rev_mode;

fn main() {
    scalar::run();
    array::run();
    slice::run();
    generic::run();
    method::run();
    unsafe_rev_mode::run();
}
