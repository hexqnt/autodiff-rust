#![feature(autodiff)]

mod data;
mod mle;
mod plot;
mod simple;

use argmin::core::Error;

fn main() -> Result<(), Error> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        None | Some("simple") => simple::run(),
        Some("mle") | Some("complex") => mle::run(),
        Some(other) => {
            eprintln!("Неизвестный пример: {other}. Используй `simple` или `mle`.");
            Ok(())
        }
    }
}
