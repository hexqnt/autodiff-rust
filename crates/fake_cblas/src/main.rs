#![feature(autodiff)]
use std::autodiff::*;

use core::ffi::c_int;
use faer::linalg::matmul::matmul;
use faer::mat;
use faer::prelude::ReborrowMut;
use faer::{Accum, Par};
use rand::Rng;

// Минимальные CBLAS enum-ы с числовыми значениями как в cblas.h
// CBLAS_LAYOUT { RowMajor=101, ColMajor=102 }
// CBLAS_TRANSPOSE { NoTrans=111, Trans=112, ConjTrans=113 (не используем) }
#[repr(C)]
#[derive(Copy, Clone)]
pub enum CBLAS_LAYOUT {
    RowMajor = 101,
    ColMajor = 102,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub enum CBLAS_TRANSPOSE {
    NoTrans = 111,
    Trans = 112,
}

// Утилиты: сбор матричных видов из «сырой» памяти со stride.
// Конструкторы предоставляет faer-core.
unsafe fn matref_f64(
    ptr: *const f64,
    rows: usize,
    cols: usize,
    row_stride: isize,
    col_stride: isize,
) -> mat::MatRef<'static, f64> {
    unsafe { mat::MatRef::from_raw_parts(ptr, rows, cols, row_stride, col_stride) }
}

unsafe fn matmut_f64(
    ptr: *mut f64,
    rows: usize,
    cols: usize,
    row_stride: isize,
    col_stride: isize,
) -> mat::MatMut<'static, f64> {
    unsafe { mat::MatMut::from_raw_parts_mut(ptr, rows, cols, row_stride, col_stride) }
}

fn scale_matrix<'a>(mut view: mat::MatMut<'a, f64>, beta: f64) {
    if beta == 1.0 {
        return;
    }

    if beta == 0.0 {
        faer::zip!(view.rb_mut().as_dyn_mut()).for_each(|faer::unzip!(elem)| {
            *elem = 0.0;
        });
    } else {
        faer::zip!(view.rb_mut().as_dyn_mut()).for_each(|faer::unzip!(elem)| {
            *elem *= beta;
        });
    }
}

/// Реализация `cblas_dgemm` с ABI C: вычисляет матричное произведение
/// `C := alpha * op(A) * op(B) + beta * C` для матриц плотного формата в духе CBLAS.
///
/// Аргументы полностью повторяют традиционный интерфейс:
/// - `layout` указывает раскладку памяти (`RowMajor` или `ColMajor`);
/// - `transa` и `transb` определяют, применяется ли транспонирование к `A` и `B`;
/// - `m`, `n`, `k` задают размеры результирующей матрицы и внутреннее измерение;
/// - `alpha` и `beta` — скаляры, масштабирующие произведение и исходное содержимое `C`;
/// - `a`, `b`, `c` — указатели на буферы с элементами матриц;
/// - `lda`, `ldb`, `ldc` — ведущие размеры (число элементов между соседними строками/столбцами).
///
/// Для `RowMajor` ведущий размер соответствует длине строки, для `ColMajor` — длине столбца.
/// Параметры `trans*` задают `op(X)` как идентичность либо транспонирование.
///
/// # Safety
/// Вызывающая сторона обязана обеспечить:
/// - корректные размеры выделенной памяти;
/// - валидные и выровненные указатели `a`, `b`, `c` на достаточное число элементов `f64`;
/// - отсутствие aliasing.
///
/// Символ помечен `#[unsafe(no_mangle)]`, чтобы его обнаруживал Enzyme.
// #[unsafe(no_mangle)]
// #[inline(never)]
// pub unsafe extern "C" fn cblas_dgemm(
//     layout: CBLAS_LAYOUT,
//     transa: CBLAS_TRANSPOSE,
//     transb: CBLAS_TRANSPOSE,
//     m: c_int,
//     n: c_int,
//     k: c_int,
//     alpha: f64,
//     a: *const f64,
//     lda: c_int,
//     b: *const f64,
//     ldb: c_int,
//     beta: f64,
//     c: *mut f64,
//     ldc: c_int,
// ) {
//     let (m, n, k) = (m as usize, n as usize, k as usize);

//     // Разворачиваем раскладку и транспонирования в размеры и страйды.
//     let (a_rows, a_cols, a_rs, a_cs, a_t) = match (layout, transa) {
//         (CBLAS_LAYOUT::ColMajor, CBLAS_TRANSPOSE::NoTrans) => (m, k, 1, lda as isize, false),
//         (CBLAS_LAYOUT::ColMajor, CBLAS_TRANSPOSE::Trans) => (k, m, 1, lda as isize, true),
//         (CBLAS_LAYOUT::RowMajor, CBLAS_TRANSPOSE::NoTrans) => (k, m, lda as isize, 1, true),
//         (CBLAS_LAYOUT::RowMajor, CBLAS_TRANSPOSE::Trans) => (m, k, lda as isize, 1, false),
//     };
//     let (b_rows, b_cols, b_rs, b_cs, b_t) = match (layout, transb) {
//         (CBLAS_LAYOUT::ColMajor, CBLAS_TRANSPOSE::NoTrans) => (k, n, 1, ldb as isize, false),
//         (CBLAS_LAYOUT::ColMajor, CBLAS_TRANSPOSE::Trans) => (n, k, 1, ldb as isize, true),
//         (CBLAS_LAYOUT::RowMajor, CBLAS_TRANSPOSE::NoTrans) => (n, k, ldb as isize, 1, true),
//         (CBLAS_LAYOUT::RowMajor, CBLAS_TRANSPOSE::Trans) => (k, n, ldb as isize, 1, false),
//     };
//     let (c_rows, c_cols, c_rs, c_cs) = match layout {
//         CBLAS_LAYOUT::ColMajor => (m, n, 1, ldc as isize),
//         CBLAS_LAYOUT::RowMajor => (n, m, ldc as isize, 1),
//     };

//     // Виды матриц
//     let mut c_view = unsafe { matmut_f64(c, c_rows, c_cols, c_rs, c_cs) };
//     let mut a_view = unsafe { matref_f64(a, a_rows, a_cols, a_rs, a_cs) };
//     let mut b_view = unsafe { matref_f64(b, b_rows, b_cols, b_rs, b_cs) };

//     if a_t {
//         a_view = a_view.transpose();
//     }
//     if b_t {
//         b_view = b_view.transpose();
//     }

//     // C := alpha*A*B + beta*C
//     let accum = if beta == 0.0 {
//         Accum::Replace
//     } else {
//         scale_matrix(c_view.rb_mut(), beta);
//         Accum::Add
//     };

//     matmul(c_view.rb_mut(), accum, a_view, b_view, alpha, Par::Seq);
// }

fn print_col_major(label: &str, data: &[f64], rows: usize, cols: usize) {
    println!("{label} ({rows}x{cols}):");
    for row in 0..rows {
        print!("    [");
        for col in 0..cols {
            let value = data[row + col * rows];
            if col + 1 == cols {
                print!("{value:8.3}");
            } else {
                print!("{value:8.3}, ");
            }
        }
        println!(" ]");
    }
}

fn main() {
    let mut rng = rand::rng();

    let m = rng.random_range(1..=4);
    let n = rng.random_range(1..=4);
    let k = rng.random_range(1..=4);

    let alpha = rng.random_range(-2.0..=2.0);
    let beta = rng.random_range(-1.0..=1.0);

    let a: Vec<f64> = (0..m * k).map(|_| rng.random_range(-5.0..=5.0)).collect();
    let b: Vec<f64> = (0..k * n).map(|_| rng.random_range(-5.0..=5.0)).collect();
    let mut c: Vec<f64> = (0..m * n).map(|_| rng.random_range(-5.0..=5.0)).collect();
    let c_initial = c.clone();

    // Ручная наивная реализация gemm для проверки результатов
    let expected = {
        let mut out = vec![0.0; m * n];
        for col in 0..n {
            for row in 0..m {
                let mut accum = 0.0;
                for p in 0..k {
                    accum += a[row + p * m] * b[p + col * k];
                }
                out[row + col * m] = alpha * accum + beta * c_initial[row + col * m];
            }
        }
        out
    };

    my_gemm(m, n, k, alpha, beta, &a, &b, &mut c);
    let mut da = vec![0.0; a.len()];
    let mut db = vec![0.0; b.len()];
    let mut dc = vec![1.0; c.len()]; // пример: L = sum(C)

    let (d_alpha, d_beta) = my_gemm_grad(
        m, n, k, alpha, beta, &a, &mut da, &b, &mut db,
        &mut c,  // как и раньше, сюда запишется результат GEMM
        &mut dc, // сюда Enzyme будет накапливать dL/dC (инициализируешь сам)
    );
    // for (actual, reference) in c.iter().zip(expected.iter()) {
    //     assert!((actual - reference).abs() < 1e-9);
    // }

    println!("alpha={d_alpha:.3}, beta={d_beta:.3}");

    println!("Случайные размеры: m={m}, n={n}, k={k}");
    println!("alpha={alpha:.3}, beta={beta:.3}");
    print_col_major("A", &a, m, k);
    print_col_major("B", &b, k, n);
    print_col_major("C (результат)", &c, m, n);
}

#[unsafe(no_mangle)]
#[inline(never)]
pub unsafe extern "C" fn cblas_dgemm(
    layout: CBLAS_LAYOUT,
    transa: CBLAS_TRANSPOSE,
    transb: CBLAS_TRANSPOSE,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: f64,
    a: *const f64,
    lda: c_int,
    b: *const f64,
    ldb: c_int,
    beta: f64,
    c: *mut f64,
    ldc: c_int,
) {
    debug_assert!(m >= 0 && n >= 0 && k >= 0);
    debug_assert!(lda >= 0 && ldb >= 0 && ldc >= 0);
}

#[autodiff_reverse(
    my_gemm_grad,
    Const,
    Const,
    Const,
    Active,
    Active,
    Duplicated,
    Duplicated,
    Duplicated
)]
fn my_gemm(
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    beta: f64,
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
) {
    unsafe {
        cblas_dgemm(
            CBLAS_LAYOUT::ColMajor,
            CBLAS_TRANSPOSE::NoTrans,
            CBLAS_TRANSPOSE::NoTrans,
            m as c_int,
            n as c_int,
            k as c_int,
            alpha,
            a.as_ptr(),
            m as c_int,
            b.as_ptr(),
            k as c_int,
            beta,
            c.as_mut_ptr(),
            m as c_int,
        );
    }
}
