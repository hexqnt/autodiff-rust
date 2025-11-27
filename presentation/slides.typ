#import "@preview/polylux:0.4.0": *
#import "@preview/metropolis-polylux:0.1.0" as metropolis
#import "@preview/mannot:0.3.0": *
#import "@local/codly:1.3.1": codly, codly-init
#import "@preview/codly-languages:0.1.1": *
#import metropolis: focus, new-section

// #set page(
//   footer: align(right + bottom, [
//     #toolbox.slide-number / #toolbox.last-slide-number из доклада
//   ]),
//   footer-descent: 100pt,   // поднимает футер от нижнего поля
// )

// Широкая горизонтальная полоска с метками и подписями
#let scalebar(width: 100%, bar: 6pt, tick: 1.5pt, gap: 30pt) = box(width: width, height: 24pt, inset: 0pt)[
  // сама полоса
  #line(start: (5%, 50%), end: (95%, 50%), stroke: bar)

  #place(bottom + left, dy: -gap)[Полное знание]
  #place(bottom + right, dy: -gap)[Чёрный ящик]
  // подписи снизу
  #place(bottom + left, dy: gap)[Аналитический подход]
  // #place(bottom + center, dy: gap)[Автодифференцирование]
  #place(bottom + right, dy: gap)[Конечная разность]
]

#show: metropolis.setup
#show raw.where(block: true): set text(size: 0.75em)

// #show: codly.codly-init.with()
#show: codly-init
#codly(zebra-fill: none, display-name: false, number-format: none, smart-indent: true, stroke: none)

#slide[
  #set page(header: none, footer: none, margin: 3em)


  #text(size: 1.3em)[
    *RustCon 2025*
  ]

  Автоматическое дифференцирование в Rust

  #metropolis.divider

  #set text(size: .8em, weight: "light")
  Кравченко Владимир

]

#slide[
  = Обо мне

  #toolbox.side-by-side(columns: (1fr, 1fr), gutter: 2em)[
    Python в области научных вычислений свыше 5 лет.

    Последние 3 года компилирую Rust.

    Область интересов: численные методы и машинное обучение.

  ][

    #align(center + top)[
      #link("https://github.com/hexqnt")[
        #image("img/my_git_qr_code.svg", width: 50%)
      ]
    ]
  ]
]

#slide[
  = Попытка применить автодиф в проде
  #align(center)[

    #image("img/cdagt.svg", height: 150%)
  ]

]

#slide[
  = Rust Project goals
  *2024h2* - #link("https://rust-lang.github.io/rust-project-goals/2024h2/Rust-for-SciComp.html#expose-experimental-llvm-features-for-automatic-differentiation-and-gpu-offloading")[Expose experimental LLVM features for automatic differentiation and GPU offloading]

  *2025h2* - #link("https://rust-lang.github.io/rust-project-goals/2025h2/finishing-gpu-offload.html")[Finish the `std::offload` module]

  - `std::batching`
  - #text(weight: "bold")[`std::autodiff`]
  - `std::offload`
]


#new-section[Что такое дифференцирование?]

#slide[
  = Дихотомия подходов

  #scalebar()
]

#slide[
  = Аналитический подход. Элементарные функции
  #set align(top)
  #toolbox.side-by-side(columns: (1fr, 1fr))[
    Элементарные функции:
    $
           (dif)/(dif x) c & = 0 \
         (dif)/(dif x) x^n & = n x^(n-1) \
      (dif)/(dif x) sin(x) & = cos x \
    $
  ][
    Векторно-матричные:
    $
      nabla_{bold(x)} (bold(x)^"T" bold(A) bold(x)) & = (bold(A) + bold(A)^"T") bold(x)
    $
  ]
  Правила:
  $
                (f g)'(x) & = f'(x) g(x) + f(x) g'(x) \
    (dif)/(dif x) f(g(x)) & = f'(g(x)) dot g'(x)
  $
]


#slide[
  = Аналитический подход: SymPy → Rust

  #set align(top)
  #toolbox.side-by-side(columns: (1fr, 1fr), gutter: 1em)[
    #box[ #image("img/sympy_logo.png", height: 2em) ]Шаг 1

    $f(bold(x)) = bold(x)^"T" bold(A) bold(x)$
    ```python
    A = MatrixSymbol("A", n, n)  # матрица
    x = MatrixSymbol("x", n, 1)  # вектор

    f = x.T * A * x # наша функция

    grad = derive_by_array(f, x) # градиент
    grad = simplify(grad) # пытаемся упростить
    print(grad)
    ```
    // #h(2.0em) #sym.arrow.b

    ```sh
    >> "A*x + A.T*x"
    ```
  ][

    #uncover(2)[

      #box[#image("img/rust_logo.svg", height: 2em)] Шаг 2

      $nabla_{bold(x)} f(bold(x)) &= (bold(A) + bold(A)^"T") bold(x)$
      ```rust
      pub fn f(
          a: ArrayView2<'_, f64>,
          x: ArrayView1<'_, f64>
      ) -> Array1<f64> {

          (&a + &a.t()).dot(&x)
      }
      ```

    ]
  ]

]


#slide[
  = Конечные разности
  // две колонки
  #toolbox.side-by-side(columns: (1fr, 1fr), gutter: 1em)[
    // Левая колонка: картинка + формулы
    // #image("img/left.png", width: 100%)
    #image("img/Finite_difference_method.svg", width: 100%)
  ][
    // Правая колонка: картинка + текст

    Прямая разность (сложность $n+1$):
    $
      markhl(f'(x), color: #teal) & approx (f(x+h) - f(x))/h
    $

    Центральная разность (сложность $2n$):
    $
      markhl(f'(x), color: #blue) & approx (f(x+h) - f(x-h))/(2h)
    $
  ]
]

#slide[
  = Конечные разности (минусы)
  // две колонки
  #toolbox.side-by-side(columns: (1fr, 1fr), gutter: 1em)[
    // Левая колонка: картинка + формулы
    Точность падает в уменьшением $h$
    #image("img/AbsoluteErrorNumericalDifferentiationExample.png", width: 100%)
  ][

    Прямая разность (точность $O(h)$):
    $
      markhl(f'(x), color: #blue) & approx (f(x+h) - f(x))/h
    $

    Центральная разность (точность $O(h^2)$):
    $
      markhl(f'(x), color: #red) & approx (f(x+h) - f(x-h))/(2h)
    $

  ]
]

#slide[
  = Автодифференцирование
  #scalebar()
  #align(center)[
    #sym.arrow.t \
    Автодифференцирование
  ]
  #v(2em)
  1. Работает с вашим кодом - не требует доп. формализации
  2. Машинная точность производной
  3. Стоимость как несколько вызовов исходной функции
]

#new-section[Автодифференцирование]

#slide[
  = Режимы автоматического дифференцирования

  - #text(weight: "bold")[Прямой режим] (forward mode, JVP — Jacobian–Vector Product)
  - #text(weight: "bold")[Обратный режим] (reverse mode, VJP — Vector–Jacobian Product)

]

#let hi(body, fill: aqua) = {
  // set text(white)
  set align(center)
  rect(
    fill: fill,
    // inset: 4pt,
    // radius: 4pt,
    [#body],
  )
}

#slide[
  = Forward mode: Арифметика дуальных чисел
  Обозначения:
  $
    x = x_0 + x_1 ε, #h(2em)
    y = y_0 + y_1 ε, #h(2em)
    #only((1, 2, 3, 5))[$ε^2 = 0$]#only(4)[#hi[$ε^2 = 0$]]
  $

  #uncover((2, 3, 4, 5))[
    Сложение:
    $
      (x_0 + x_1 ε) + (y_0 + y_1 ε) = (x_0 + y_0) + (x_1 + y_1) ε
    $
  ]
  #uncover((3, 4, 5))[
    Умножение:
    $
      (x_0 + x_1 ε) dot (y_0 + #only((3, 4))[$y_1$] #only(5)[#hi[$y_1$]]ε) & = \
      & = x_0 y_0 + x_0 y_1 ε + x_1 y_0 ε + #only(3)[$x_1 y_1 ε^2$]#only(4)[#hi[$cancel(x_1 y_1 ε^2)$]] = \
      & = (x_0 dot y_0) + (#only((3, 4))[$x_0 dot y_1$]#only(5)[#hi[$cancel(x_0 dot y_1)$]] + x_1 dot y_0) ε
    $
  ]
]

#slide[
  = Forward mode: Арифметика дуальных чисел
  Обозначения:
  $
    x = x_0 + x_1 ε, #h(2em)
    y = y_0 + y_1 ε, #h(2em)
    ε^2 = 0
  $

  Правило для функций:
  $
    f(x_0 + x_1ε) = f(x_0) + f'(x_0) x_1 ε
  $

  #uncover(2)[
    Примеры:
    $
      // exp(x) = & exp(x_0) + exp(x_0) x_1 ε \
      sin(x) = & sin(x_0) + cos(x_0) x_1 ε \
      log(x) = & log(x_0) + (x_1 / x_0) ε \
      //  x^n = & x_0^n + n x_0^(n-1) x_1 ε \
    $
  ]
]


#slide[
  = Forward mode: Реализация на Rust
  // #text(size: 20pt)[

  #toolbox.side-by-side(columns: (1fr, 1fr), gutter: 2em)[

    #set align(top)
    #align(center)[
      ```rust
      struct Dual {
        p: f64,  // Значение ("primal")
        d: f64,  // Производная ("tangent")
      }
      ```
    ]
  ][
    ```rust
    fn add(self, rhs: Dual) -> Dual;
    fn sub(self, rhs: Dual) -> Dual;
    fn neg(self) -> Dual;
    fn mul(self, rhs: Dual) -> Dual;
    fn div(self, rhs: Dual) -> Dual;
    fn powf(self, k: f64) -> Dual;
    fn powi(self, k: i32) -> Dual;
    fn sin(self) -> Dual;
    fn cos(self) -> Dual;
    fn exp(self) -> Dual;
    fn ln (self) -> Dual;
    ```
    ...
  ]
]

#slide[
  = Forward mode: Реализация на Rust
  #align(center)[
    $
      f(x,y)=x dot y+sin(x) dot y
    $
    ```rust
    fn f(x:Dual, y:Dual) -> Dual  {
      x * y + x.sin() * y
    }
    ```
  ]
]

#slide[
  = Forward mode: Пример вызова

  // #set align(top)
  Прямой проход - выставляем 1 там где надо, а где не надо выставляем 0.

  #toolbox.side-by-side(columns: (1.5fr, 1fr))[

    ```rust
    fn f(x:Dual, y:Dual) -> Dual;

    let (x, y) = (1.2, 3.4);

    let (v, dx) = f(
      Dual { p: x, d: 1.0 },  // ∂f/∂x: берём по x
      Dual { p: y, d: 0.0 },  // держим y константой
    );

    let (v, dy) = f(
      Dual { p: x, d: 0.0 },  // держим x константой
      Dual { p: y, d: 1.0 },  // ∂f/∂y: берём по y
    );
    ```
  ][
    // #align(center)[

    //   #uncover(2)[
    //     $
    //       bold(x) = (1, 0, #text(gray)[$0)$] \
    //       bold(y) = (0, 1, #text(gray)[$0)$] \
    //       #text(gray)[$bold(z) = (0, 0, 0)$] \
    //     $
    //   ]
    // ]
  ]

]

// #slide[
//   = Forward mode: Векторизованный вариант


//   ```rust
//   struct Dual {
//     p: f64,  // Значение
//     d: f64,  // Производная
//   }
//   ```

//   #h(3.0em) #sym.arrow.b

//   ```rust
//   struct Dual<const N: usize> {
//     p: f64,                // Значение
//     derivatives: [f64; N], // Производная по направлению
//   }
//   ```
// ]

#slide[
  = Forward mode: Библиотеки

  На уровне исходного кода:

  - #text(weight: "bold")[num-dual] — обобщённые (hyper-)dual числа
  - #text(weight: "bold")[autodiff] — форвард-режим через dual числа (diff/grad)
  - #text(weight: "bold")[numdiff] — форвард-AD на dual числах плюс численная разность

]

#slide[
  = Reverse mode

  - #text(weight: "bold")[Обратный режим] (reverse mode, VJP — Vector–Jacobian Product)
]

#slide[
  = Reverse mode: Определение

  $f(x,y)=x dot y+sin(x) dot y$

  ```rust
  fn f(x: f64, y: f64) -> f64 {
      x * y + x.sin() * y
  }
  ```
  #h(3.0em) #sym.arrow.b
  ```rust
  /// Некоторая структура
  struct Tape { /* ... */ }

  ///  Прямой проход - расчёт значения функции и некого `Tape`
  fn f_augmented_primal(x: f64, y: f64) -> (f64, Tape);

  /// Обратный проход - получение дифференциала (dx, dy)
  fn f_reverse(x: f64, y: f64, seed_df: f64, tape: &Tape) -> (f64, f64);
  ```

]

#slide[
  = Reverse mode: Прямой проход

  ```rust
  fn f(x: f64, y: f64) -> f64 {
      x * y + x.sin() * y
  }
  ```

  #h(3.0em) #sym.arrow.b
  ```rust
  fn f_augmented_primal(x: f64, y: f64) -> (f64, Tape) {
      let s1 = x * y;
      let sinx = x.sin();
      let s2 = sinx * y;
      let f = s1 + s2;

      (f, Tape { sinx })
  }

  struct Tape { sinx: f64 }
  ```
]

#slide[
  = Reverse mode: Обратный проход

  #set align(top)
  #toolbox.side-by-side(columns: (1fr, 1.5fr), gutter: 1em)[

    #only(2)[
      #codly(highlights: (
        (line: 7, start: 13, end: 14),
      ))
    ]
    #only(3)[
      #codly(highlights: (
        (line: 7, start: 18, end: 19),
      ))
    ]
    #only(4)[
      #codly(highlights: (
        (line: 6, start: 14, end: 17),
      ))
    ]
    #only(5)[
      #codly(highlights: (
        (line: 6, start: 21, end: 21),
      ))
    ]
    #only(6)[
      #codly(highlights: (
        (line: 5, start: 16, end: 22),
      ))
    ]
    #only(7)[
      #codly(highlights: (
        (line: 4, start: 18, end: 18),
        (line: 4, start: 14, end: 14),
      ))
    ]
    ```rust
    struct Tape { sinx: f64 }

    fn f_augmented_primal(x: f64, y: f64) -> (f64, Tape) {
        let s1 = x * y;
        let sinx = x.sin();
        let s2 = sinx * y;
        let f = s1 + s2;

        (f, Tape { sinx })
    }
    ```
  ][

    #only(1)[
      ```rust
      fn f_reverse(x: f64, y: f64, seed_df: f64, tape: &Tape) -> (f64, f64) {
          let (mut bx, mut by) = (0.0, 0.0);
          /* ... */
      ```
    ]
    #only(2)[
      #codly(
        highlights: (
          (line: 4, tag: "  ∂f/∂s1 = 1"),
        ),
      )
      ```rust
      fn f_reverse(x: f64, y: f64, seed_df: f64, tape: &Tape) -> (f64, f64) {
          let (mut bx, mut by) = (0.0, 0.0);

          let bs1 = 1.0 * seed_df;
          /* ... */
      ```
    ]

    #only(3)[
      #codly(
        highlights: (
          (line: 5, tag: "  ∂f/∂s2 = 1"),
        ),
      )
      ```rust
      fn f_reverse(x: f64, y: f64, seed_df: f64, tape: &Tape) -> (f64, f64) {
          let (mut bx, mut by) = (0.0, 0.0);

          let bs1 = 1.0 * seed_df;
          let bs2 = 1.0 * seed_df;
          /* ... */
      ```
    ]
    #only(4)[
      #codly(
        highlights: (
          (line: 7, tag: "  ∂s2/∂sinx = y"),
        ),
      )

      ```rust
      fn f_reverse(x: f64, y: f64, seed_df: f64, tape: &Tape) -> (f64, f64) {
          let (mut bx, mut by) = (0.0, 0.0);

          let bs1 = 1.0 * seed_df;
          let bs2 = 1.0 * seed_df;

          let bsinx = y * bs2;
          /* ... */
      ```
    ]

    #only(5)[
      #codly(
        highlights: (
          (line: 8, tag: "  ∂s2/∂y = sinx"),
        ),
      )

      ```rust
      fn f_reverse(x: f64, y: f64, seed_df: f64, tape: &Tape) -> (f64, f64) {
          let (mut bx, mut by) = (0.0, 0.0);

          let bs1 = 1.0 * seed_df;
          let bs2 = 1.0 * seed_df;

          let bsinx = y * bs2;
          by += tape.sinx * bs2;
          /* ... */
      ```
    ]

    #only(6)[
      #codly(
        highlights: (
          (line: 9, tag: "  ∂sinx/∂x = cos(x)"),
        ),
      )

      ```rust
      fn f_reverse(x: f64, y: f64, seed_df: f64, tape: &Tape) -> (f64, f64) {
          let (mut bx, mut by) = (0.0, 0.0);

          let bs1 = 1.0 * seed_df;
          let bs2 = 1.0 * seed_df;

          let bsinx = y * bs2;
          by += tape.sinx * bs2;
          bx += x.cos() * bsinx;
          /* ... */
      ```
    ]

    #only(7)[
      #codly(
        highlights: (
          (line: 10, tag: "  ∂s1/∂x = y"),
          (line: 11, tag: "  ∂s1/∂y = x"),
        ),
      )
      ```rust
      fn f_reverse(x: f64, y: f64, seed_df: f64, tape: &Tape) -> (f64, f64) {
          let (mut bx, mut by) = (0.0, 0.0);

          let bs1 = 1.0 * seed_df;
          let bs2 = 1.0 * seed_df;

          let bsinx = y * bs2;
          by += tape.sinx * bs2;
          bx += x.cos() * bsinx;
          bx += y * bs1;
          by += x * bs1;

          (bx, by)
      }
      ```
    ]
  ]
]

#slide[
  = Reverse mode: Пример вызова

  ```rust
  let (x, y) = (1.2, 3.4);
  let (f_val, tape) = f_augmented_primal(x, y);
  let (dx, dy) = f_reverse(x, y, /* seed_df */ 1.0, &tape);
  ```
]

#slide[
  = Reverse mode: Библиотеки

  Динамический DAG в памяти:

  - #text(weight: "bold")[Burn] (burn-autodiff) — декоратор бэкенда с динамическим графом.
  - #text(weight: "bold")[Candle] (burn-autodiff) — динамический граф вычислений и backprop.
  - #text(weight: "bold")[autograd] (rust-autograd) — define-by-run, ленивое вычисление графа.
  - #text(weight: "bold")[reverse] — явный tape для backprop.
  - #text(weight: "bold")[dfdx] — DL-библиотека с backprop и автодифом

]


#slide[
  = Какой алгоритм выбрать?

  Смотрим на размерность входных и выходных данных
  #v(2.0em)
  #align(center)[
    #table(
      columns: (auto, auto),
      inset: 10pt,
      stroke: 0.5pt,
      align: center + horizon,
      [
        Подход
      ],
      [
        Стоимость на $∇f(bold(x))$
      ],

      // Автодифф
      [Автодифф: прямой режим],
      [
        $~ n times "cost"(f)$ \ (один проход на вход)
      ],

      [Автодифф: обратный режим],
      [
        $~ 2..4 times "cost"(f)$    \ (на скалярный выход)
      ],
    )
  ]
]

#new-section[
  // #image("img/enzyme_logo.svg", width: 15%)
  std::autodiff (Enzyme AD)
]

#slide[
  = Enzyme AD

  // #toolbox.side-by-side(columns: (1fr, 1fr, auto, auto, auto), gutter: 1em)[

  #grid(columns: (auto, auto, auto, auto, auto, auto, auto), gutter: .3em)[

    #grid(columns: (auto, auto), gutter: .3em)[
      #image("img/c_logo.svg", height: 10%)
      #image("img/cpp_logo.svg", height: 10%)
      #image("img/rust_logo.svg", height: 10%)
    ][
      #image("img/julia_logo.svg", height: 10%)
      #image("img/fortran_logo.svg", height: 10%)
      #image("img/swift_logo.svg", height: 10%)
    ]

  ][


    #block(width: 6em, height: 3em, {
      align(center + horizon)[#text(size: 3em)[#sym.arrow.r]]
      place(top + center, dx: 0pt, dy: 4pt)[#text(size: .9em, weight: "bold")[IR]]
    })
  ][

    #block({
      align(center + horizon)[#image("img/llvm_logo.png", height: 35%)]
      place(top + center, dx: 0pt, dy: 4pt)[#text(size: .9em, weight: "bold")[Stage 1]]
    })



  ][

    #block(width: 6em, height: 3em, {
      align(center + horizon)[#text(size: 3em)[#sym.arrow.r]]
      place(top + center, dx: 0pt, dy: 4pt)[#text(size: .9em, weight: "bold")[#image(
        "img/enzyme_logo.svg",
        width: 18%,
      )]]
    })
  ][

    #block({
      align(center + horizon)[#image("img/llvm_logo.png", height: 35%)]
      place(top + center, dx: 0pt, dy: 4pt)[#text(size: .9em, weight: "bold")[Stage 2]]
    })
  ][

    #text(size: 3em, weight: "bold")[#sym.arrow.r]

  ][
    #image("img/bin_file.svg", height: 15%)

  ]
  #align(
    center,
  )[Stage 1 #sym.approx O2 без #link("https://llvm.org/docs/Vectorizers.html#loop-vectorizer")[loop-vectorize], #link("https://llvm.org/docs/Vectorizers.html#slp-vectorizer")[slp-vectorize], #link("https://llvm.org/docs/Passes.html#loop-unroll-unroll-loops")[loop-unroll]]

]




#slide[
  = Установка
  ```sh
  # Выкачиваем репу
  git clone https://github.com/rust-lang/rust.git
  cd rust
  ```
  #uncover((2, 3))[

    ```sh
    # Выставляем необходимые флаги компиляции
    ./configure --enable-llvm-link-shared --enable-llvm-plugins --enable-llvm-enzyme --release-channel=nightly --enable-llvm-assertions --enable-clang --enable-lld --enable-option-checking --enable-ninja --disable-docs

    # Собирае компилятор
    ./x build --stage 1 library
    ```
  ]

  #uncover(3)[
    ```sh
    # Добавляем в тулчейн
    rustup toolchain link enzyme build/host/stage1
    rustup toolchain install nightly
    ```
    #link("https://enzyme.mit.edu/rust/installation.html")
  ]

]

#slide[
  = Компиляция
  Собирается программа так:
  ```sh
  RUSTFLAGS="-Z autodiff=Enable" cargo +enzyme build --release
  ```

  #uncover(2)[
    Без LTO пока не собирается:
    ```toml
    [profile.release]
    lto = "fat"
    ```

    Без `-Cembed-bitcode=yes` Enzyme не видит биткод зависимостей на стадии LTO и
    `rustc` падает с ошибкой `Can't find section .llvmbc`.
  ]
]

#slide[
  = Just команды

  #toolbox.side-by-side(columns: (2fr, 1fr), gutter: 1em)[

    Скачать и скомпилировать rustc+enzyme:
    ```sh
    just install
    ```
    Запустить какой-то из примеров из директории `crates`:
    ```sh
    just run <crate_name>
    ```
    Посмотреть декларацию дифф. функции (развернуть до AST):
    ```sh
    just expand <crate_name>
    ```
  ][
    #align(center)[
      #link("https://github.com/hexqnt/autodiff-rust")[
        #image("img/qr_repo.svg", width: 80%)
      ]
    ]
  ]
]

#slide[
  = std::autodiff

  #set align(top)
  #v(0.3em)
  #only(2)[
    #codly(highlights: (
      (line: 2, start: 20, end: 23),
    ))
  ]
  #only(3)[
    #codly(highlights: (
      (line: 2, start: 26, end: 41),
    ))
  ]
  #only(4)[
    #codly(highlights: (
      (line: 2, start: 44, end: 58),
    ))
  ]
  ```rust
  #[autodiff_forward(NAME, INPUT_ACTIVITIES, OUTPUT_ACTIVITY)]
  #[autodiff_reverse(NAME, INPUT_ACTIVITIES, OUTPUT_ACTIVITY)]
  ```
  #only((1, 2, 3, 4))[
    #v(1em)

    #only(2)[
      #codly(highlights: (
        (line: 2, start: 5, end: 6),
      ))
    ]
    #only(3)[
      #codly(highlights: (
        (line: 3, start: 5, end: 18),
      ))
    ]
    #only(4)[
      #codly(highlights: (
        (line: 4, start: 5, end: 10),
      ))
    ]
    ```rust
    #[autodiff_reverse(
        df,                   // имя сгенерированной функции
        Active, Active,       // активности аргументов (x, y)
        Active,               // активен скалярный результат f
    )]
    fn f(x: f64, y: f64) -> f64 {
      /* ... */
    }
    ```
  ]
  #only(5)[
    ```rust
    enum DiffActivity {
        None,           // Не участвует в дифференцировании
        Const,          // Учитывается как константа
        Active,         // (RM) Для скаляров f32, f64
        ActiveOnly,     // (RM) Считаем только градиент, без исходного значения
        Duplicated,     // (RM) для &T или *T, предоставляем свой теневой буфер
        DuplicatedOnly, // (RM) То же, но без пересчёта исходного значения (только градиент)
        Dual,           // (FM) Значение с одной "тенью"
        Dualv,          // (FM) Значение с векторизованной "тенью" (несколько направлений)
        DualOnly,       // (FM) Только "тень", без пересчёта значения
        DualvOnly,      // (FM) Только векторизованные "тени", без значения
    }
    ```
  ]

  #only(6)[
    #codly(
      highlights: (
        (line: 3, start: 5, end: 9),
        (line: 4, start: 5, end: 10),
        (line: 6, start: 5, end: 15),
        (line: 8, start: 5, end: 8),
      ),
    )
    ```rust
    enum DiffActivity {
        None,           // Не участвует в дифференцировании
        Const,          // Учитывается как константа
        Active,         // (RM) Для скаляров f32, f64
        ActiveOnly,     // (RM) Считаем только градиент, без исходного значения
        Duplicated,     // (RM) для &T или *T, предоставляем свой теневой буфер
        DuplicatedOnly, // (RM) То же, но без пересчёта исходного значения (только градиент)
        Dual,           // (FM) Значение с одной "тенью"
        Dualv,          // (FM) Значение с векторизованной "тенью" (несколько направлений)
        DualOnly,       // (FM) Только "тень", без пересчёта значения
        DualvOnly,      // (FM) Только векторизованные "тени", без значения
    }
    ```
  ]
]


#slide[
  = Пример обратного прохода
  #align(center)[
    #image("img/points.svg", height: 80%)
    Cуточная температура по СПб.
  ]
]

#slide[
  = Пример обратного прохода
  Модель: $t_i = a dot sin(2 pi (i+c)/p) + b$
  ```rust
  fn model(i: f32, a: f32, b: f32, p: f32, c: f32) -> f32 {
      let arg = 2.0 * PI * (i + c) / p;
      a * arg.sin() + b
  }
  ```
  $L(a,b,p,c) = sum_(i=0)^N ( t_i - T_i )^2$
  ```rust
  #[autodiff_reverse(d_sse, Const, Active, Active, Active, Active, Active)]
  fn sse_loss(temps: &[f32], a: f32, b: f32, p: f32, c: f32) -> f32 {
      temps.iter().enumerate().map(|(i, &temp)| {
              let r = model(i as f32, a, b, p, c) - temp;
              r * r
          }).sum()
  }
  ```
]

#slide[
  = Пример обратного прохода
  ```rust
  // отдаём градиент и loss библиотеке argmin
  impl<'a> Gradient for RegressionProblem<'a> {
      fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error> {
          /*...*/
          let (loss, da, db, dp, dc) = d_sse(self.temps, a, b, p, c, 1.0);
          /*...*/
      }
  }
  ```
]

#slide[
  = Пример обратного прохода
  #align(center)[
    #image("img/fit.svg", height: 80%)
    Питерская норма.
  ]
]


#slide[
  = AD артефакты. Пример №1

  $
     f(x) & =x \
    f'(x) & = 1
  $

  #v(2em)
  #toolbox.side-by-side(columns: (1fr, 1fr), gutter: 1em)[
    Правильная реализация:
    ```rust
    #[autodiff_forward(df, Dual, Dual)]
    fn f(x: f32) -> f32 {
        x
    }

    let (v, dx) = df(0.0, 1.0);
    // v=0 dx=1
    ```
  ][

    #uncover(2)[
      "Неправильная" реализация:
      ```rust
      #[autodiff_forward(df, Dual, Dual)]
      fn f(x: f32) -> f32 {
          if x == 0.0 { 0.0 } else { x }
      }

      let (v, dx) = df(0.0, 1.0);
      // v=0 dx=0
      ```
    ]

  ]

]

#slide[
  = AD артефакты. Пример №2

  #toolbox.side-by-side(columns: (1fr, 1fr), gutter: 1em)[
    ```rust
    #[autodiff_forward(d_round, Dual, Dual)]
    fn f_round(x: f64) -> f64 {
        x.round()
    }

    #[autodiff_forward(d_floor, Dual, Dual)]
    fn f_floor(x: f64) -> f64 {
        x.floor()
    }

    #[autodiff_forward(d_ceil, Dual, Dual)]
    fn f_ceil(x: f64) -> f64 {
        x.ceil()
    }
    ```
  ][
    ```rust
    // x = 1.51
    // round(x) = 2,  d/dx round = 0
    // floor(x) = 1,  d/dx floor = 0
    // ceil(x)  = 2,  d/dx ceil  = 0

    // x = 1
    // round(x) = 1,  d/dx round = 0
    // floor(x) = 1,  d/dx floor = 0
    // ceil(x)  = 1,  d/dx ceil  = 0
    ```
  ]
]

#new-section[Положение Enzyme в пайплайне rustc]

#slide[
  = Положение Enzyme в пайплайне rustc. AST

  #toolbox.side-by-side(columns: (1fr, 0.7fr), gutter: 1em)[
    ```rust
    #[autodiff_reverse(df, Active, Active)]
    fn f(x: f32) -> f32 { x * x }
    ```
    #h(3.0em) #sym.arrow.b
    ```rust
    #[rustc_autodiff] // внутренний атрибут rustc
    fn f(x: f32) -> f32 { x * x }
    ```
    #h(3.0em) #sym.arrow.b
    ```rust
    fn df(x: f32, df: f32) -> (f32, f32) {
        core::intrinsics::autodiff(f, df, (x, df))
    }
    ```
  ][
    #image("img/rustc_pipeline.svg", height: 100%)
  ]
]

#slide[
  = Конвейер rustc. HIR

  #toolbox.side-by-side(columns: (1fr, 0.7fr), gutter: 1em)[
    Атрибуты участвуют в дальнейшем как обычные «компиляторные» атрибуты: их видит HIR.

    Отдельно ведётся работа по корректному кросс-крейтовому кодированию/декодированию `rustc_autodiff` в метаданных, чтобы `#[autodiff]` работал через зависимости.
  ][
    #image("img/rustc_pipeline.svg", height: 100%)
  ]
]

#slide[
  = Конвейер rustc. THIR, MIR итд

  #toolbox.side-by-side(columns: (1fr, 0.7fr), gutter: 1em)[
    Задача остального пайплайна пронести `rustc_autodiff` дальше к LLVM без изменений

    Никаких специальных MIR-пассов под `autodiff` пока нет
  ][
    #image("img/rustc_pipeline.svg", height: 100%)
  ]
]

#slide[
  = Конвейер LLVM

  #toolbox.side-by-side(columns: (1fr, 0.7fr), gutter: 1em)[
    Где-то в середине конвейера оптимизации запускается Enzyme:
    - ищет вызовы `__enzyme_autodiff(...)` (reverse mode)
      и `__enzyme_fwddiff(...)` (forward mode)
    - на их месте порождает и подставляет градиентные функции
    - дальше всё это идёт через обычный LLVM-план оптимизаций

  ][
    #image("img/rustc_pipeline.svg", height: 100%)
  ]

]

#new-section[Вывод]

#slide[
  #show: focus
  Очень перспективно

  и

  очень сыро
]

#slide[
  = Ограничения Enzyme
  Что есть в Enzyme, но пока нет в Rust:
  - Кастомные производные для функций
  - Batching-режимы
  - Кастомные аллокаторы под AD.
  - Управление чекпоинтингом/рематериализацией (min-cut и пр.).
  - Параллелизм: OpenMP, MPI, и др. Rayon пока нет в Enzyme.
  - GPU-дифференцирование из Rust, поддержка CUDA/ROCm.
  Ограничения Enzyme:
  - Жёсткие требования к «статически анализируемому» коду
  - Инструментальность и сообщения об ошибках
  - GPU (работы ведутся)

]

#slide[
  = Спасибо за внимание

  #toolbox.side-by-side(columns: (1fr, 1fr), gutter: 1em)[
    #align(center)[
      #link("https://github.com/hexqnt/autodiff-rust")[
        #image("img/qr_voting.png", width: 50%)
      ]
      Оцените доклад
    ]
  ][
    #align(center)[
      #link("https://github.com/hexqnt/autodiff-rust")[
        #image("img/qr_repo.svg", width: 50%)
      ]
      Репа с кодом из доклада
    ]
  ]

]

#new-section[Приложение]

#slide[
  = Где применяется автодиф?
  - Обучение моделей и оптимизация:  #text(style: "italic")[градиенты, Гессианы, обратное распространение].
  - Научные расчёты: #text(style: "italic")[чувствительность ODE/PDE, оценка параметров].
  - Управление и робототехника: #text(style: "italic")[траекторная оптимизация, диф. MPC].
  - Графика и зрение: #text(style: "italic")[дифференцируемый рендеринг и инверсная графика].
  - Вероятностное программирование: #text(style: "italic")[вариационный вывод (ADVI)].
  - Финансы: #text(style: "italic")[грейки и риск в MC/PDE, калибровка моделей].
  - Инжиниринг и дизайн-оптимизация: #text(style: "italic")[CFD, атмосферные и инженерные задачи].
]

#slide[
  = Открытые math-LLM

  #table(
    columns: (auto, auto, auto),
    inset: 8pt,
    stroke: none,
    align: horizon,

    table.header([Модель], [Веса], [Год выпуска]),

    [Qwen2.5-Math],
    [
      #link("https://huggingface.co/Qwen/Qwen2.5-Math-1.5B")[1.5B],
      #link("https://huggingface.co/Qwen/Qwen2.5-Math-7B")[7B],
      #link("https://huggingface.co/Qwen/Qwen2.5-Math-72B")[72B]
    ],
    [2025],

    [Mathstral],
    [
      #link("https://huggingface.co/mistralai/Mathstral-7B-v0.1")[7B]
    ],
    [2024],

    [DeepSeekMath],
    [
      #link("https://huggingface.co/deepseek-ai/deepseek-math-7b-base")[7B]
    ],
    [2024],

    [Llemma],
    [
      #link("https://huggingface.co/EleutherAI/llemma_7b")[7B],
      #link("https://huggingface.co/EleutherAI/llemma_34b")[34B]
    ],
    [2023],

    [MetaMath],
    [
      #link("https://huggingface.co/meta-math/MetaMath-7B-V1.0")[7B],
      #link("https://huggingface.co/meta-math/MetaMath-70B-V1.0")[70B]
    ],
    [2023],

    [OpenMath],
    [
      #link("https://huggingface.co/nvidia/OpenMath-Mistral-7B-v0.1")[Mistral-7B],
      #link("https://huggingface.co/nvidia/OpenMath-Nemotron-7B")[Nemotron-7B]
    ],
    [2025],

    [InternLM2-Math-Plus],
    [
      #link("https://huggingface.co/internlm/internlm2-math-plus-1_8b")[1.8B],
      #link("https://huggingface.co/internlm/internlm2-math-plus-7b")[7B],
      #link("https://huggingface.co/internlm/internlm2-math-plus-20b")[20B],
      #link("https://huggingface.co/internlm/internlm2-math-plus-mixtral8x22b")[8×22B]
    ],
    [2025],

    [OpenThinker2],
    [
      #link("https://huggingface.co/open-thoughts/OpenThinker2-7B")[7B],
      #link("https://huggingface.co/open-thoughts/OpenThinker2-32B")[32B]
    ],
    [2025],
  )
]

#slide[
  = Reverse on Forward mode
  ```rust
  #[autodiff(ddf, Forward, Dual, Dual, Dual, Dual)]
  fn df2(x: &[f32;2], dx: &mut [f32;2], out: &mut [f32;1], dout: &mut [f32;1]) {
      df(x, dx, out, dout);
  }

  #[autodiff(df, Reverse, Duplicated, Duplicated)]
  fn f(x: &[f32;2], y: &mut [f32;1]) {
      y[0] = x[0] * x[0] + x[1] * x[0]
  }
  ```
]

#slide[
  = Кастомные функции Julia
  Пока отсутствует в Rust
  ```jl
  function f(y, x)
      y .= x.^2
      return sum(y)
  end

  # Кастомный forward-rule на Julia
  function forward(config::FwdConfig,
                   func::Const{typeof(f)}, # для какой функции
                   ::Type{<:Duplicated},   # аннотация активности результата
                   y::Duplicated, x::Duplicated)
      ret = func.val(y.val, x.val)
      y.dval .= 2 .* x.val .* x.dval
      return Duplicated(ret, sum(y.dval))
  end
  ```
]

#slide[
  = Символьный подход. Оч. сложные

  #toolbox.side-by-side(columns: (1fr, 1fr))[
    $
      "QR:"\ bold(A) = bold(Q) bold(R) \
      bold(Q)^top bold(Q) = bold(I) \
      op("diag")(bold(R)) > 0 \
      bold(Q)^top dif(bold(Q)) "кососимм." \
      dif(bold(A)) = dif(bold(Q)) bold(R) + bold(Q) dif(bold(R))
    $
  ][
    $
      "Cholesky:"\ bold(Σ) = bold(L) bold(L)^top \
      bold(Σ) ≻ 0 \
      bold(S) = (bold(L)^{-1}) dif(bold(Σ)) (bold(L)^{-1})^top \
      dif(bold(L)) = bold(L) Φ(bold(S))
    $
  ]
]

#slide[
  = Символьный подход (минусы)
  "Взрыв" выражений
  $
    f_n (x) = product_(i = 1)^n (x + sin x), quad
  $
  $
    f'_n (x) = sum_(k = 1)^n [(1 + cos x) dot product_(j = 1, j != k)^n (x + sin x)]
  $
  при полном разворачивании $~ n 2^(n-1)$ термов
]

#slide[
  = Якобиан, JVP и VJP
  // Есть функция
  $
    RR^n #sym.arrow RR^m
  $

  // Определение Якобиана

  $
    J_f(bold(x)) = mat(
      ( partial f_1 )/ (partial x_1), dots, (partial f_1) / (partial x_n);
      dots.v, dots.down, dots.v;
      (partial f_m) / (partial x_1), dots, (partial f_m) / (partial x_n)
    ) in RR^{m times n}
  $

  #toolbox.side-by-side(columns: (1fr, 1fr), gutter: 1em)[
    // Forward-mode: один прогон = один столбец
    $ text(underbrace(J_f(bold(x)) bold(e)_k, "JVP: столбец k при v = e_k")) $
  ][
    // Reverse-mode: один прогон = одна строка
    $ text(underbrace(bold(e)_j^"T" J_f(bold(x)), "VJP: строка j при y = e_j")) $
  ]
]

#slide[
  = Конечные разности
  Градиент по центральным разностям для $f(x_1, x_2) = sin(x_1) + x_2^2$ :
  ```rust
  use ndarray::array;
  use finitediff::ndarr;

  let f = |x: &ndarray::Array1<f64>| Ok(x[0].sin() + x[1].powi(2));
  let x = array![1.0, 2.0];
  let grad = ndarr::central_diff(&f)(&x)?;

  ```

  Якобиан по центральным разностям для $f(bold(x)) = vec(x_1 x_2, x_1 + x_2)$:
  ```rust
  use numdiff::central_difference as cd;

  fn f(x: &[f64]) -> Vec<f64> { vec![x[0] * x[1], x[0] + x[1]] }
  let j = cd::jacobian(f, &[1.0, 2.0]);
  ```

]

#slide[
  = Мат-осведомлённые градиенты Enzyme
  // Но если мы притворимся вот так вот, чтобы в Enzyme увидел в LLVM IR знакомый вызов. То Enzyme подставит
  ```rust
  #[no_mangle]
  #[inline(never)]
  pub unsafe extern "C" fn cblas_dgemm(/* прототип CBLAS из cblas.h */) {
    /* тело, например реализация `dgemm` из крейта faer  */
  }
  ```
  `cblas_dgemm` это реализация: $bold(C) = alpha bold(A) bold(B) + beta bold(C)$
]

#slide[
  = Где смотреть мат-осведомлённые градиенты Enzyme
  #link(
    "https://github.com/EnzymeAD/Enzyme/blob/main/enzyme/Enzyme/BlasDerivatives.td",
  )[`main/enzyme/Enzyme/BlasDerivatives.td`]

  Там декларации такого вида:

  ```td
  // C := alpha*op( A )*op( B ) + beta*C
  def gemm : CallBlasPattern<(Op $layout, $transa, $transb, $m, $n, $k, $alpha, $A, $lda, $B, $ldb, $beta, $C, $ldc), ...
  ```
]


#slide[
  = std::autodiff (reverse mode)
  ```rust
  #[autodiff_reverse(d_f_affine, Active, Const, Active)]
  fn f_affine(x: f64, bias: f64) -> f64;
  ```
  #h(2.0em) #sym.arrow.b

  ```rust
    pub fn d_f_affine(
        x: f64,
        bias: f64,
        df: f64,
    ) -> (f64, f64)

  // хотим df/dx, поэтому seed по выходу df = 1.0
  let (y, dx) = d_f_affine(x, bias, 1.0);
  ```
]

#slide[
  = std::autodiff (reverse mode)

  ```rust
  #[autodiff_reverse(d_f_dot, Duplicated, Duplicated, Active)]
  fn f_dot(x: &[f32], w: &[f32]) -> f32;
  ```
  #h(2.0em) #sym.arrow.b

  ```rust
  pub fn d_f_dot(
      x:  &[f32], dx: &mut [f32],
      w:  &[f32], dw: &mut [f32],
  ) -> f32  // возвращаем только значение


    // буферы для градиентов, должны совпадать по длине c x, w
    let mut dx = [0.0; 3];
    let mut dw = [0.0; 3];
    let y = d_f_dot(&x, &mut dx, &w, &mut dw);
  ```
]

#slide[
  = std::autodiff (forward mode)

  ```rust
  #[autodiff_forward(d_f_affine, Dual, Const, Dual)]
  fn f_affine(x: f64, bias: f64) -> f64;
  ```
  #h(2.0em) #sym.arrow.b
  ```rust
  fn d_f_affine(
      x: f64, dx: f64,
      bias: f64,
  ) -> (f64, f64)

  // хотим df/dx, значит задаём dx = 1.0
  let (y, dx) = d_f_affine(x, 1.0, bias);
  ```
]


#slide[
  = std::autodiff (forward mode)

  #toolbox.side-by-side(columns: (1fr, 1fr), gutter: 1em)[
    ```rust
    #[autodiff_forward(d_f_dot, Dual, Dual, Dual)]
    fn f_dot(x: &[f32], w: &[f32]) -> f32;
    ```
    #h(2.0em) #sym.arrow.b
    ```rust
    fn d_f_dot(
        x: &[f32], dx: &[f32],
        w: &[f32], dw: &[f32],
    ) -> (f32, f32)
    ```

  ][
    ```rust
    let dw = vec![0.0; n];
    let mut dx = vec![0.0; n];

    // Нужно много вызовов чтобы получить grad_x
    for i in 0..n {
        dx[i] = 1.0;
        let (y, df) = d_f_dot(x, &dx, w, &dw);
        grad_x[i] = df;
        dx[i] = 0.0;
    }
    ```
  ]
]

#slide[
  = Автодифференцирование
  #text(size: 20pt)[
    #scalebar()
    #align(center)[
      #sym.arrow.t \
      Автодифференцирование
    ]
  ]
  #align(center)[
    #text(size: 14pt)[
      #table(
        columns: (auto, auto, auto),
        inset: 10pt,
        stroke: 0.5pt,
        align: center + horizon,
        [
          Подход
        ],
        [
          Точность
        ],
        [
          Стоимость на $∇f(bold(x))$
        ],

        // Символьный / ручной
        [Символьный (аналитический)],
        [
          Машинная точность (или лучше)
        ],
        [
          Оценка выведенной формулы
        ],

        // Конечные разности
        [Конечные разности: прямая/обратная],
        [
          $O(h)$
        ],
        [
          $n$ вызовов $f()$ \ (по 1 на координату)
        ],

        [Конечные разности: центральная],
        [
          $O(h^2)$
        ],
        [
          $2n$ вызовов $f()$ \ (по 2 на координату)
        ],

        // Автодифф
        [Автодифф: прямой режим],
        [
          Машинная точность
        ],
        [
          $~ n times "cost"(f)$ \ (один проход на вход)
        ],

        [Автодифф: обратный режим],
        [
          Машинная точность
        ],
        [
          $~ 2..4 times "cost"(f)$    \ (на скалярный выход)
        ],
      )
    ]
  ]
]

#slide[
  = AD артефакты. Пример №3

  #toolbox.side-by-side(columns: (1fr, 1fr), gutter: 1em)[

    #align(center)[#image("img/ReLU.svg", width: 40%)]
    ```rust
    #[autodiff_forward(d_relu, Dual, Dual)]
    fn relu(x: f32) -> f32 {
        if x > 0.0 { x } else { 0.0 }
    }

    let (v, dx) = d_relu(0.0, 1.0);
    // в точке 0 для relu субградиент [0,1],
    // а получаем dx=0
    ```
  ][
    #align(center)[#image("img/abs.svg", width: 58%)]

    ```rust
    #[autodiff_forward(d_abs, Dual, Dual)]
    fn abs(x: f32) -> f32 {
        if x >= 0.0 { x } else { -x }
    }

    let (v, dx) = d_abs(0.0, 1.0);
    // в точке 0 для abs субградиент [-1,1],
    // а получаем dx=1
    ```
  ]
]
