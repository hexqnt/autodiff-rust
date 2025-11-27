## regression: пример нелинейной регрессии

Крейт подбирает параметры синусоидальной модели дневной температуры Петербурга. Используются `std::autodiff` для вычисления градиента (reverse-mode) и `argmin` (steepest descent + line search) для оптимизации. На выходе создаются графики `points.svg` и `fit.svg`.

### Структура
- `src/data.rs` — массив температур.
- `src/plot.rs` — генерация графиков.

### Как запустить
- Из корня репозитория: `just run regression` (после `just install` для тулчейна).
- Либо напрямую: `cargo +enzyme run -p regression`.
