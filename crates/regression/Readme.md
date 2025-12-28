## regression: пример нелинейной регрессии

Небольшой пример подбора параметров синусоидальных моделей дневной температуры Петербурга. Градиенты считаются через `std::autodiff` (reverse-mode), оптимизация — `argmin` (steepest descent + line search). На выходе создаются графики `points.svg` и `fit.svg`.

Сначала идёт простой вариант с одной синусоидой и [SSE](https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%BD%D0%B0%D0%B8%D0%BC%D0%B5%D0%BD%D1%8C%D1%88%D0%B8%D1%85_%D0%BA%D0%B2%D0%B0%D0%B4%D1%80%D0%B0%D1%82%D0%BE%D0%B2), затем — более практичная модель с двумя гармониками и меняющейся дисперсией, где используется [MLE](https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%BC%D0%B0%D0%BA%D1%81%D0%B8%D0%BC%D0%B0%D0%BB%D1%8C%D0%BD%D0%BE%D0%B3%D0%BE_%D0%BF%D1%80%D0%B0%D0%B2%D0%B4%D0%BE%D0%BF%D0%BE%D0%B4%D0%BE%D0%B1%D0%B8%D1%8F).

### Простая модель (SSE)

Базовая модель:
$$ T_{model}(t) = a \cdot sin(2π \frac{t + c}{p} ) + b$$

где $t$ — номер дня, $a$ — амплитуда, $b$ — средний уровень, $p$ — период, $c$ — сдвиг по времени.

Лосс (SSE):
$$ L = \sum_i{ (T_{model}(i) - T_i)^2} $$

### Более сложная модель (две гармоники + MLE)

Среднее задаётся двумя фурье-гармониками и линейным трендом, а дисперсия меняется по гармоническому закону с линейным трендом.

Среднее (Фурье-гармоники + линейный тренд):
$$ T_{model}(t) = b + b_{trend} \cdot \tilde{t} + \sum_{k=1}^{2} \left( a_{ks} \cdot sin(2π \frac{k t}{p} ) + a_{kc} \cdot cos(2π \frac{k t}{p} ) \right)$$

Лог-дисперсия:
$$ \log \sigma^2(t) = v0 + v_{trend} \cdot \tilde{t} + \sum_{k=1}^{1} \left( v_{ks} \cdot sin(2π \frac{k t}{p} ) + v_{kc} \cdot cos(2π \frac{k t}{p} ) \right)$$
$$ \sigma^2(t) = exp(\log \sigma^2(t))$$

Логарифм для того, чтобы ограничить дисперсию положительными значениями.

где $a_{ks}, a_{kc}$ — коэффициенты Фурье для гармоник $k=1,2$, $p$ — общий период, $b_{trend}$ — линейный тренд среднего, $v0$ — базовый уровень лог-дисперсии, $v_{ks}, v_{kc}$ — коэффициенты дисперсии (здесь только $k=1$), $v_{trend}$ — линейный тренд дисперсии. Используется нормированное время $\tilde{t} = (t - t_0) / p$, где $t_0$ — середина ряда.

Лосс ([NLL](https://ru.wikipedia.org/wiki/%D0%9B%D0%BE%D0%B3%D0%B0%D1%80%D0%B8%D1%84%D0%BC%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B0%D1%8F_%D1%84%D1%83%D0%BD%D0%BA%D1%86%D0%B8%D1%8F_%D0%BF%D1%80%D0%B0%D0%B2%D0%B4%D0%BE%D0%BF%D0%BE%D0%B4%D0%BE%D0%B1%D0%B8%D1%8F) для нормального распределения, без константы):
$$ L = \sum_i \left( \log \sigma^2(i) + \frac{(T_{model}(i) - T_i)^2}{\sigma^2(i)} \right) $$

### Структура

- `src/data.rs` — массив температур.
- `src/simple.rs` — базовая синусоида + SSE.
- `src/mle.rs` — две гармоники + MLE (гетероскедастичность).
- `src/plot.rs` — генерация графиков.

### Как запустить

- Из корня репозитория: `just run regression` (после `just install` для тулчейна).
- Через `just` с выбором модели: `just run regression simple` или `just run regression mle`.
- Либо напрямую: `cargo +enzyme run -p regression`.
- По умолчанию запускается `simple` (можно явно: `cargo +enzyme run -p regression -- simple`).
- Для MLE-модели: `cargo +enzyme run -p regression -- mle`.
