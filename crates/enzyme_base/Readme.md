## enzyme_base

Набор небольших примеров использования экспериментального автодиффа (`std::autodiff` / Enzyme) для скалярных и векторных функций, срезов, `ndarray` и обобщённых норм.

### Как запустить

```bash
RUSTFLAGS="-Zautodiff=Enable -Cembed-bitcode=yes" cargo +enzyme run -p enzyme_base --release
```

Флаг `-Cembed-bitcode=yes` обязателен: Enzyme выполняется на стадии LTO и требует
биткод всех зависимостей, иначе `rustc` не найдёт секцию `.llvmbc`.

Основные примеры находятся в модулях `scalar`, `array`, `generic`, `slice` и `ndarray`.
