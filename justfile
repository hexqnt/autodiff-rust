# Justfile

set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# Настройка
# Куда скачиваем репу для кастомной сборки rustc
repo := home_directory() / "rust_enzyme"
# Имя тулчейна которое будет зарегано в rustup
toolchain := "+enzyme"
target_dir := "target"
# Флаги включения прохода Enzyme и принудительно сохранения .llvmbc
autodiff_rustflags := "-Zautodiff=Enable -Cembed-bitcode=yes"

default: list

alias toolchains := toolchain

# Список рецептов
list:
	just --list

# Установка rustc+enzyme
install:
    #!/usr/bin/env bash
    if [ -d "{{repo}}/.git" ]; then
        git -C "{{repo}}" fetch --tags --prune
        git -C "{{repo}}" pull --ff-only
    else
        git clone https://github.com/rust-lang/rust.git "{{repo}}"
    fi
    cd "{{repo}}"
    ./configure --release-channel=nightly --enable-llvm-enzyme --enable-llvm-link-shared --enable-llvm-assertions --enable-ninja --enable-option-checking --disable-docs --set llvm.download-ci-llvm=false
    ./x build --stage 1 library
    rustup toolchain link enzyme build/host/stage1
    rustup toolchain install nightly # enables -Z unstable-options

toolchain:
    rustup toolchain list


uninstall:
    #!/usr/bin/env bash
    rustup toolchain uninstall enzyme
    rm -rf ~/rust_enzyme


# HIR
hir:
	cargo {{toolchain}} rustc -- -Z unpretty=hir > {{target_dir}}/hir.txt
	@echo "HIR → {{target_dir}}/hir.txt"

# THIR
thir:
	cargo {{toolchain}} rustc -- -Z unpretty=thir-tree > {{target_dir}}/thir.txt
	@echo "THIR → {{target_dir}}/thir.txt"

# MIR (все проходы → target/mir_dump/)
mir:
	RUSTFLAGS="{{autodiff_rustflags}}" cargo {{toolchain}} rustc -- -Z dump-mir=all
	@echo "MIR dumps → {{target_dir}}/mir_dump/"

# LLVM IR (текстовые .ll в target/)
ll:
	RUSTFLAGS="{{autodiff_rustflags}} --emit=llvm-ir" cargo {{toolchain}} build
	@echo "LLVM IR → {{target_dir}}/*/*.ll"

# Лог LLVM-проходов до/после (очень многословно)
llvm-log:
	RUSTFLAGS="-Cllvm-args=-print-before-all -Cllvm-args=-print-after-all" \
	cargo {{toolchain}} build -v 2> {{target_dir}}/llvm-pass.log
	@echo "LLVM pass log → {{target_dir}}/llvm-pass.log"

# Запуск примеров
run name:
	RUSTFLAGS="{{autodiff_rustflags}}" cargo +enzyme run -p {{name}} --release

expand name:
	RUSTFLAGS="{{autodiff_rustflags}}" cargo +enzyme expand -p {{name}} --release
# Очистка
clean:
	cargo clean
