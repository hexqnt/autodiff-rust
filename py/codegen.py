from io import StringIO

from sympy import Eq, MatrixSymbol, derive_by_array, latex, simplify
from sympy.utilities.codegen import RustCodeGen, make_routine

n = 4
A = MatrixSymbol("A", n, n)  # вход: матрица
x = MatrixSymbol("x", n, 1)  # вход: вектор
g = MatrixSymbol("g", n, 1)  # выход: градиент-вектор

f = x.T * A * x
grad = derive_by_array(f, x)  # n×1
grad = simplify(grad)

print(grad)
print(latex(grad))  # LaTeX-формула градиента
print(latex(Eq(g, grad)))  # g = ∂f/∂x

r2 = make_routine(
    "quad_and_grad",
    (f, Eq(g, grad)),
    language="rust",
    argument_sequence=[A, x, g],
)

gen = RustCodeGen()
buf = StringIO()
gen.dump_rs([r2], buf, prefix="linalg_kernels", header=True, empty=False)
print(buf.getvalue())
