from sympy import MatrixSymbol, derive_by_array, simplify

n = 4
A = MatrixSymbol("A", n, n)  # вход: матрица
x = MatrixSymbol("x", n, 1)  # вход: вектор

f = x.T * A * x

grad = derive_by_array(f, x)  # n×1
grad = simplify(grad)
print(f"{grad=}")
