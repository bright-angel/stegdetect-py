"""Faithful Python port of stegdetect's ``math.c``.

Only the routines actually used by the ported code are carried over:

* :func:`matrix_invert` -- in-place Gauss-Jordan inversion with implicit
  partial pivoting (Numerical-Recipes style), operating on a ``double **``
  which we represent as a list of row lists ``List[List[float]]``.  The
  algorithm is reproduced verbatim from the original C so that numerical
  results match the reference binary.
"""


def matrix_invert(a, n):
    """Invert the ``n x n`` matrix ``a`` *in place* (list of ``n`` rows).

    Mirrors ``matrix_invert(double **a, int n)`` from ``math.c``.  The input
    matrix is destroyed and replaced by its inverse.  Raises ``ValueError``
    on a (numerically) singular matrix, matching the original ``err(1, ...)``.
    """
    indxc = [0] * n
    indxr = [0] * n
    ipiv = [0] * n

    for i in range(n):
        big = 0.0
        irow = -1
        icol = -1
        for j in range(n):
            if ipiv[j] == 1:
                continue
            for k in range(n):
                if ipiv[k] == 0:
                    av = abs(a[j][k])
                    if av >= big:
                        big = av
                        irow = j
                        icol = k
                elif ipiv[k] > 1:
                    raise ValueError("matrix_invert: singular matrix")
        ipiv[icol] += 1
        if irow != icol:
            a[irow], a[icol] = a[icol], a[irow]
        indxr[i] = irow
        indxc[i] = icol
        if a[icol][icol] == 0.0:
            raise ValueError("matrix_invert: singular matrix")
        pivinv = 1.0 / a[icol][icol]
        a[icol][icol] = 1.0
        for l in range(n):
            a[icol][l] *= pivinv
        for ll in range(n):
            if ll != icol:
                tmp = a[ll][icol]
                a[ll][icol] = 0.0
                for l in range(n):
                    a[ll][l] -= a[icol][l] * tmp
    for l in range(n - 1, -1, -1):
        if indxr[l] != indxc[l]:
            for k in range(n):
                a[k][indxr[l]], a[k][indxc[l]] = a[k][indxc[l]], a[k][indxr[l]]


def print_matrix(str_, a, m, n):
    """Diagnostic pretty-printer mirroring ``print_matrix`` in ``math.c``."""
    print("%s:  (%d x %d)" % (str_, m, n))
    for i in range(m):
        line = "> "
        for j in range(n):
            line += " %4.2f" % a[i][j]
        print(line)
