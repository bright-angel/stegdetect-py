import math

import numpy as np

from stegdetect_py.mathx import matrix_invert


def test_matrix_invert_2x2_exact():
    # [[4, 7], [2, 6]] has inverse [[0.6, -0.7], [-0.2, 0.4]]
    a = [[4.0, 7.0], [2.0, 6.0]]
    matrix_invert(a, 2)
    assert abs(a[0][0] - 0.6) < 1e-9
    assert abs(a[0][1] + 0.7) < 1e-9
    assert abs(a[1][0] + 0.2) < 1e-9
    assert abs(a[1][1] - 0.4) < 1e-9


def test_matrix_invert_matches_numpy():
    rng = np.random.RandomState(7)
    m = rng.rand(5, 5) + np.eye(5) * 2.0  # ensure well-conditioned
    a = [[float(m[i][j]) for j in range(5)] for i in range(5)]
    matrix_invert(a, 5)
    inv = np.array(a)
    expected = np.linalg.inv(m)
    assert np.allclose(inv, expected, atol=1e-9)


def test_matrix_invert_singular_raises():
    a = [[1.0, 2.0], [2.0, 4.0]]  # rank deficient
    try:
        matrix_invert(a, 2)
    except ValueError:
        return
    assert False, "expected ValueError for singular matrix"
