"""
8x8 DCT / IDCT helpers, faithful to ``dct.c``.

The original code keeps DCT coefficients in two different layouts:
  * a flat 64-element list stored in *zig-zag* (scan) order (this is what the
    JPEG entropy data and the detection histograms use); and
  * a row-major (natural) 8x8 matrix where element [u][v] is the (u, v)
    frequency component.

``dcttomat``/``mattodct`` convert between the flat list (row-major, i = u*8+v)
and the matrix form.  ``idct`` computes ``D' * A * D`` and ``dct`` computes
``D * A * D'`` where ``D`` is the cosine basis below -- exactly the transforms
used by libjpeg's float DCT and by the F5 recompression attack in ``f5.c``.
"""

from __future__ import annotations

import numpy as np

# Cosine basis (identical constants to dct.c's _D[8][8]).
_D = np.array([
    [0.35355339059327, 0.35355339059327, 0.35355339059327, 0.35355339059327,
     0.35355339059327, 0.35355339059327, 0.35355339059327, 0.35355339059327],
    [0.49039264020162, 0.41573480615127, 0.27778511650980, 0.09754516100806,
     -0.09754516100806, -0.27778511650980, -0.41573480615127, -0.49039264020162],
    [0.46193976625564, 0.19134171618254, -0.19134171618254, -0.46193976625564,
     -0.46193976625564, -0.19134171618255, 0.19134171618255, 0.46193976625564],
    [0.41573480615127, -0.09754516100806, -0.49039264020162, -0.27778511650980,
     0.27778511650980, 0.49039264020162, 0.09754516100806, -0.41573480615127],
    [0.35355339059327, -0.35355339059327, -0.35355339059327, 0.35355339059327,
     0.35355339059327, -0.35355339059327, -0.35355339059327, 0.35355339059327],
    [0.27778511650980, -0.49039264020162, 0.09754516100806, 0.41573480615127,
     -0.41573480615127, -0.09754516100806, 0.49039264020162, -0.27778511650980],
    [0.19134171618254, -0.46193976625564, 0.46193976625564, -0.19134171618254,
     -0.19134171618255, 0.46193976625564, -0.46193976625564, 0.19134171618254],
    [0.09754516100806, -0.27778511650980, 0.41573480615127, -0.49039264020162,
     0.49039264020162, -0.41573480615127, 0.27778511650980, -0.09754516100806],
], dtype=np.float64)

_Dt = _D.T


def dcttomat(block: list):
    """Flat natural-order (row-major) 64-list -> 8x8 numpy matrix."""
    m = np.empty((8, 8), dtype=np.float64)
    for i in range(64):
        m[i >> 3][i & 7] = block[i]
    return m


def mattodct(mat: np.ndarray) -> list:
    """8x8 numpy matrix -> flat natural-order (row-major) 64-list."""
    return [int(round(mat[i >> 3][i & 7])) for i in range(64)]


def mat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b


def mat_transpose(m: np.ndarray) -> np.ndarray:
    return m.T


def mat_sadd(m: np.ndarray, val: float) -> np.ndarray:
    return m + val


def dct(inp: list) -> list:
    """Forward DCT: out = D * A * D', matching dct.c's dct()."""
    a = dcttomat(inp)
    tmp = _D @ a
    tmp = tmp @ _Dt
    return mattodct(tmp)


def idct(inp: list) -> list:
    """Inverse DCT: out = D' * A * D, matching dct.c's idct()."""
    a = dcttomat(inp)
    tmp = _Dt @ a
    tmp = tmp @ _D
    return mattodct(tmp)


def dequant_block(out: list, in_: list, table: list) -> None:
    """Dequantize ``in_`` with ``table`` in place into ``out`` (natural order).

    Mirrors ``dequant_block`` in ``jutil.c``: ``out[j] = in_[j] * table[j]`` for
    each of the 64 natural-order coefficients.  ``table`` is the quantization
    table in natural (row-major) order.
    """
    for j in range(64):
        out[j] = in_[j] * table[j]
