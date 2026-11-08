"""
Calibration / feature-statistics extraction -- faithful Python port of extraction.c.

In the original stegdetect this file is NOT the hidden-message extractor; it
implements the *calibration* feature extraction used by the detection engine
(Fridrich-style statistics over DCT coefficients).  It exposes a small family of
``transform`` functions that turn a flat DCT-coefficient buffer into a vector of
statistical features:

    * ``spline_transform``     -- spline-interpolation error distribution
    * ``gradient_transform``   -- gradient-x of the DCT plane, then spline stats
    * ``roughness_transform``  -- frequency-weighted roughness statistics
    * ``diffsquare_transform`` -- horizontal/vertical difference-square statistics

plus the supporting primitives ``compute_stats``, ``histogram``, ``distribution``,
``dependency``, and ``transform_lookup`` (name -> transform function).

The coordinate convention matches the C: the input ``data`` is a flat ``short``
array of ``bits`` coefficients where block ``j``'s coefficient ``slot`` lives at
``data[j * DCTSIZE2 + slot]`` -- exactly what :func:`stegdetect_py.common.prepare_all`
produces.  ``gradient_transform`` is special: like the C it ignores the flat
buffer and recomputes from the source image (here the decoded ``JPEG``), via
:func:`stegdetect_py.common.prepare_all_gradx`.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from . import common

DCTSIZE = common.DCTSIZE        # 8
DCTSIZE2 = common.DCTSIZE2      # 64

SPLINEN = 10
HOWMANY = 18


# ---------------------------------------------------------------------------
# transform registry
# ---------------------------------------------------------------------------
def transform_lookup(name: str):
    """Return the transform function registered under ``name`` (or ``None``)."""
    return _CD_TRANSFORMS.get(name)


# ---------------------------------------------------------------------------
# statistics primitives
# ---------------------------------------------------------------------------
def compute_stats(hist: List[float], size: int) -> Tuple[float, float, float, float]:
    """mean, std, skew, kurtosis of ``hist``.  Mirrors compute_stats()."""
    mean = 0.0
    for i in range(size):
        mean += hist[i]
    mean /= size

    if size < 2:
        return 0.0, 0.0, 0.0, 0.0

    std = skew = kurt = 0.0
    for i in range(size):
        tmp = hist[i] - mean
        tmp2 = tmp * tmp
        std += tmp2
        skew += tmp2 * tmp
        kurt += tmp2 * tmp2
    std = math.sqrt(std / (size - 1))

    if std != 0:
        skew = skew / (size - 1) / (std * std * std)
        kurt = kurt / (size - 1) / (std * std * std * std) - 3
    else:
        skew = 0.0
        kurt = 0.0

    return mean, std, skew, kurt


def spline_setup(x, y, n, y2, u) -> None:
    """Natural-cubic-spline second-derivative computation (in-place)."""
    y2[0] = -0.5
    u[0] = (3.0 / (x[1] - x[0]) * ((y[1] - y[0]) / (x[1] - x[0])))

    for i in range(1, n - 1):
        sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])
        p = sig * y2[i - 1] + 2.0
        y2[i] = (sig - 1.0) / p
        u[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1])
        u[i] = (6.0 * u[i] / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p

    un = (3.0 / (x[n - 1] - x[n - 2]) * (y[n - 2] - y[n - 1]) / (x[n - 1] - x[n - 2]))
    y2[n - 1] = (un - 0.5 * u[n - 2]) / (0.5 * y2[n - 2] + 1.0)

    for i in range(n - 2, -1, -1):
        y2[i] = y2[i] * y2[i + 1] + u[i]


def spline_inter(x, y, y2, n, lo, hi) -> float:
    """Spline interpolation of the gap between samples ``lo`` and ``hi``."""
    h = 2.0
    a = 0.5
    b = 0.5
    return (a * y[lo] + b * y[hi]
            + ((a * a * a - a) * y2[lo] + (b * b * b - b) * y2[hi]) * (h * h) / 6.0)


def esterror2(h: List[float], ind: int, maxi: int) -> float:
    """Spline-interpolation error at histogram index ``ind`` (local window)."""
    s = ind - SPLINEN // 2
    if s < 0:
        s = 0
    e = ind + SPLINEN // 2
    if e > maxi:
        e = maxi

    n = 0
    x = [0.0] * SPLINEN
    y = [0.0] * SPLINEN
    y2 = [0.0] * SPLINEN
    u = [0.0] * SPLINEN
    lo = 0
    hi = 0
    for i in range(s, e + 1):
        if n >= SPLINEN:
            break
        if i == ind:
            lo = n - 1
            hi = n
            continue
        x[n] = i
        y[n] = h[i]
        n += 1

    if hi >= n:
        # In the C this is an err()/abort; treat as zero contribution.
        return 0.0

    spline_setup(x, y, n, y2, u)

    error = spline_inter(x, y, y2, n, lo, hi)
    error -= h[ind]

    return abs(error)


def esterror(h: List[float], ind: int) -> float:
    """Two-sided linear-estimate error at index ``ind``.

    The original C reads ``h[ind-2]``..``h[ind+2]`` unconditionally, which is an
    out-of-bounds access at the array edges (undefined behaviour in C, a crash in
    Python).  We clamp the neighbour indices so this debug-only routine never
    raises -- the values involved are only used for diagnostic printing.
    """
    n = len(h)
    lo2 = max(0, ind - 2)
    lo1 = max(0, ind - 1)
    hi1 = min(n - 1, ind + 1)
    hi2 = min(n - 1, ind + 2)
    est1 = (h[lo1] - h[lo2]) / 2 + h[lo1]
    est2 = (h[hi1] - h[hi2]) / 2 + h[hi1]

    error1 = h[ind] - est1
    error2 = h[ind] - est2

    return math.sqrt(error1 * error1 + error2 * error2)


def histogram(data: List[int], datsiz: int) -> Tuple[List[float], int]:
    """Build a rounded, zero-centred histogram of ``data``.  Returns (hist, n)."""
    mn = 32767
    mx = -32767
    for i in range(datsiz):
        if data[i] > mx:
            mx = data[i]
        if data[i] < mn:
            mn = data[i]

    n = mx - mn
    while True:
        n += 1
        n = ((n + 511) // 512) * 512
        if n // 2 >= -mn and n // 2 > mx:
            break

    mn = -n // 2

    hist = [0.0] * n
    for i in range(datsiz):
        off = data[i] - mn
        if off < 0 or off >= n:
            raise ValueError(f"Bad offset: {off}")
        hist[off] += 1

    return hist, n


def distribution(slot: int, data: List[int], bits: int) -> List[float]:
    """Four stats (mean/std/skew/kurt) of the spline-error distribution for one
    DCT ``slot`` across all blocks.  Mirrors ``distribution()``."""
    n = bits // DCTSIZE2
    ndata = [data[j * DCTSIZE2 + slot] for j in range(n)]

    simple, nsimple = histogram(ndata, n)

    herror = [0.0] * nsimple
    for i in range(2, nsimple - 2):
        if simple[i] != 0:
            herror[i] = esterror2(simple, i, nsimple - 1) / simple[i]
        else:
            herror[i] = 0.0

    mean, std, skew, kurt = compute_stats(herror, nsimple)
    return [mean, std, skew, kurt]


def depprint(hm, hd, hs, hk) -> str:
    """Dependency-print helper; returns the whitespace-separated 16-stat line."""
    hmean = [0.0] * 256
    hstd = [0.0] * 256
    hskew = [0.0] * 256
    hkurt = [0.0] * 256

    n = 0
    for i in range(256):
        hmean[n] = esterror(hm, i)
        hstd[n] = esterror(hd, i)
        hskew[n] = esterror(hs, i)
        hkurt[n] = esterror(hk, i)
        if hm[i] != 0 and hd[i] != 0:
            n += 1

    out = []
    for arr in (hmean, hstd, hskew, hkurt):
        mean, std, skew, kurt = compute_stats(arr, n)
        out.append(f"{mean} {std} {skew} {kurt} ")
    return "".join(out)


def dependency(one: int, two: int, data: List[int], bits: int):
    """Calibration dependency features between coefficient slots ``one``/``two``.

    Returns the four 256-element stat vectors (mean/std/skew/kurt) as a tuple.
    """
    hmean = [0.0] * 256
    hstd = [0.0] * 256
    hskew = [0.0] * 256
    hkurt = [0.0] * 256

    model = [[0] * 256 for _ in range(256)]
    simple: List[Optional[List[int]]] = [None] * 256
    nsimple = [0] * 256

    for i in range(0, bits, DCTSIZE2):
        dct = data[i + one]
        dctnext = data[i + two]
        dct += 128
        dctnext += 128
        if dct > 255 or dctnext > 255:
            continue
        model[dct][dctnext] += 1

    for i in range(256):
        n = sum(model[i])
        nsimple[i] = n
        arr = []
        for j in range(256):
            arr.extend([j] * model[i][j])
        simple[i] = arr

    for i in range(256):
        if nsimple[i] > 4:
            mean, std, skew, kurt = compute_stats(simple[i], nsimple[i])
            hmean[i] = mean
            hstd[i] = std
            hskew[i] = skew
            hkurt[i] = kurt

    return hmean, hstd, hskew, hkurt


# ---------------------------------------------------------------------------
# transform family
# ---------------------------------------------------------------------------
def spline_transform(dcts: List[int], bits: int) -> Tuple[List[float], int]:
    """``HOWMANY`` x 4 spline-error stats over the first DCT slots."""
    output: List[float] = []
    for i in range(HOWMANY):
        output.extend(distribution(i, dcts, bits))
    return output, HOWMANY * 4


def gradient_transform(jpg, bits: int = None) -> Tuple[List[float], int]:
    """Gradient-x of the DCT plane, then spline stats (18 x 4 features).

    Like the C original, this recomputes from the source image rather than the
    supplied flat buffer (which it ignores), via :func:`common.prepare_all_gradx`.
    """
    ndcts, nbits = common.prepare_all_gradx(jpg)
    return spline_transform(ndcts, nbits)


def roughness_transform(dcts: List[int], bits: int) -> Tuple[List[float], int]:
    """Frequency-weighted roughness statistics (4 features)."""
    n = bits // DCTSIZE2
    points = [0.0] * n
    for i in range(n):
        total = 0.0
        weight = 0.0
        off = i * DCTSIZE2
        for j in range(DCTSIZE2):
            u = j // 8
            v = j % 8
            val = dcts[off + j]
            total += (u * u + v * v) * val * val
            weight += (u * u + v * v)
        points[i] = math.sqrt(total / weight)

    mean, std, skew, kurt = compute_stats(points, n)
    output = [mean, std, skew, kurt]
    return output, 4


def diffsquare_transform(dcts: List[int], bits: int) -> Tuple[List[float], int]:
    """Horizontal then vertical difference-square statistics (64 features)."""
    n = bits // DCTSIZE2
    output: List[float] = []

    # horizontal differences: for each freq row k, across the 8 block columns
    for k in range(DCTSIZE):
        points = [0.0] * n
        for i in range(n):
            total = 0.0
            weight = 0.0
            off = i * DCTSIZE2
            for j in range(DCTSIZE - 1):
                val1 = dcts[off + k * DCTSIZE + j]
                val2 = dcts[off + k * DCTSIZE + j + 1]
                total += (val2 - val1) * (val2 - val1)
                weight += abs(val1)
            points[i] = math.sqrt(total) / (weight + 1)
        mean, std, skew, kurt = compute_stats(points, n)
        output.extend([mean, std, skew, kurt])

    # vertical differences: for each freq column k, across the 8 block rows
    for k in range(DCTSIZE):
        points = [0.0] * n
        for i in range(n):
            total = 0.0
            weight = 0.0
            off = i * DCTSIZE2
            for j in range(DCTSIZE - 1):
                val1 = dcts[off + k + DCTSIZE * j]
                val2 = dcts[off + k + DCTSIZE * (j + 1)]
                total += (val2 - val1) * (val2 - val1)
                weight += abs(val1)
            points[i] = math.sqrt(total) / (weight + 1)
        mean, std, skew, kurt = compute_stats(points, n)
        output.extend([mean, std, skew, kurt])

    return output, len(output)


_CD_TRANSFORMS: Dict[str, object] = {
    "spline": spline_transform,
    "gradient": gradient_transform,
    "rough": roughness_transform,
    "diffsquare": diffsquare_transform,
}
