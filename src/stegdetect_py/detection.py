"""
Detection engine, faithful to ``stegdetect.c``.

This module reproduces the chi-square tests that identify JPEG steganography:
  * JSteg / OutGuess  -- pair (k, -k) / (2i, 2i+1) statistics via unify_*
  * JPHide           -- the ``ltab``-ordered traversal with positional checks
  * F5               -- a 4-bin beta/KL histogram (fast path = comment string)

The public entry points used by the CLI are :func:`histogram_chi_jsteg`,
:func:`histogram_chi_outguess`, :func:`histogram_chi_jphide` and
:func:`histogram_f5`, which take a flat coefficient list + length and return a
detection score (0 / negative = no detection).

Performance note: the JSteg detector evaluates chi2 over every prefix [0, i) of
the coefficient stream.  In C that is cheap; in Python we use a cumulative
histogram so each sweep is O(n) instead of O(n^2) -- the *computed values are
identical* to rebuilding the histogram from scratch each time.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .chi2 import chi2cdf

# Sensitivity scaling (set by the -s CLI flag; also mutated internally by the
# JSteg detector, exactly as in the C sources).
SCALE = 1.0

# ---- histogram construction -------------------------------------------------

# Module-level cache of the *last* histogram built (mirrors the C global
# DCThist that the JPHide secondary checks jphide_zero_one / jphide_empty_pair
# read as a side effect of the most recent buildDCThist call).
_last_hist = [0.0] * 257


def buildDCThist(data: List[int], a: int, b: int) -> List[float]:
    """Build a 257-bin histogram of coefficient values (index = value + 128).

    Mirrors ``buildDCThist``.  Coefficients outside [-128, 127] (typically DC
    outliers) are ignored.  The original keeps an incremental optimization; since
    the result is order independent we simply rebuild over ``[a, b)``.  The result
    is also cached in the module global ``_last_hist`` for the JPHide helpers.
    """
    global _last_hist
    hist = [0.0] * 257
    n = len(data)
    end = min(b, n)
    for i in range(max(0, a), end):
        off = data[i]
        if off < -128 or off > 127:
            continue
        hist[off + 128] += 1
    _last_hist = hist
    return hist


def _binidx(data: List[int]):
    """Precompute (bin_index_array, valid_mask) for fast cumulative counting.

    Valid coefficients (in [-128,127]) map to bin index value+128 (0..255);
    invalid ones are masked out so they never contribute to any bin (matching the
    C behaviour where buildDCThist skips them).
    """
    arr = np.asarray(data, dtype=np.int64)
    valid = (arr >= -128) & (arr <= 127)
    binidx = np.where(valid, arr + 128, 0).astype(np.int32)
    return binidx, valid


def _chi2_over_prefix(binidx, valid, percent, bits, unify):
    """Yield (i, chi2) for i = percent, 2*percent, ... <= bits over [0, i).

    Maintains a single cumulative histogram, so the whole sweep is O(n) while
    producing exactly the same per-(i) histogram as ``buildDCThist(data,0,i)``.
    """
    run = np.zeros(256, dtype=np.float64)
    prev = 0
    i = float(percent)  # matches the C 'float i' incremented by `period`
    while int(i) <= bits:
        ii = int(i)
        run += np.bincount(
            binidx[prev:ii], minlength=256,
            weights=valid[prev:ii].astype(np.float64))
        prev = ii
        hist = run.tolist()
        theo = [0.0] * 128
        obs = [0.0] * 128
        discard = [0.0]
        size = unify(hist, theo, obs, discard)
        yield ii, chi2(theo, obs, size, discard[0])
        i += percent


# ---- unify: theoretical / observed histograms ------------------------------

def unify_false_jsteg(hist, theo, obs, discard):
    size = 0
    for i in range(128):
        if i == 64 or i == 65 or i == 0:
            continue
        theo[size] = (hist[2 * i - 1] + hist[2 * i]) / 2.0
        obs[size] = hist[2 * i]
        size += 1
    return size


def unify_false_outguess(hist, theo, obs, discard):
    size = 0
    for i in range(128):
        if i == 64 or i == 65:
            continue
        idx = 2 * i - 1
        one = hist[idx] if idx >= 0 else 0.0
        two = hist[2 * i]
        theo[size] = (one + two) / 2.0
        obs[size] = two
        size += 1
    return size


def unify_false_jphide(hist, theo, obs, discard):
    size = 0
    for i in range(128):
        if i == 64:
            continue
        if i < 64:
            theo[size] = (hist[2 * i] + hist[2 * i + 1]) / 2.0
            obs[size] = hist[2 * i + 1]
        else:
            theo[size] = (hist[2 * i - 1] + hist[2 * i]) / 2.0
            obs[size] = hist[2 * i]
        size += 1
    return size


def unify_normal(hist, theo, obs, discard):
    size = 0
    for i in range(128):
        if i == 64:
            continue
        theo[size] = (hist[2 * i] + hist[2 * i + 1]) / 2.0
        obs[size] = hist[2 * i + 1]
        size += 1
    return size


def unify_outguess(hist, theo, obs, pdiscard):
    size = 0
    total = 0.0
    for i in range(256):
        if i == 64 or i == 65:
            continue
        total += hist[i]
    discard = 0.0
    for i in range(128):
        if i == 64:
            continue
        one = hist[2 * i]
        two = hist[2 * i + 1]
        if one > two:
            f, fbar = one, two
        else:
            f, fbar = two, one
        # Skip pairs where OutGuess could not plausibly have produced them.
        if (fbar > f / 4) and ((f - f / 3) - (fbar + f / 3) > 0):
            discard += one + two
            continue
        theo[size] = (one + two) / 2.0
        obs[size] = two
        size += 1
    pdiscard[0] = discard / total if total else 0.0
    return size


def unify_jphide(hist, theo, obs, discard):
    size = 0
    for i in range(256):
        if 127 <= i <= 129:
            continue
        if i < 128 and not (i & 1):
            continue
        elif i >= 128 and (i & 1):
            continue
        theo[size] = (hist[i] + hist[i + 1]) / 2.0
        obs[size] = hist[i]
        size += 1
    return size


# ---- chi-square combination ------------------------------------------------

def chi2(DCTtheo: List[float], DCTobs: List[float], size: int, discard: float) -> float:
    ymt = ytt = 0.0
    sumchi = 0.0
    dgf = 0
    for i in range(size):
        ymt += DCTobs[i]
        ytt += DCTtheo[i]
        if ytt >= 5:
            chi = ymt - ytt
            chi = chi * chi
            chi /= ytt
            sumchi += chi
            dgf += 1
            ymt = ytt = 0.0
    f = 1 - chi2cdf(sumchi, dgf - 1)
    return f * (1 - discard)


def chi2test(data: List[int], bits: int,
             unify, a: int, b: int) -> float:
    if a < 0:
        a = 0
    if b > bits:
        b = bits
    if a >= b:
        return -1.0
    hist = buildDCThist(data, a, b)
    theo = [0.0] * 128
    obs = [0.0] * 128
    discard = [0.0]
    size = unify(hist, theo, obs, discard)
    return chi2(theo, obs, size, discard[0])


# ---- JSteg detector --------------------------------------------------------

def histogram_chi_jsteg(data: List[int], bits: int) -> float:
    global SCALE
    if bits == 0:
        return -1.0
    binidx, valid = _binidx(data)
    end = bits // 100
    if end < 4000:
        end = 4000

    _max, _min, _good = end, 200, end + 1
    _iteration = 0
    percent = end
    _sum = 0.0
    while _iteration < 6:
        _sum = 0.0
        for i, f in _chi2_over_prefix(binidx, valid, percent, bits, unify_false_jsteg):
            if f == 0:
                break
            if f > 0.4:
                _sum += f * percent
        # BINSEARCH_NEXT(400)
        if _iteration == 0:
            if _sum >= 400:
                break
            _good = percent
            percent = _min
        else:
            if _sum < 400:
                _good = percent
                if _good == _min:
                    break
                _max = percent
                percent = (_max - _min) / 2 + _min
            else:
                _min = percent
                percent = (_max - _min) / 2 + _min
        _iteration += 1

    # BINSEARCH_IFABORT(end)
    percent = _good
    if _good > end:
        return -1.0

    where = count = 0
    aftercount = maxv = 0.0
    scale = 0.95
    _sum = 0.0
    for i, f in _chi2_over_prefix(binidx, valid, percent, bits, unify_normal):
        if f == 0:
            break
        if f > 0.4:
            _sum += f * percent
            count += 1
        if f >= (maxv * scale):
            if f > maxv:
                maxv = f
                fs = (maxv - 0.4) / 0.6
                if fs > 0:
                    scale = 1 - (0.15 * fs + (1 - fs) * 0.05)
            aftercount = -3
            where = i
        elif f > 0.05 * maxv:
            if aftercount >= 0:
                aftercount += f * percent
            else:
                aftercount += 1
                where = i

    length = jsteg_size_safe(data, bits)
    minlen = where // 8
    maxlen = (where + percent) // 8
    if aftercount > 0:
        _sum -= aftercount
    if _sum < 0 or count < 3:
        _sum = 0
    if length < minlen // 2 or length > maxlen * 2:
        _sum = 0
    SCALE = scale
    return scale * _sum / (2 * percent)


def jsteg_size_safe(data: List[int], bits: int) -> int:
    """Wrapper that tolerates short inputs (mirrors ``jsteg_size``)."""
    from .common import jsteg_size
    return jsteg_size(data, bits)


# ---- OutGuess detector -----------------------------------------------------

NORM_OUTGUESS = [
    0.5, 0.7071067811865475244, 1, 1.2247448713915890491,
    1.4142135623730950488, 1.581138830084189666, 1.73205080756887729353,
    1.87082869338697069279, 2, 2.1213203435596425732, 2.23606797749978969641,
    2.34520787991171477728, 2.4494897427831780982, 2.54950975679639241501,
    2.6457513110645905905, 2.73861278752583056728, 2.8284271247461900976,
    2.91547594742265023544, 3, 3.08220700148448822513,
    3.162277660168379332,
]


def histogram_chi_outguess(data: List[int], bits: int) -> float:
    global SCALE
    _max, _min, _good = 10, 0.1, 10 + 1
    _iteration = 0
    percent = 10
    _sum = 0.0
    while _iteration < 9:
        range_ = int(percent * bits / 100)
        _sum = 0.0
        for i in range(101):
            off = i * bits // 100
            f = chi2test(data, bits, unify_false_outguess, off - range_, off + range_)
            _sum += f
        # BINSEARCH_NEXT(0.6)
        if _iteration == 0:
            if _sum >= 0.6:
                break
            _good = percent
            percent = _min
        else:
            if _sum < 0.6:
                _good = percent
                if _good == _min:
                    break
                _max = percent
                percent = (_max - _min) / 2 + _min
            else:
                _min = percent
                percent = (_max - _min) / 2 + _min
        _iteration += 1

    # BINSEARCH_IFABORT(10)
    percent = _good
    if _good > 10:
        return 0.0

    range_ = int(percent * bits / 100)
    count = 0.0
    _sum = 0.0
    for i in range(101):
        off = i * bits // 100
        f = chi2test(data, bits, unify_outguess, off - range_, off + range_)
        if f > 0.25:
            _sum += f
        if f > 0.001:
            count += 1

    count /= percent

    off = int(percent * 2)
    if off >= len(NORM_OUTGUESS):
        off = len(NORM_OUTGUESS) - 1
    norm = _sum / NORM_OUTGUESS[off]

    if count < 15:
        _sum -= (15 - count) * 0.5
        if _sum < 0:
            _sum = 0
    return SCALE * _sum / 0.5


# ---- JPHide detector -------------------------------------------------------

def jphide_runlength(data: List[int], bits: int) -> int:
    rundct = [0] * 128
    runlen = [0] * 128
    mx = -1
    for i in range(bits):
        coeff = data[i]
        if coeff < -127 or coeff > 127:
            continue
        if -1 <= coeff <= 1:
            continue
        if coeff < 0:
            off = -coeff // 2
        else:
            off = coeff // 2 + 64
        if rundct[off] != coeff:
            rundct[off] = coeff
            runlen[off] = 1
        else:
            runlen[off] += 1
            if runlen[off] > mx:
                mx = runlen[off]
    return mx


def jphide_zero_one() -> int:
    hist = _last_hist
    one = hist[-1 + 128] + hist[1 + 128]
    zero = hist[0 + 128]
    total = one + zero
    if total <= 10:
        return 0
    if one > zero:
        res = (3 * zero - one)
    else:
        res = (3 * one - zero)
    negative = 1 if res < -1 else 0
    if total >= 15 and res <= -1:
        negative = 1
    return negative


def jphide_empty_pair() -> int:
    hist = _last_hist
    res = 0
    for i in range(256):
        if 127 <= i <= 129:
            continue
        if i < 128 and not (i & 1):
            continue
        elif i >= 128 and (i & 1):
            continue
        if (hist[i] + hist[i + 1]) >= 5 and (not hist[i] or not hist[i + 1]):
            res += 1
    if res > 3:
        return 1
    return 0


def histogram_chi_jphide(data: List[int], bits: int) -> float:
    from .common import JPHPOS
    if JPHPOS[0] < 500:
        return 0.0

    f = chi2test(data, bits, unify_jphide, 0, JPHPOS[0])
    if f < 0.9:
        return 0.0
    if jphide_runlength(data, JPHPOS[0]) > 16:
        return 0.0
    if jphide_zero_one():
        return 0.0
    if jphide_empty_pair():
        return 0.0

    f2 = chi2test(data, bits, unify_false_jphide, 0, JPHPOS[0])
    if f2 * 0.95 > f:
        return 0.0

    f = chi2test(data, bits, unify_jphide, JPHPOS[0] // 2, JPHPOS[0])
    if f < 0.9:
        return 0.0
    f2 = chi2test(data, bits, unify_false_jphide, JPHPOS[0] // 2, JPHPOS[0])
    if f2 * 0.95 > f:
        return 0.0

    f = chi2test(data, bits, unify_jphide, 0, JPHPOS[0] // 2)
    f2 = chi2test(data, bits, unify_false_jphide, 0, JPHPOS[0] // 2)
    if f2 * 0.95 > f:
        return 0.0

    range_ = JPHPOS[0] // 12
    for i in range(11, 0, -1):
        if range_ < 250:
            range_ = JPHPOS[0] // i
    if range_ < 250:
        range_ = 250

    negative = highpeak = 0
    falsev = _sum = 0.0
    i = range_
    while i <= bits and (not negative or i < 4 * JPHPOS[0]):
        f = chi2test(data, bits, unify_jphide, 0, i)
        f2 = chi2test(data, bits, unify_false_jphide, 0, i)
        if i <= JPHPOS[0] and jphide_zero_one():
            negative += 1
        if i <= JPHPOS[0] and jphide_empty_pair():
            negative += 1
        if i <= JPHPOS[1] and f2 >= 0.95:
            falsev += f2 * range_
            if falsev * 1.10 >= JPHPOS[1]:
                negative += 1
        if f >= 0.95:
            highpeak = 1
        if i > JPHPOS[0] and not highpeak:
            negative += 1
        if highpeak and f < 0.90 and _sum < JPHPOS[0]:
            negative += 1
        if i <= JPHPOS[1] and f2 * 0.99 > f:
            negative += 1
        if f >= 0.9:
            _sum += f * range_
        elif f < 0.2:
            break
        i += range_

    _sum /= 1000

    if negative:
        return 0.0

    _sum *= 1100.0 / JPHPOS[0]
    return SCALE * _sum


# ---- F5 detector (4-bin beta/KL histogram) --------------------------------

def histogram_f5(hist: List[float], size: int) -> float:
    if size < 63:
        return 0.0
    n = 4
    DCTtheo = [0.0] * 8
    DCTobs = [0.0] * 8
    for i in range(n):
        DCTobs[i] = hist[i * 8]
        mean = hist[i * 8 + 1]
        if i != 0:
            mean = (mean + hist[i * 8 - 1]) / 2.0
        DCTtheo[i] = mean
    return chi2(DCTtheo, DCTobs, n, 0.0)
