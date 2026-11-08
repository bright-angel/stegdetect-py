"""
Chi-square cumulative distribution function.

The original stegdetect ships a precomputed table ``chitable[256][1024]``
(``chi2cdf.c`` + ``chi2cdf.h``) indexed by ``(degrees-of-freedom, chi)`` with
``chi`` sampled at step 0.5 up to ~512.  We replace that table with a direct
evaluation of the regularized lower incomplete gamma function:

    P(chi^2_{k} <= chi) = gamma(k/2, chi/2) / Gamma(k/2) = gammp(k/2, chi/2)

This is identical in semantics to the original ``chi2cdf(chi, dgf)`` which
returns the lower-tail probability of a chi-square variable with ``dgf`` degrees
of freedom.  The detection engine then uses ``f = 1 - chi2cdf(sumchi, dgf-1)``;
a *high* ``f`` (near 1) indicates the statistics expected under steganography,
i.e. a positive detection.
"""

from __future__ import annotations

import math


def _gser(a: float, x: float) -> float:
    """Series representation of the regularized lower incomplete gamma P(a, x).

    Valid for x < a + 1.  Uses the standard Numerical-Recipes recurrence.
    """
    gln = math.lgamma(a)
    if x <= 0:
        return 0.0
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(2000):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * 1e-16:
            break
    return total * math.exp(-x + a * math.log(x) - gln)


def _gcf(a: float, x: float) -> float:
    """Continued fraction for the *upper* regularized incomplete gamma Q(a, x).

    Valid for x >= a + 1.  gammp = 1 - _gcf.
    """
    gln = math.lgamma(a)
    FPMIN = 1e-300
    b = x + 1.0 - a
    c = 1.0 / FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, 2000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < FPMIN:
            d = FPMIN
        c = b + an / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def gammp(a: float, x: float) -> float:
    """Regularized lower incomplete gamma function P(a, x)."""
    if a <= 0:
        raise ValueError("a must be > 0")
    if x <= 0:
        return 0.0
    if x < a + 1.0:
        return _gser(a, x)
    return 1.0 - _gcf(a, x)


def chi2cdf(chi: float, dgf: int) -> float:
    """Lower-tail probability of a chi-square variable with ``dgf`` degrees of
    freedom evaluated at ``chi``.

    Faithful to the original ``chi2cdf``:
      * ``dgf <= 0`` -> 1.0  (the calling code uses this to yield f = 0)
      * ``chi <= 0`` -> 0.0
    """
    if dgf <= 0:
        return 1.0
    if chi <= 0:
        return 0.0
    # P(chi^2_{dgf} <= chi) = gammp(dgf/2, chi/2)
    return gammp(dgf / 2.0, chi / 2.0)
