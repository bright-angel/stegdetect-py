"""
Tests for the calibration feature-extraction port (extraction.py).

These verify the statistical primitives and the four transforms produce the
expected number of finite features, and that transform_lookup maps names to the
right functions.  Numerical fidelity to the C is checked by cross-checking
compute_stats against an independent implementation on known inputs.
"""

import io
import math
import os

import numpy as np
from PIL import Image

from stegdetect_py import common, extraction


def _make_jpg(tmp_path, seed=5, size=160, quality=95):
    rng = np.random.RandomState(seed)
    im = Image.fromarray(rng.randint(0, 256, (size, size, 3), dtype=np.uint8), "RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, subsampling=0)
    path = os.path.join(str(tmp_path), "img.jpg")
    with open(path, "wb") as fh:
        fh.write(buf.getvalue())
    return common.load_image(path)


def test_transform_lookup():
    assert extraction.transform_lookup("spline") is extraction.spline_transform
    assert extraction.transform_lookup("gradient") is extraction.gradient_transform
    assert extraction.transform_lookup("rough") is extraction.roughness_transform
    assert extraction.transform_lookup("diffsquare") is extraction.diffsquare_transform
    assert extraction.transform_lookup("nope") is None


def test_compute_stats_known():
    # Independent reference computation.
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    mean = sum(data) / len(data)
    var = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    std = math.sqrt(var)
    m, s, sk, ku = extraction.compute_stats(data, len(data))
    assert abs(m - mean) < 1e-9
    assert abs(s - std) < 1e-9
    # third/fourth central moments
    skew = sum((x - mean) ** 3 for x in data) / (len(data) - 1) / std ** 3
    kurt = sum((x - mean) ** 4 for x in data) / (len(data) - 1) / std ** 4 - 3
    assert abs(sk - skew) < 1e-9
    assert abs(ku - kurt) < 1e-9


def test_compute_stats_size_lt_2():
    assert extraction.compute_stats([7.0], 1) == (0.0, 0.0, 0.0, 0.0)


def test_histogram_centred_and_complete():
    data = [-5, 0, 3, 3, 10]
    hist, n = extraction.histogram(data, len(data))
    # every input value is accounted for exactly once
    assert sum(hist) == len(data)
    # zero-centred: replicate the centering to find where each value lands.
    mn_data = min(data)
    mx_data = max(data)
    nn = mx_data - mn_data
    while True:
        nn += 1
        nn = ((nn + 511) // 512) * 512
        if nn // 2 >= -mn_data and nn // 2 > mx_data:
            break
    mn = -nn // 2
    from collections import Counter
    counts = Counter(data)
    for v, c in counts.items():
        assert hist[v - mn] == float(c)
    assert n % 512 == 0


def test_distribution_returns_four(tmp_path):
    jpg = _make_jpg(tmp_path)
    dcts, bits = common.prepare_all(jpg)
    stats = extraction.distribution(0, dcts, bits)
    assert len(stats) == 4
    assert all(math.isfinite(x) for x in stats)


def test_spline_transform_length(tmp_path):
    jpg = _make_jpg(tmp_path)
    dcts, bits = common.prepare_all(jpg)
    out, npoints = extraction.spline_transform(dcts, bits)
    assert npoints == extraction.HOWMANY * 4
    assert len(out) == extraction.HOWMANY * 4
    assert all(math.isfinite(x) for x in out)


def test_roughness_transform_length(tmp_path):
    jpg = _make_jpg(tmp_path)
    dcts, bits = common.prepare_all(jpg)
    out, npoints = extraction.roughness_transform(dcts, bits)
    assert npoints == 4
    assert len(out) == 4
    assert all(math.isfinite(x) for x in out)


def test_diffsquare_transform_length(tmp_path):
    jpg = _make_jpg(tmp_path)
    dcts, bits = common.prepare_all(jpg)
    out, npoints = extraction.diffsquare_transform(dcts, bits)
    assert npoints == 64
    assert len(out) == 64
    assert all(math.isfinite(x) for x in out)


def test_gradient_transform_length(tmp_path):
    jpg = _make_jpg(tmp_path)
    out, npoints = extraction.gradient_transform(jpg)
    assert npoints == extraction.HOWMANY * 4
    assert len(out) == extraction.HOWMANY * 4
    assert all(math.isfinite(x) for x in out)


def test_dependency_returns_256_vectors(tmp_path):
    jpg = _make_jpg(tmp_path)
    dcts, bits = common.prepare_all(jpg)
    hmean, hstd, hskew, hkurt = extraction.dependency(1, 2, dcts, bits)
    for vec in (hmean, hstd, hskew, hkurt):
        assert len(vec) == 256
        assert all(math.isfinite(x) for x in vec)


def test_depprint_returns_string():
    hm = [0.0] * 256
    hd = [0.0] * 256
    hs = [0.0] * 256
    hk = [0.0] * 256
    for i in range(10, 50):
        hm[i] = float(i)
        hd[i] = float(256 - i)
    line = extraction.depprint(hm, hd, hs, hk)
    assert isinstance(line, str)
    parts = line.split()
    assert len(parts) == 16
