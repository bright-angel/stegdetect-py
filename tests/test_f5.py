import io
import math

import numpy as np
from PIL import Image

from stegdetect_py import f5
from stegdetect_py.dct import idct
from stegdetect_py.f5 import (
    F5Image,
    Jeasy,
    betakl,
    detect_f5,
    f5_blur,
    f5_crop,
    f5_dobeta,
    f5_ekl,
    f5_hkl,
    f5_luminanceimage,
)
from stegdetect_py.jpeg import decode_coefficients


def _single_block_jeasy(block, needscale=0):
    """Build a 1x1-block Jeasy (component 0) with a uniform quant table of 1s."""
    return Jeasy(
        comp=1,
        height=[1],
        width=[1],
        blocks=[[list(block)]],
        table=[[1] * 64],
        needscale=needscale,
        scale=[1.0],
    )


def test_f5_hkl_counts_matching_coefficients():
    b0 = [0] * 64
    b1 = [0] * 64
    b2 = [0] * 64
    b0[17] = 0  # il=2, ik=1 -> pos 17
    b1[17] = 1
    b2[17] = 2
    je = Jeasy(comp=1, height=[3], width=[1],
               blocks=[[b0, b1, b2]], table=[[1] * 64],
               needscale=0, scale=[1.0])
    assert f5_hkl(je, 1, 2, 0) == 1
    assert f5_hkl(je, 1, 2, 1) == 1
    assert f5_hkl(je, 1, 2, 2) == 1
    assert f5_hkl(je, 1, 2, 5) == 0


def test_f5_luminanceimage_is_uniform_for_dc_only():
    # A DC-only block reconstructs to a uniform image (idct of DC term).
    dc = 16
    block = [dc] + [0] * 63
    je = _single_block_jeasy(block)
    img = F5Image()
    f5_luminanceimage(je, img)
    assert img.x == 8 and img.y == 8
    dq = list(block)  # table is all 1s, so dequant == block
    recon = idct(dq)
    expected = np.clip(
        np.array(recon, dtype=np.int32).reshape(8, 8) + 128, 0, 255
    ).astype(np.uint8)
    assert np.all(img.arr == expected)
    assert img.arr.min() == img.arr.max()


def test_f5_crop_reduces_dimensions():
    img = F5Image(x=16, y=16, depth=1)
    img.allocate()
    img.arr[:] = np.arange(256, dtype=np.uint8).reshape(16, 16)
    f5_crop(img)
    assert img.x == 8 and img.y == 8


def test_f5_blur_averages_and_stays_in_range():
    img = F5Image(x=10, y=10, depth=1)
    img.allocate()
    rng = np.random.RandomState(1)
    img.arr[:] = rng.randint(0, 256, (10, 10)).astype(np.uint8)
    before = img.arr.copy()
    f5_blur(img, 0.1)
    assert img.arr.min() >= 0 and img.arr.max() <= 255
    # Blur smooths: the interior should differ from (be closer to neighbours than) raw.
    assert not np.array_equal(before[1:9, 1:9], img.arr[1:9, 1:9])


def test_betakl_deterministic():
    # orig block has pos 17 == 0; est block has pos 17 == 1.
    orig_block = [0] * 64
    est_block = [0] * 64
    est_block[17] = 1
    orig = _single_block_jeasy(orig_block)
    est = _single_block_jeasy(est_block)
    # With the configuration above beta == 1.0 (see derivation in module notes).
    assert math.isclose(betakl(orig, est, 1, 2), 1.0, rel_tol=1e-9)


def test_f5_ekl_zero_when_perfect():
    orig_block = [0] * 64
    est_block = [0] * 64
    est_block[17] = 1
    orig = _single_block_jeasy(orig_block)
    est = _single_block_jeasy(est_block)
    assert math.isclose(f5_ekl(orig, est, 1.0, 1, 2), 0.0, abs_tol=1e-9)


def test_f5_dobeta_three_positions():
    orig_block = [0] * 64
    est_block = [0] * 64
    for pos in (10, 17, 18):  # (2,1)=10, (1,2)=17, (2,2)=18
        est_block[pos] = 1
    orig = _single_block_jeasy(orig_block)
    est = _single_block_jeasy(est_block)
    beta, ekl = f5_dobeta(orig, est)
    assert math.isclose(beta, 1.0, rel_tol=1e-9)
    assert math.isclose(ekl, 0.0, abs_tol=1e-9)


def test_detect_f5_runs_on_real_image():
    rng = np.random.RandomState(4)
    im = Image.fromarray(rng.randint(0, 256, (128, 128, 3), dtype=np.uint8), "RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90, subsampling=0)
    jpg = decode_coefficients(buf.getvalue())
    beta = detect_f5(jpg)
    assert math.isfinite(beta)
