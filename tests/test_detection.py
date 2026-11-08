"""
Validation of the detection engine (stegdetect_py.detection) and the
coefficient-preparation helpers (stegdetect_py.common).

Strategy
--------
* Consistency: the cumulative-histogram fast path in histogram_chi_jsteg must
  agree with a from-scratch per-prefix chi2test (numerically identical values).
* Clean-image sanity: a freshly encoded JPEG must not trigger any detector.
* Core statistic: the chi-square p-value (f) on *real* JPEG DCT coefficients is
  low when clean and jumps toward 1.0 when the LSBs of the first N coefficients
  are randomised -- this is the detection primitive, tested deterministically.
* Embedding: embedding a properly-formed JSteg message (5-bit width field,
  width-bit length field, then random payload) into the first portion of the
  coefficient stream must make histogram_chi_jsteg fire (return > 0),
  faithfully reproducing the original tool's behaviour on a JSteg carrier.

Note on image size: stegdetect's JSteg detector initialises its binary search
window with ``end = bits/100`` (floored at 4000).  That search range only makes
sense for images whose coefficient count exceeds ~400k, so the embedding test
uses a 512x512 image (≈ 5e5 eligible coefficients).  Smaller test images are
still valid for the clean / core / consistency tests.
"""

import io

import numpy as np
from PIL import Image

from stegdetect_py import common, detection
from stegdetect_py.jpeg import decode_coefficients


def make_rgb(size, seed):
    rng = np.random.RandomState(seed)
    return Image.fromarray(
        rng.randint(0, 256, (size, size, 3), dtype=np.uint8), "RGB")


def load_jpg(im, quality=85, subsampling=0):
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, subsampling=subsampling)
    return decode_coefficients(buf.getvalue())


def embed_jsteg(jpg, msg_coeffs=1500, seed=7):
    """Embed a properly-formed JSteg message into the first ``msg_coeffs``
    eligible DCT coefficients (everything except {0, 1}), in natural order.

    Layout (as expected by ``common.jsteg_size``): 5-bit width, then ``width``
    bits holding the message length in bytes, then the random payload bits.  The
    remaining coefficients are left untouched (clean), mirroring how a real JSteg
    message occupies only the leading part of a carrier.

    Returns the embedded message length in bytes.
    """
    rng = np.random.RandomState(seed)
    width = 8  # bits used to encode the length field (max 255 bytes)
    bits = []
    for _ in range(5):
        bits.append((width >> (4 - _)) & 1)
    length = (msg_coeffs - 5 - width) // 8
    for _ in range(width):
        bits.append((length >> (width - 1 - _)) & 1)
    while len(bits) < msg_coeffs:
        bits.append(int(rng.randint(0, 2)))
    bits = bits[:msg_coeffs]

    idx = 0
    for comp in jpg.components:
        for blk in comp.blocks:
            for s in range(64):
                v = blk[s]
                if (v & 1) == v:        # 0 or 1 -> not an eligible slot
                    continue
                if idx < len(bits):
                    blk[s] = (v & ~1) | bits[idx]
                # slots beyond the message keep their (clean) LSBs
                idx += 1
    return length


# --------------------------------------------------------------------------- #

def test_cumulative_matches_naive():
    data = np.random.RandomState(1).randint(-50, 50, size=5000).tolist()
    bits = len(data)
    binidx, valid = detection._binidx(data)
    cum = {i: f for i, f in
           detection._chi2_over_prefix(binidx, valid, 250, bits, detection.unify_normal)}
    for i, fc in cum.items():
        fn = detection.chi2test(data, bits, detection.unify_normal, 0, i)
        assert abs(fc - fn) < 1e-9, (i, fc, fn)
    print("cumulative == naive (unify_normal): OK")


def test_prepare_consistency():
    im = make_rgb(40, 9)
    jpg = load_jpg(im)
    a, ba = common.prepare_all(jpg)
    n, bn = common.prepare_normal(jpg)
    js, bjs = common.prepare_jsteg(jpg)
    # normal and jsteg select the identical set (all coeffs except {0,1}).
    assert bn == bjs, (bn, bjs)
    # prepare_all contains every coefficient; normal excludes exactly {0,1}.
    assert ba > bn
    allv = a
    expected = [v for v in allv if not (v & 1) == v]
    assert n == expected, (len(n), len(expected))
    print("prepare_* consistency: OK")


def test_clean_detects_nothing():
    im = make_rgb(48, 3)
    jpg = load_jpg(im)
    dcts, bits = common.prepare_jsteg(jpg)
    jres = detection.histogram_chi_jsteg(dcts, bits)
    dcts_n, _ = common.prepare_normal(jpg)
    ores = detection.histogram_chi_outguess(dcts_n, bits)
    jp = detection.histogram_chi_jphide(dcts_n, bits) if common.JPHPOS[0] >= 500 else 0
    print(f"clean  jsteg={jres:.3f} outguess={ores:.3f} jphide={jp:.3f}")
    # Clean images should not be flagged.
    assert jres <= 0, jres
    assert ores <= 0, ores


def test_chi2_core_real_image():
    """Deterministic core check on real (non-uniform) JPEG coefficients.

    A real carrier's LSBs have structure, so the chi-square p-value f is low.
    Randomising the LSBs of the first N coefficients must push f toward 1.
    """
    im = make_rgb(160, 11)
    jpg = load_jpg(im)
    dcts, bits = common.prepare_jsteg(jpg)
    N = min(4000, bits)
    f_clean = detection.chi2test(dcts, bits, detection.unify_normal, 0, N)

    emb = list(dcts)
    rng = np.random.RandomState(5)
    for i in range(N):
        if (emb[i] & 1) == emb[i]:
            continue
        emb[i] = (emb[i] & ~1) | int(rng.randint(0, 2))
    f_emb = detection.chi2test(emb, bits, detection.unify_normal, 0, N)
    print(f"chi2 core: clean_f={f_clean:.4f}  embedded_f={f_emb:.4f}  (N={N})")
    assert f_clean <= 0.4, f_clean
    assert f_emb > 0.9, f_emb


def test_jsteg_embedding_fires():
    im = make_rgb(512, 4)
    jpg = load_jpg(im)
    dcts, bits = common.prepare_jsteg(jpg)
    before = detection.histogram_chi_jsteg(dcts, bits)

    embed_jsteg(jpg, msg_coeffs=1500)
    dcts2, bits2 = common.prepare_jsteg(jpg)
    after = detection.histogram_chi_jsteg(dcts2, bits2)
    print(f"jsteg score: clean={before:.4f}  embedded={after:.4f}")
    assert after > 0.5, "embedding should trigger JSteg detection"
    assert after > before + 0.5


if __name__ == "__main__":
    test_cumulative_matches_naive()
    test_prepare_consistency()
    test_clean_detects_nothing()
    test_chi2_core_real_image()
    test_jsteg_embedding_fires()
    print("\nALL DETECTION TESTS OK")
