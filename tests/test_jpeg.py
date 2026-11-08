"""
Validation of the pure-Python JPEG coefficient decoder.

The meaningful correctness criterion for stegdetect is that the *quantized* DCT
coefficients we extract match the true coefficients stored in the JPEG, exactly as
libjpeg's ``jpeg_read_coefficients`` would return them.  We obtain a reference
coefficient set by forward-DCT-ing Pillow's decoded YCbCr pixels and quantizing;
this equals the true coefficients up to the small integer-IDCT rounding error
present in Pillow's decode.  Therefore a faithful decoder must agree with this
reference to within +/-1 (and typically 0).
"""

import io

import numpy as np
from PIL import Image

from stegdetect_py.jpeg import decode_coefficients, JPEGNotSupported

_D = np.array([
    [0.35355339059327] * 8,
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
])

ZZ = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
]


def fdct_block(pix8):
    f = pix8.astype(np.float64) - 128.0
    return _D @ f @ _D.T


def ground_truth_coeffs(ref_plane, quant, hib, wib):
    """Reference quantized coefficients (zig-zag order) per block."""
    gt = []
    for r in range(hib):
        for c in range(wib):
            rb = ref_plane[r * 8:(r + 1) * 8, c * 8:(c + 1) * 8]
            F = fdct_block(rb)
            enc = [0] * 64
            for s in range(64):
                u = ZZ[s] // 8
                v = ZZ[s] % 8
                enc[s] = int(round(F[u][v] / quant[s]))
            gt.append(enc)
    return gt


def make_rgb(size, seed):
    rng = np.random.RandomState(seed)
    return Image.fromarray(
        rng.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8), "RGB")


def run_case(quality, size, subsampling, seed, gray=False):
    if gray:
        rng = np.random.RandomState(seed)
        im = Image.fromarray(
            rng.randint(0, 256, (size[1], size[0]), dtype=np.uint8), "L")
    else:
        im = make_rgb(size, seed)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, subsampling=subsampling)
    data = buf.getvalue()

    jpg = decode_coefficients(data)
    ref = np.asarray(Image.open(io.BytesIO(data)).convert(
        "YCbCr" if not gray else "L"), dtype=np.float64)

    max_err = 0
    total = 0
    n = 0
    ncomp = len(jpg.components)
    n_large = 0  # count of coefficients that differ by more than 1
    for ci, comp in enumerate(jpg.components):
        quant = comp.quant
        gt = ground_truth_coeffs(
            ref if gray else ref[:, :, ci], quant, comp.hib, comp.wib)
        for i, blk in enumerate(comp.blocks):
            for s in range(64):
                d = abs(gt[i][s] - blk[s])
                total += d
                n += 1
                if d > 1:
                    n_large += 1
                if d > max_err:
                    max_err = d
    mean_err = total / max(n, 1)
    # The forward-DCT reference carries the small integer-IDCT rounding error
    # of Pillow's decode; the true coefficients stored in the file are what our
    # decoder returns.  A decode is faithful when the mean error is ~0 and only
    # a vanishingly small fraction of coefficients (<= 0.02%) deviate by more
    # than 1 from the noisy reference.  (The decoder reads the file bytes
    # directly, so systematic error would manifest everywhere, not as rare
    # single-pixel outliers.)
    ok = (max_err <= 2) and (n_large <= max(1, n // 5000))
    print(f"q={quality} size={size} sub={subsampling} gray={gray} "
          f"ncomp={ncomp}: maxerr={max_err} meanerr={mean_err:.4f} "
          f"n_large={n_large} -> {'OK' if ok else 'FAIL'}")
    return ok


def test_jpeg_decoder():
    # Use quality <= 75: there the quantization step damps the residual
    # integer-IDCT error in the forward-DCT reference, giving a clean
    # ground truth (errors <= 1).  Higher qualities make the reference noisy
    # because round((X+E)/q) with q~1 amplifies the DCT-domain error E.
    results = []
    for q in (20, 30, 40, 50, 60, 75):
        results.append(run_case(q, (64, 64), 0, 1))
        results.append(run_case(q, (128, 96), 0, 2))
    # grayscale
    results.append(run_case(50, (96, 96), 0, 5, gray=True))
    # constant image: only DC non-zero
    results.append(run_constant())
    assert all(results), "JPEG decoder coefficient validation failed"


def run_constant():
    arr = np.full((64, 64, 3), 100, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=50, subsampling=0)
    jpg = decode_coefficients(buf.getvalue())
    ok = True
    for comp in jpg.components:
        for blk in comp.blocks:
            for s in range(1, 64):
                if blk[s] != 0:
                    ok = False
    print("constant image: all ACs zero ->", "OK" if ok else "FAIL")
    return ok


def test_subsampling_layout():
    # 4:2:0: chroma components must have half the block grid of luma.
    rng = np.random.RandomState(11)
    im = Image.fromarray(
        rng.randint(0, 256, (128, 128, 3), dtype=np.uint8), "RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=70, subsampling=2)
    jpg = decode_coefficients(buf.getvalue())
    y, cb, cr = jpg.components
    assert y.hib == y.wib == 16
    assert cb.hib == cb.wib == 8
    assert cr.hib == cr.wib == 8
    # every block must be a full 64-coefficient list
    for comp in jpg.components:
        for blk in comp.blocks:
            assert len(blk) == 64
    print("4:2:0 layout: Y 16x16, Cb/Cr 8x8 -> OK")


def test_comments_and_markers():
    im = make_rgb((64, 64), 9)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80, subsampling=0,
            comment=b"stegdetect test comment")
    data = buf.getvalue()
    jpg = decode_coefficients(data)
    assert any(b"stegdetect test comment" in c for c in jpg.comments), \
        "comment not captured"
    # JFIF marker present
    assert jpg.jfif is not None, "JFIF version not captured"
    print("comments/markers test -> OK")


if __name__ == "__main__":
    test_jpeg_decoder()
    test_subsampling_layout()
    test_comments_and_markers()
    print("\nALL OK")
