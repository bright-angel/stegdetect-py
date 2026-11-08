"""Faithful Python port of stegdetect's ``f5.c`` (F5 slow-path detector).

This module implements the calibration/``beta``-``KL`` statistics used to spot
F5-embedded JPEGs.  The fast path actually used by ``stegdetect`` lives in
:func:`stegdetect_py.detection.histogram_f5`; the code here is the *slow*
recompression-based variant described in the original source.

Functions ported verbatim from the C:

* :func:`f5_hkl` -- count DCT blocks whose ``(il, ik)`` coefficient equals a
  value (0/1/2).
* :func:`f5_luminanceimage` -- reconstruct the luminance image from the DCT
  blocks (dequantize + IDCT + level shift).
* :func:`f5_crop`, :func:`f5_blur` -- spatial preprocessing.
* :func:`betakl`, :func:`f5_ekl`, :func:`f5_dobeta` -- the beta/KL divergence
  between the original and a recompressed ("estimated") image.
* :func:`detect_f5` -- orchestrates the pipeline and returns the ``beta`` score.

Faithful-data note: the C ``f5_compress`` re-encodes the reconstructed luminance
image with libjpeg, copying the *original* quantization tables (the ``quality ==
0`` branch).  A pure-Python environment has no public API to force arbitrary
quantization tables, so :func:`f5_compress` re-encodes via Pillow, passing the
original luminance quant table through the ``qtables`` argument when Pillow
accepts it and otherwise falling back to a high ``quality`` setting.  This is a
documented best-effort approximation; every other routine is exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .jpeg import decode_coefficients

try:  # The standard zig-zag scan order (zz[s] = row*8 + col for scan position s).
    from .jpeg import ZZ as _ZZ
except Exception:  # pragma: no cover - ZZ is always present in jpeg.py
    _ZZ = [
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
        12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
    ]

from .dct import dequant_block, idct


@dataclass
class Jeasy:
    """Mirror of the C ``struct jeasy`` used by the F5 routines.

    ``blocks[i]`` is a list of DCT blocks for component ``i`` stored in
    *natural* (row-major) order (the C ``jpeg_read_coefficients`` order), each
    block a flat 64-int list.  ``table[i]`` is the per-component quantization
    table in natural order.  ``needscale``/``scale`` support the per-component
    block-count scaling done in :func:`f5_dobeta`.
    """

    comp: int = 0
    height: List[int] = field(default_factory=list)
    width: List[int] = field(default_factory=list)
    blocks: List[List[List[int]]] = field(default_factory=list)
    table: List[List[int]] = field(default_factory=list)
    needscale: int = 0
    scale: List[float] = field(default_factory=list)

    @classmethod
    def from_jpeg(cls, jpg) -> "Jeasy":
        """Build a :class:`Jeasy` from a decoded :class:`stegdetect_py.jpeg.JPEG`.

        Converts each component's zig-zag-ordered blocks and quantization table
        into the natural order the C code expects.
        """
        je = cls()
        je.comp = len(jpg.components)
        for c in jpg.components:
            hib = c.hib
            wib = c.wib
            je.height.append(hib)
            je.width.append(wib)
            # zig-zag -> natural for the quant table
            quant_nat = [0] * 64
            qz = c.quant if c.quant is not None else [1] * 64
            for s in range(64):
                quant_nat[_ZZ[s]] = qz[s]
            je.table.append(quant_nat)
            # zig-zag -> natural for each block
            comp_blocks: List[List[int]] = []
            for blk in c.blocks:
                nat = [0] * 64
                for s in range(64):
                    nat[_ZZ[s]] = blk[s]
                comp_blocks.append(nat)
            je.blocks.append(comp_blocks)
        je.needscale = 0
        je.scale = [1.0] * je.comp
        return je


@dataclass
class F5Image:
    """Mirror of the C ``struct image`` (spatial pixel buffer)."""

    x: int = 0
    y: int = 0
    depth: int = 1
    max: int = 255
    arr: Optional[np.ndarray] = None  # dtype uint8, shape (y, x)

    def allocate(self):
        self.arr = np.zeros((self.y, self.x), dtype=np.uint8)


def f5_hkl(je: Jeasy, ik: int, il: int, val: int) -> int:
    """Count blocks of component 0 whose ``(il, ik)`` coefficient equals ``val``.

    Mirrors ``f5_hkl`` in ``f5.c`` (only component 0 is consulted -- the code's
    ``for (i = 0; i < 1; i++)`` loop).  ``ik`` is the column, ``il`` the row,
    both natural-order 8x8 indices.
    """
    pos = il * 8 + ik
    total = 0
    blocks = je.blocks[0]
    for blk in blocks:
        if blk[pos] == val:
            total += 1
    if je.needscale:
        total = int(total * je.scale[0])
    return total


def f5_luminanceimage(je: Jeasy, image: F5Image) -> None:
    """Reconstruct the luminance image from component-0 DCT blocks."""
    blocks = je.blocks[0]
    table = je.table[0]
    hib = je.height[0]
    wib = je.width[0]

    image.x = wib * 8
    image.y = hib * 8
    image.max = 255
    image.depth = 1
    image.allocate()

    rowspan = image.x * image.depth
    tmp = [0] * 64
    dq = [0] * 64
    for k in range(hib):
        for l in range(wib):
            dequant_block(dq, blocks[k * wib + l], table)
            tmp = idct(dq)
            for i in range(64):
                v = tmp[i] + 128
                if v > image.max:
                    v = image.max
                if v < 0:
                    v = 0
                x = l * 8 + (i % 8)
                y = k * 8 + (i // 8)
                image.arr[y, x] = int(v)


def f5_crop(image: F5Image) -> None:
    """Drop a 4-pixel border on every side (mirrors ``f5_crop``)."""
    new = F5Image()
    new.x = image.x - 8
    new.y = image.y - 8
    new.depth = image.depth
    new.max = image.max
    new.allocate()
    new.arr[:] = image.arr[4:image.y - 4, 4:image.x - 4]
    image.x, image.y, image.arr = new.x, new.y, new.arr  # type: ignore
    image.arr = new.arr


def f5_blur(image: F5Image, d: float) -> None:
    """3x3 box-ish blur with weight ``d`` on the four neighbours."""
    h, w = image.y, image.x
    src = image.arr.astype(np.int32)
    new = src.copy()
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            for c in range(image.depth):
                val = (d * src[y, x - 1] + d * src[y, x + 1]
                       + d * src[y - 1, x] + d * src[y + 1, x]
                       + (1 - 4 * d) * src[y, x])
                new[y, x] = int(val)
    image.arr = new.astype(np.uint8)


def f5_compress(image: F5Image, je: Jeasy, quality: int, pfin=None):
    """Re-encode the spatial image (best-effort libjpeg replacement).

    Returns a freshly decoded :class:`stegdetect_py.jpeg.JPEG` of the
    recompressed data so callers can build a new :class:`Jeasy` from it.  See
    the module docstring for the quantization-table approximation note.
    """
    import io
    from PIL import Image as PILImage

    pil = PILImage.fromarray(image.arr, mode="L")
    buf = io.BytesIO()
    if quality == 0:
        # Use the original luminance quant table.  Pillow expects zig-zag order.
        qnat = je.table[0]
        qzig = [0] * 64
        for s in range(64):
            qzig[s] = qnat[_ZZ[s]]
        try:
            pil.save(buf, format="JPEG", qtables=[qzig], quality=100)
        except Exception:
            pil.save(buf, format="JPEG", quality=95)
    else:
        pil.save(buf, format="JPEG", quality=quality)
    return decode_coefficients(buf.getvalue())


def f5_decompress(jinfo) -> F5Image:
    """Decode a JPEG object to a spatial image (used by the disabled path)."""
    import io
    from PIL import Image as PILImage

    data = getattr(jinfo, "raw_bytes", None)
    if data is None:
        raise ValueError("f5_decompress: no raw bytes available")
    pil = PILImage.open(io.BytesIO(data))
    arr = np.asarray(pil.convert("L"), dtype=np.uint8)
    img = F5Image(x=arr.shape[1], y=arr.shape[0], depth=1)
    img.arr = arr
    return img


def betakl(orig: Jeasy, est: Jeasy, k: int, l: int) -> float:
    first = f5_hkl(est, k, l, 1) * (f5_hkl(orig, k, l, 0) - f5_hkl(est, k, l, 0))
    second = (f5_hkl(orig, k, l, 1) - f5_hkl(est, k, l, 1)) * (
        f5_hkl(est, k, l, 2) - f5_hkl(est, k, l, 1))
    third = f5_hkl(est, k, l, 1)
    third = third * third
    fourth = f5_hkl(est, k, l, 2) - f5_hkl(est, k, l, 1)
    fourth = fourth * fourth
    return float(first + second) / float(third + fourth)


def f5_ekl(orig: Jeasy, est: Jeasy, beta: float, k: int, l: int) -> float:
    first = f5_hkl(orig, k, l, 0) - f5_hkl(est, k, l, 0) - beta * f5_hkl(est, k, l, 1)
    first = first * first
    for j in range(1, 3):
        second = (f5_hkl(orig, k, l, j) - (1 - beta) * f5_hkl(est, k, l, j)
                  - beta * f5_hkl(est, k, l, j + 1))
        second = second * second
        first += second
    return float(first)


def f5_dobeta(je: Jeasy, jne: Jeasy, verbose: int = 0) -> Tuple[float, float]:
    """Compute the beta/KL divergence between ``je`` (original) and ``jne``.

    Returns ``(beta, ekl)``.  Mirrors ``f5_dobeta`` -- it sets the estimated
    image's ``needscale`` and per-component scale, then averages the three
    beta/KL terms at positions (1,2), (2,1), (2,2).
    """
    jne.needscale = 1
    for i in range(jne.comp):
        jne.scale[i] = ((je.width[i] * je.height[i])
                        / float(jne.width[i] * jne.height[i]))

    b1 = betakl(je, jne, 1, 2)
    b2 = betakl(je, jne, 2, 1)
    b3 = betakl(je, jne, 2, 2)

    beta = (b1 + b2 + b3) / 3.0
    ekl = (f5_ekl(je, jne, beta, 1, 2)
           + f5_ekl(je, jne, beta, 2, 1)
           + f5_ekl(je, jne, beta, 2, 2))

    if verbose:
        print("Beta: %f - %f" % (beta, ekl))

    return beta, ekl


f5_elim2compress = 0


def detect_f5(jpeg) -> float:
    """Run the F5 slow-path detector on a decoded JPEG; return the beta score.

    Mirrors ``detect_f5`` with ``f5_elim2compress == 0``: reconstruct the
    luminance image, crop, blur, recompress, and measure the divergence.
    """
    je = Jeasy.from_jpeg(jpeg)

    image = F5Image()
    f5_luminanceimage(je, image)
    f5_crop(image)
    f5_blur(image, 0.05)

    jnew = f5_compress(image, je, 0, None)
    jne = Jeasy.from_jpeg(jnew)

    beta, _ekl = f5_dobeta(je, jne)
    return beta
