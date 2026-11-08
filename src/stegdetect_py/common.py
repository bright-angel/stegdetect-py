"""
Image loading and coefficient preparation, faithful to ``common.c``.

A JPEG is decoded by our pure-Python decoder (:mod:`stegdetect_py.jpeg`) into a
:class:`JPEG` whose ``components`` each carry a grid of 8x8 blocks stored in
*zig-zag (scan) order* -- exactly the layout libjpeg exposes via
``jpeg_read_coefficients`` (``dctcompbuf[comp][row][col][i]`` with ``i`` in
zig-zag order).  The ``prepare_*`` functions below select and flatten those
coefficients exactly as the C sources do, producing the ``(dcts, bits)`` pair
consumed by the detection engine.

Note on ordering: ``buildDCThist`` builds an order-independent histogram, so for
jsteg / outguess only the *set* of selected coefficients matters (the original's
MCU-order vs natural-order callbacks yield the same set).  JPHide, however, feeds
positional slices of the prepared array to ``chi2test`` (e.g. ``[0, jphpos[0])``),
so its traversal order (``ltab``) is reproduced exactly.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .jpeg import decode_coefficients, JPEG, JPEGError

DCTSIZE = 8
DCTSIZE2 = 64
JPHMAXPOS = 2

# JPHide DCT traversal table: triples of (component, spos, mode).  Public domain
# (identical to jphide_table.c).  A negative component terminates the table.
LTAB: List[int] = [
    0, 0, 0, 2, 0, 0, 1, 0, 0,
    0, 8, 0, 0, 1, 0, 0, 9, 0,
    0, 16, 0, 0, 25, 0, 0, 11, 0,
    0, 2, 0, 0, 24, 0, 2, 1, 1,
    1, 1, 1, 2, 8, 1, 1, 8, 1,
    0, 17, 0, 0, 10, 0, 0, 3, 0,
    0, 32, 0, 0, 18, 0, 0, 40, 0,
    2, 2, 1, 1, 2, 1, 0, 26, 0,
    0, 4, 0, 0, 33, 0, 2, 16, 1,
    1, 16, 1, 0, 19, 0, 2, 9, 1,
    1, 9, 1, 0, 27, 0, 2, 10, 2,
    1, 10, 2, 2, 17, 2, 1, 17, 2,
    0, 12, 0, 0, 34, 0, 0, 41, 0,
    0, 48, 0, 2, 3, 2, 1, 3, 2,
    0, 5, 1, 0, 20, 1, 2, 18, 2,
    1, 18, 2, 2, 24, 2, 1, 24, 2,
    2, 11, 2, 1, 11, 2, 0, 42, 1,
    0, 35, 1, 0, 28, 1, 2, 25, 2,
    1, 25, 2, 0, 49, 1, 0, 6, 1,
    0, 21, 2, 0, 43, 2, 0, 13, 2,
    0, 36, 2, 0, 50, 2, 0, 56, 2,
    0, 14, 2, 0, 22, 2, 0, 8, -15,
    0, 9, -7, 0, 0, -15, 0, 1, -7,
    0, 16, -7, 0, 2, -7, 0, 17, -3,
    0, 10, -3, 0, 23, 2, 0, 25, -3,
    0, 18, -3, 0, 24, -3, 0, 7, 2,
    0, 3, -3, 0, 32, -3, 2, 4, 2,
    1, 4, 2, 2, 19, 2, 1, 19, 2,
    0, 29, 2, 0, 11, -3, 2, 26, 2,
    1, 26, 2, 2, 12, 2, 1, 12, 2,
    0, 51, 2, 0, 15, 2, 0, 44, 2,
    2, 27, 2, 1, 27, 2, 0, 57, 2,
    2, 5, 2, 1, 5, 2, 0, 31, 2,
    0, 58, 2, 0, 30, 2, 2, 40, 2,
    1, 40, 2, 0, 26, -3, 2, 20, 2,
    1, 20, 2, 0, 33, -3, 0, 19, -3,
    2, 32, 2, 1, 32, 2, 2, 33, 2,
    1, 33, 2, 2, 13, 2, 1, 13, 2,
    2, 34, 2, 1, 34, 2, 0, 59, 2,
    0, 37, 2, 0, 4, -3, 2, 28, 2,
    1, 28, 2, 2, 35, 2, 1, 35, 2,
    0, 52, 2, 2, 21, 2, 1, 21, 2,
    0, 39, 2, 2, 6, 2, 1, 6, 2,
    0, 38, 2, 0, 45, 2, 0, 12, -3,
    2, 14, 2, 1, 14, 2, 2, 36, 2,
    1, 36, 2, 0, 40, -3, 0, 60, 2,
    2, 29, 2, 1, 29, 2, 2, 22, 2,
    1, 22, 2, 2, 42, 2, 1, 42, 2,
    2, 7, 2, 1, 7, 2, 0, 27, -3,
    2, 43, 2, 1, 43, 2, 2, 41, 2,
    1, 41, 2, 2, 15, 2, 1, 15, 2,
    0, 47, 2, 0, 46, 2, 2, 48, 2,
    1, 48, 2, 0, 53, 2, 0, 61, 2,
    0, 54, 2, 0, 62, 3, 0, 55, 3,
    0, 34, -3, 0, 63, 3, 0, 20, -3,
    2, 23, 3, 1, 23, 3, 2, 44, 3,
    1, 44, 3, 2, 37, 3, 1, 37, 3,
    2, 30, 3, 1, 30, 3, 2, 50, 3,
    1, 50, 3, 2, 52, 3, 1, 52, 3,
    2, 31, 3, 1, 31, 3, 2, 49, 3,
    2, 38, 3, 1, 49, 3, 1, 38, 3,
    2, 53, 3, 1, 53, 3, 2, 45, 3,
    1, 45, 3, 2, 51, 3, 1, 51, 3,
    2, 56, 3, 1, 56, 3, 2, 39, 3,
    1, 39, 3, 2, 46, 3, 1, 46, 3,
    0, 5, -3, 0, 41, -3, 2, 57, 3,
    1, 57, 3, 2, 58, 3, 1, 58, 3,
    2, 60, 3, 1, 60, 3, 2, 47, 3,
    1, 47, 3, 2, 61, 3, 2, 54, 3,
    1, 61, 3, 1, 54, 3, 2, 59, 3,
    1, 59, 3, 2, 62, 3, 1, 62, 3,
    2, 55, 3, 1, 55, 3, 2, 63, 3,
    1, 63, 3, 0, 13, -3, 0, 6, -3,
    0, 28, -3, 0, 48, -3, 0, 21, -3,
    0, 35, -3, 0, 42, -3, 0, 36, -3,
    0, 23, -3, 0, 14, -3, 0, 7, -3,
    0, 49, -3, 0, 43, -3, 0, 22, -3,
    0, 63, -3, 0, 62, -3, 0, 61, -3,
    0, 60, -3, 0, 59, -3, 0, 58, -3,
    0, 57, -3, 0, 56, -3, 0, 55, -3,
    0, 54, -3, 0, 53, -3, 0, 52, -3,
    0, 51, -3, 0, 50, -3, 0, 47, -3,
    0, 46, -3, 0, 45, -3, 0, 44, -3,
    0, 39, -3, 0, 38, -3, 0, 37, -3,
    0, 31, -3, 0, 30, -3, 0, 29, -3,
    0, 15, -3, -1, 0, 0,
]

# Populated by prepare_jphide(); consulted by the JPHide detector.
JPHPOS = [0, 0]

# Comments captured during loading (mirrors comments[]/commentsize[] in common.c).
comments: List[bytes] = []
commentsize: List[int] = []
ncomments: int = 0
jpg_markers: int = 0  # bitmask of APPn presence (1 << (APPn - APP0))


class NotRGBError(JPEGError):
    pass


def load_image(path: str) -> JPEG:
    """Decode ``path`` and return the :class:`JPEG` with quantized coefficients.

    The original ``jpg_open`` requires an RGB image with exactly three color
    components; we replicate that requirement so the detection logic (which hard
    codes three components) behaves identically.
    """
    global comments, commentsize, ncomments, jpg_markers
    with open(path, "rb") as fh:
        data = fh.read()
    jpg = decode_coefficients(data)
    if len(jpg.components) != 3:
        raise NotRGBError(
            "%s: error: wrong number of color components: %d "
            "(stegdetect requires a 3-component RGB JPEG)" % (path, len(jpg.components)))

    comments = list(jpg.comments)
    commentsize = [len(c) for c in jpg.comments]
    ncomments = len(jpg.comments)
    jpg_markers = jpg.markers
    return jpg


def _require_three(jpg: JPEG) -> None:
    if len(jpg.components) != 3:
        raise NotRGBError("requires a 3-component RGB JPEG")


def prepare_all(jpg: JPEG) -> Tuple[List[int], int]:
    """All quantized coefficients of every component, in natural order.

    Order: for each component, for each block row, for each block column, for
    i in 0..63 (zig-zag order): append value.  (Equivalent to ``dctcompbuf``.)
    """
    _require_three(jpg)
    dcts: List[int] = []
    for comp in jpg.components:
        for blk in comp.blocks:
            dcts.extend(blk)
    return dcts, len(dcts)


def prepare_all_gradx(jpg: JPEG) -> Tuple[List[int], int]:
    """Horizontal gradient of every DCT block, faithful to common.c.

    For each component, for each block row, for each pair of horizontally
    adjacent blocks (col, col+1), for each of the 64 coefficients::

        grad[bits++] = block[col][i] - block[col+1][i]

    The last column of every row is skipped (no right neighbour).  This is what
    ``extraction.c::gradient_transform`` consumes.
    """
    _require_three(jpg)
    dcts: List[int] = []
    for c in jpg.components:
        wib, hib = c.wib, c.hib
        if wib - 1 <= 0:
            raise ValueError("image too small for gradx")
        blocks = c.blocks
        for row in range(hib):
            for col in range(wib - 1):
                b0 = blocks[row * wib + col]
                b1 = blocks[row * wib + col + 1]
                for i in range(DCTSIZE2):
                    dcts.append(b0[i] - b1[i])
    return dcts, len(dcts)


def prepare_normal(jpg: JPEG) -> Tuple[List[int], int]:
    """Coefficients excluding values in {0, 1} (skip ``(val & 1) == val``).

    Used by the OutGuess detector.
    """
    _require_three(jpg)
    dcts: List[int] = []
    for comp in jpg.components:
        for blk in comp.blocks:
            for val in blk:
                if (val & 1) == val:
                    continue
                dcts.append(val)
    return dcts, len(dcts)


def prepare_jsteg(jpg: JPEG) -> Tuple[List[int], int]:
    """Coefficients excluding {0, 1} -- the JSteg embedding set.

    The C implementation collects these in MCU order via the ``jsteg_cb``
    callback; because the chi-square statistic is order independent, the resulting
    *set* is identical to :func:`prepare_normal`, so we reuse the same selection.
    """
    return prepare_normal(jpg)


def jsteg_size(dcts: List[int], bits: int, off: Optional[List[int]] = None) -> int:
    """Recover the embedded message length field from a JSteg carrier.

    Mirrors common.c's ``jsteg_size``: read ``width`` (5 LSBs) then ``len``
    (``width`` bits), interpreting each coefficient's LSB.
    """
    width = mode = length = count = 0
    i = 0
    for i in range(bits):
        val = dcts[i]
        if mode == 0:
            width = (width << 1) | (val & 1)
        else:
            length = (length << 1) | (val & 1)
        count += 1
        if mode == 0 and count >= 5:
            mode = 1
            count = 0
        elif mode == 1 and count >= width:
            break
    if off is not None:
        off[0] = i + 1
    return length


def prepare_jphide(jpg: JPEG) -> Tuple[List[int], int]:
    """JPHide coefficient traversal, faithful to ``common.c::prepare_jphide``.

    Walks the ``LTAB`` sequence, emitting coefficients in the exact order the
    original produced (so positional slices such as ``[0, jphpos[0])`` line up),
    and records ``JPHPOS[0]`` / ``JPHPOS[1]``.
    """
    _require_three(jpg)
    comps = jpg.components
    wib = [c.wib for c in comps]
    hib = [c.hib for c in comps]
    lwib = [DCTSIZE2 * wib[c] - 1 for c in range(3)]
    mbits = sum(hib[c] * wib[c] * DCTSIZE2 for c in range(3))

    dcts: List[int] = []
    # ``back`` records which linear offsets have already been emitted.
    back = [[0] * (hib[c] * wib[c] * DCTSIZE2) for c in range(3)]

    comp = LTAB[0]
    spos = LTAB[1]
    mode = LTAB[2]
    nheight = 0
    nwidth = spos - DCTSIZE2
    bits = j = 0

    while bits < mbits:
        nwidth += DCTSIZE2
        if nwidth > lwib[comp]:
            nwidth = spos
            nheight += 1
            if nheight >= hib[comp]:
                if j == 0:
                    JPHPOS[0] = bits
                if j == 3:
                    JPHPOS[1] = bits
                j += 3
                if LTAB[j] < 0:
                    break
                comp = LTAB[j]
                nwidth = spos = LTAB[j + 1]
                mode = LTAB[j + 2]
                nheight = 0
        # Protect the IV (first eight coefficients of component 0, height 0).
        if comp == 0 and nheight == 0 and nwidth <= 7:
            continue
        c = comps[comp]
        blk_idx = nheight * wib[comp] + (nwidth // DCTSIZE2)
        val = c.blocks[blk_idx][nwidth % DCTSIZE2]
        if mode < 0:
            continue
        elif mode > 1:
            continue
        elif mode and -1 <= val <= 1:
            continue
        off = nheight * wib[comp] * DCTSIZE2 + nwidth
        if back[comp][off]:
            break
        back[comp][off] = 1
        dcts.append(val)
        bits += 1

    return dcts, bits


# --- randomness classification (appended / camouflage / alpha-channel) ---------

_TABLE = [0] * 256


def _init_table() -> None:
    _TABLE[0] = 0
    _TABLE[1] = 1
    _TABLE[2] = 1
    _TABLE[3] = 2
    _TABLE[4] = 1
    _TABLE[5] = 2
    _TABLE[6] = 2
    _TABLE[7] = 3
    for i in range(8, 16):
        _TABLE[i] = 1 + _TABLE[i & 0x7]
    for i in range(16, 256):
        _TABLE[i] = _TABLE[(i >> 4) & 0xF] + _TABLE[i & 0xF]


_init_table()


def is_random(buf: bytes, size: int) -> bool:
    """Classify ``buf`` as random data (True) or not, per ``common.c``.

    Combines a bit-balance test (fraction of 1-bits near 0.5) with a 64-bucket
    chi-square test over overlapping bit windows.
    """
    if size <= 0:
        return False
    one = 0
    for i in range(size):
        one += _TABLE[buf[i]]
    ratio = one / (size * 8.0)
    if ratio < 0.46 or ratio > 0.54:
        return False

    NBUCKETS = 64
    bucket = [0] * NBUCKETS
    val = 0
    p = 1  # index into buf (we consume buf[0] specially below)
    one_reg = buf[0]
    for j in range((size - 1) * 8):
        bucket[one_reg & (NBUCKETS - 1)] += 1
        if (j % 8) == 0:
            val = buf[p]
            p += 1
        one_reg >>= 1
        one_reg |= (val & 0x80)
        val <<= 1

    exp = j / NBUCKETS
    total = 0.0
    for i in range(NBUCKETS):
        d = bucket[i] - exp
        total += d * d / exp
    if total > 160:
        return False
    return True


# --- file-type identification (libmagic-style success gate) ------------------
# The original stegbreak links against a trimmed libmagic (file_process).  The
# crackers use it only as a boolean "does this decrypted buffer look like a real
# file?".  We implement the same contract: return a non-zero value for a
# recognised file type (binary magic or ASCII text), 0 otherwise.

_MAGICS = [
    b"\x89PNG\r\n\x1a\n",            # PNG
    b"\xff\xd8\xff",                  # JPEG
    b"GIF87a",                        # GIF87a
    b"GIF89a",                        # GIF89a
    b"%PDF-",                         # PDF
    b"PK\x03\x04",                    # ZIP
    b"\x1f\x8b",                      # gzip
    b"BM",                            # BMP
    b"\x42\x5a\x68",                  # bzip2 ("BZh")
    b"\x37\x7a\xbc\xaf\x27\x1c",      # 7z
    b"RIFF",                          # RIFF (WAV/AVI)
    b"\x50\x4b\x05\x06",              # ZIP empty archive
    b"\xca\xfe\xba\xbe",              # Java class
]


def file_process(data: bytes, nbytes: Optional[int] = None) -> int:
    """Return a truthy value if ``data`` resembles a known file type, else 0.

    Mirrors the contract of libmagic's ``file_process`` as used by the crackers:
    a non-zero result means "this is a plausible embedded payload".
    """
    if nbytes is None:
        nbytes = len(data)
    if nbytes <= 0:
        return 0

    for magic in _MAGICS:
        if data[:len(magic)] == magic:
            return ord("s")

    # ASCII-text heuristic (ascmagic): mostly printable, contains a letter.
    printable = 0
    has_letter = False
    for i in range(nbytes):
        c = data[i]
        if 32 <= c < 127 or c in (9, 10, 13):
            printable += 1
        else:
            # high-bit / control byte -> not plain ASCII text
            if c >= 128 or (c < 32 and c not in (9, 10, 13)):
                return 0
        if chr(c).isalpha():
            has_letter = True
    if printable == nbytes and has_letter:
        return ord("a")

    return 0

