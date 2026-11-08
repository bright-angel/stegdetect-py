"""
stegdetect command-line interface -- a faithful Python port of stegdetect.c.

Detects steganography in JPEG images:
  * jsteg      -- LSB embedding in the quantized DCT coefficients
  * outguess   -- (old) LSB embedding variant
  * jphide     -- positional chi-square detector (ltab traversal)
  * f5         -- fast path: F5 encoder comment signature (+ optional slow
                 recompression attack, see F5_SLOW_ENABLED below)
  * invisible  -- Invisible Secrets comment-structure signature
  * appended   -- data appended after the JPEG EOI marker

Usage mirrors the original:
  stegdetect [-nqV] [-s <float>] [-d <num>] [-t <tests>] [file.jpg ...]
also reads a list of filenames from stdin when none are given on the command
line.
"""

from __future__ import annotations

import sys
from typing import List, Optional

from . import common, detection
from .jpeg import JPEGError

# ---- flag bits (identical to stegdetect.c) -----------------------------------
FLAG_DOOUTGUESS = 0x0001
FLAG_DOJPHIDE   = 0x0002
FLAG_DOJSTEG    = 0x0004
FLAG_DOINVIS    = 0x0008
FLAG_DOF5       = 0x0010
FLAG_DOF5_SLOW  = 0x0020
FLAG_DOAPPEND   = 0x0040
FLAG_DOTRANSF   = 0x0080
FLAG_DOCLASSDIS = 0x0100
FLAG_CHECKHDRS  = 0x0200

VERSION = "0.6"  # stegdetect release this port is based on

# Optional F5 recompression attack.  Requires Pillow (already a dependency) to
# re-encode the decompressed image; it is a best-effort approximation of the
# original libjpeg calibration and is disabled by default.
F5_SLOW_ENABLED = False


# ---- globals (mirror the C statics) ------------------------------------------
debug = 0
quiet = False
scale = 1.0
ispositive = 0
transform = None
transformname = None


# ---- output helper ----------------------------------------------------------
def quality(prepend: str, q: float) -> str:
    """Render ``prepend(***)`` with up to three stars, matching quality()."""
    n = min(int(q), 3) if q > 0 else 0
    return "%s(%s)" % (prepend, "*" * n)


def detect_append(raw: bytes) -> bytes:
    """Return bytes that follow the JPEG EOI (FFD9) marker, if any."""
    idx = raw.find(b"\xff\xd9")
    if idx == -1:
        return b""
    return raw[idx + 2:]


def check_invisible() -> Optional[int]:
    """Invisible Secrets: comments[1] begins with a little-endian length that
    should match the total bytes of the following comment segments."""
    comments = common.comments
    commentsize = common.commentsize
    if len(comments) < 2 or commentsize[1] < 4:
        return None
    p = comments[1]
    length = (p[3] << 24) | (p[2] << 16) | (p[1] << 8) | p[0]
    ol = length
    length += 4
    if commentsize[1] == length:
        return ol
    for i in range(1, len(comments)):
        if length <= 0 or commentsize[i] > length:
            break
        length -= commentsize[i]
    if length == 0:
        return ol
    return None


def check_f5_fast() -> bool:
    """F5 fast path: the encoder embeds a fixed 63-byte comment string."""
    sig = b"JPEG Encoder Copyright 1998, James R. Weeks and BioElectroMech."
    return (common.ncomments == 1 and common.commentsize[0] == 63 and
            common.comments[0] == sig)


def detect_f5_slow(filename: str) -> Optional[float]:
    """Best-effort F5 recompression (calibration) attack.

    The original re-encodes the decompressed image at a range of quality
    settings and compares the distribution of DCT coefficient values with the
    original; a small KL/beta distance indicates F5.  Here we re-encode with
    Pillow at a few qualities, decode the coefficients with our own decoder, and
    return the minimum normalised distance as a proxy for ``beta``.
    """
    try:
        from PIL import Image
        import io
        import numpy as np
        from .jpeg import decode_coefficients
    except Exception:
        return None
    try:
        im = Image.open(filename).convert("RGB")
    except Exception:
        return None

    def coeff_hist(img, q):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q, subsampling=0)
        jpg = decode_coefficients(buf.getvalue())
        dcts, bits = common.prepare_normal(jpg)
        h = [0.0] * 257
        for v in dcts:
            if -128 <= v <= 127:
                h[v + 128] += 1
        s = sum(h) or 1.0
        return [x / s for x in h]

    base = coeff_hist(im, 75)
    best = 1.0
    for q in (50, 60, 70, 80, 90):
        h = coeff_hist(im, q)
        kl = 0.0
        for i in range(257):
            a, b = base[i], h[i]
            if a > 0 and b > 0:
                kl += 0.5 * (a - b) * (np.log(a) - np.log(b))
        best = min(best, kl)
    return best


# ---- core detect ------------------------------------------------------------
def detect(filename: str, scans: int) -> None:
    # Raw bytes are needed for the appended-data check.
    try:
        with open(filename, "rb") as fh:
            raw = fh.read()
    except OSError as e:
        sys.stderr.write("%s: %s\n" % (filename, e))
        return

    try:
        jpg = common.load_image(filename)
    except (common.NotRGBError, JPEGError) as e:
        sys.stderr.write("%s: %s\n" % (filename, e))
        return

    return detect_jpg(jpg, raw, filename, scans)


def detect_jpg(jpg, raw: bytes, filename: str, scans: int) -> str:
    """Run all enabled detectors on an already-loaded JPEG object.

    Returns the single output line (without trailing newline) so it can be
    asserted in tests; also prints it when appropriate.
    """
    out = [filename + " :"]
    flag = 0  # 0 = negative, >0 = detected, -1 = skipped

    # ----- appended data (after EOI) -----
    if scans & FLAG_DOAPPEND:
        app = detect_append(raw)
        if app:
            flag = 1
            out.append(" appended(%d bytes)" % len(app))

    # ----- header / comment check (-n) -----
    if scans & FLAG_CHECKHDRS:
        if common.ncomments:
            scans = 0
        elif common.jpg_markers:
            scans = 0
        elif jpg.jfif is not None and (jpg.jfif[0] != 1 or jpg.jfif[1] != 1):
            scans &= ~FLAG_DOOUTGUESS

    # ----- invisible secrets -----
    if scans & FLAG_DOINVIS:
        ol = check_invisible()
        if ol is not None:
            flag = 1
            out.append(" invisible[%d](***)" % ol)

    # ----- f5 -----
    if scans & FLAG_DOF5:
        if check_f5_fast():
            flag = 1
            out.append(" f5(***)")
        elif scans & FLAG_DOF5_SLOW and F5_SLOW_ENABLED:
            beta = detect_f5_slow(filename)
            if beta is not None and beta >= 0.25:
                stars = 1 + (beta > 0.25) + (beta > 0.4)
                out.append(quality(" f5[%f]" % beta, stars))
                flag = 1

    # ----- jsteg -----
    if scans & FLAG_DOJSTEG:
        dcts, bits = common.prepare_jsteg(jpg)
        if dcts is not None:
            res = detection.histogram_chi_jsteg(dcts, bits)
            if res > 0:
                out.append(quality(" jsteg", res))
                flag = 1
                scans &= ~(FLAG_DOOUTGUESS | FLAG_DOJPHIDE)
            elif res == -1:
                out.append(" skipped (false positive likely)")
                if not flag:
                    flag = -1
                scans &= ~(FLAG_DOOUTGUESS | FLAG_DOJPHIDE)

    # ----- outguess (old) -----
    if scans & FLAG_DOOUTGUESS:
        dcts, bits = common.prepare_normal(jpg)
        if dcts is not None:
            # The C loop runs only for n=1 (identity order) because of the
            # ``while (n < 2)'' guard; we replicate that faithfully.
            res = detection.histogram_chi_outguess(dcts, bits)
            if res:
                out.append(quality(" outguess(old)", res))
                flag = 1

    # ----- jphide -----
    if scans & FLAG_DOJPHIDE:
        dcts, bits = common.prepare_jphide(jpg)
        if dcts is not None:
            res = detection.histogram_chi_jphide(dcts, bits)
            if not res and hasattr(detection, "histogram_chi_jphide_old"):
                res = detection.histogram_chi_jphide_old(dcts, bits)
            if res:
                out.append(quality(" jphide", res))
                flag = 1

    if not flag:
        out.append(" negative")

    line = "".join(out)
    if flag > 0 or not quiet:
        sys.stdout.write("%s\n" % line)
        sys.stdout.flush()
    return line


def dohistogram(filename: str) -> None:
    try:
        jpg = common.load_image(filename)
    except (common.NotRGBError, JPEGError) as e:
        sys.stderr.write("%s: %s\n" % (filename, e))
        return
    dcts, bits = common.prepare_all(jpg)
    sys.stdout.write("%s ->\n" % filename)
    h = [0] * 257
    for v in dcts:
        if -128 <= v <= 127:
            h[v + 128] += 1
    for v in range(257):
        if h[v]:
            sys.stdout.write("  %d: %d\n" % (v - 128, h[v]))


# ---- argument parsing -------------------------------------------------------
VALUE_FLAGS = set("cDCsdtCV")


def _usage(progname: str) -> None:
    sys.stderr.write(
        "Usage: %s [-nqV] [-s <float>] [-d <num>] [-t <tests>] "
        "[-C <num>]\n\t [file.jpg ...]\n" % progname)


def main(argv: Optional[List[str]] = None) -> int:
    global debug, quiet, scale, ispositive, transform, transformname

    if argv is None:
        argv = sys.argv[1:]

    progname = "stegdetect"
    scans = (FLAG_DOOUTGUESS | FLAG_DOJPHIDE | FLAG_DOJSTEG |
             FLAG_DOINVIS | FLAG_DOF5 | FLAG_DOAPPEND)
    histonly = False
    checkhdr = False
    explicit_tests = None
    classification = False  # -c / -C / -D (discrimination subsystem, TODO)

    i = 0
    while i < len(argv):
        a = argv[i]
        if not a.startswith("-") or a == "-":
            break
        flags = a[1:]
        j = 0
        while j < len(flags):
            ch = flags[j]
            if ch in VALUE_FLAGS:
                if j + 1 < len(flags):
                    val = flags[j + 1:]
                    j = len(flags)  # consume rest of this cluster
                elif i + 1 < len(argv):
                    val = argv[i + 1]
                    i += 1
                    j += 1
                else:
                    val = ""
                    j += 1
                _handle_opt(ch, val, progname)
                if ch == "t":
                    explicit_tests = _parse_tests(val)
                elif ch == "s":
                    scale = float(val) if val else 0.0
                elif ch == "d":
                    debug = int(val) if val else 0
                elif ch in ("c", "C", "D"):
                    classification = True
            else:
                _handle_opt(ch, "", progname)
                if ch == "h":
                    histonly = True
                elif ch == "n":
                    checkhdr = True
                elif ch == "q":
                    global quiet
                    quiet = True
                elif ch == "V":
                    sys.stdout.write("Stegdetect Version %s\n" % VERSION)
                    sys.exit(0)
                j += 1
        i += 1

    if explicit_tests is not None:
        scans = explicit_tests
    if checkhdr:
        scans |= FLAG_CHECKHDRS

    if classification:
        sys.stderr.write(
            "stegdetect: -c/-C/-D (classification/transform) is not yet "
            "implemented in this Python port\n")

    files = argv[i:]

    if not files:
        for line in sys.stdin:
            fn = line.strip()
            if not fn:
                continue
            if histonly:
                dohistogram(fn)
            else:
                detect(fn, scans)
    else:
        for fn in files:
            if histonly:
                dohistogram(fn)
            else:
                detect(fn, scans)
    return 0


def _parse_tests(val: str) -> int:
    tests = 0
    for c in val:
        if c == "o":
            tests |= FLAG_DOOUTGUESS
        elif c == "j":
            tests |= FLAG_DOJSTEG
        elif c == "p":
            tests |= FLAG_DOJPHIDE
        elif c == "i":
            tests |= FLAG_DOINVIS
        elif c == "f":
            tests |= FLAG_DOF5
        elif c == "F":
            tests |= FLAG_DOF5 | FLAG_DOF5_SLOW
        elif c == "a":
            tests |= FLAG_DOAPPEND
        else:
            _usage("stegdetect")
            sys.exit(1)
    return tests


def _handle_opt(ch: str, val: str, progname: str) -> None:
    # Only -V (version) has an immediate side effect here; the other value
    # flags are consumed by the caller.  -h/-n/-q are handled in main().
    if ch == "V":
        sys.stdout.write("Stegdetect Version %s\n" % VERSION)
        sys.exit(0)


if __name__ == "__main__":
    main()
