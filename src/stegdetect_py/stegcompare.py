"""stegcompare — compare the DCT coefficients of two JPEG images.

Faithful Python port of the original ``stegcompare.c``.  Given an original and a
(possibly steganography-modified) JPEG, it decodes the quantized DCT
coefficients of each, then reports, position by position, where they differ.

Two analysis modes are supported by the ``-t`` flag (the original only ever
honoured the *first* character of the argument — that quirk is preserved):

* ``-t p``  -> JPHide mode   (``prepare_jphide``)
* ``-t o``  -> OutGuess mode (``prepare_all``)

The default is JPHide mode, matching the C original's ``scans = FLAG_DOJPHIDE``.

``-h`` switches to a raw histogram dump of the two images' coefficient
distributions (mirrors the C ``-h`` / ``FLAG_HIST`` path).

Known quirk (inherited faithfully from the original): in JPHide mode
(``-t p``), the number of coefficients reported by ``prepare_jphide`` can differ
slightly between two images of the *same* dimensions, because that function's
revisit-detection is coefficient-value dependent. When the two counts differ,
stegcompare prints ``Size of images differs`` and stops — exactly as the C tool.
For a guaranteed like-for-like comparison of two arbitrary images, use
``-t o`` (OutGuess / all-coefficients mode), whose count is purely structural.

This module does NOT depend on the discrimination/extraction code; it is a
standalone coefficient-diff tool built directly on :mod:`stegdetect_py.common`.
"""

import argparse
import os
import sys

from . import common

VERSION = "0.1"

# Flag bits, copied verbatim from stegcompare.c.
FLAG_DOOUTGUESS = 0x0001
FLAG_DOJPHIDE = 0x0002
FLAG_HIST = 0x1000

# Coefficient range that the histogram tracks: [-127, 127] -> index [1, 255].
_HMIN = -127
_HMAX = 127
_HBINS = 257  # indices 0..256 -> coefficient value -128..128


def _prepare(scans, jpg):
    """Return ``(dcts, bits)`` for ``jpg`` in the selected mode."""
    if scans & FLAG_DOJPHIDE:
        return common.prepare_jphide(jpg)
    return common.prepare_all(jpg)


def docompare(file1, file2, scans=FLAG_DOJPHIDE, out=None):
    """Compare the DCT coefficients of ``file1`` and ``file2``.

    Parameters mirror the C ``docompare``: ``scans`` selects the prepare mode
    (JPHide vs all-coefficients) and whether to emit a raw histogram (``-h``).
    Output is written to ``out`` (default ``sys.stdout``).  Returns ``True`` if
    the two images had comparable coefficient counts, ``False`` otherwise.
    """
    if out is None:
        out = sys.stdout

    # The C code keeps these as file-scope ``int[257]`` arrays; we keep them
    # local so repeated calls do not contaminate one another.
    hist = [0] * _HBINS
    hist1 = [0] * _HBINS
    hist2 = [0] * _HBINS
    shist = [0] * _HBINS
    shist1 = [0] * _HBINS
    shist2 = [0] * _HBINS

    jpg1 = common.load_image(file1)
    dcts1, bits1 = _prepare(scans, jpg1)
    jpg2 = common.load_image(file2)
    dcts2, bits2 = _prepare(scans, jpg2)

    if bits1 != bits2:
        sys.stderr.write(
            "Size of images differs: %d != %d\n" % (bits1, bits2)
        )
        return False

    out.write("Size: %d\n" % bits1)

    last = -1
    sumlast = 0
    count = 0
    one = 0
    ratio = 0.5

    for i in range(bits1):
        v1 = dcts1[i]
        v2 = dcts2[i]

        if _HMIN <= v1 <= _HMAX:
            hist1[v1 + 128] += 1
        if _HMIN <= v2 <= _HMAX:
            hist2[v2 + 128] += 1

        if abs(v2) & 1:
            one += 1

        if (i % 500) == 0 and i:
            ratio = 0.7 * ratio + 0.3 * ((float(one) / 500))
            one = 0

        if v1 != v2:
            count += 1

            if scans & FLAG_DOJPHIDE:
                first = v1 if v1 > 0 else -v1
                second = v2 if v2 > 0 else -v2
            else:
                first = v1
                second = v2

            sumlast += (i - last) if last >= 0 else 0

            if (scans & FLAG_HIST) == 0:
                gap = (i - last) if last >= 0 else 0
                out.write(
                    "% 9d: % 5d != % 5d: %x | % 5d : %7.2f| %4.2f\n"
                    % (i, v1, v2, first ^ second, gap,
                       (float(sumlast) / count), ratio)
                )

            last = i

            if _HMIN <= v1 <= _HMAX:
                hist[v1 + 128] += 1

            if scans & FLAG_DOJPHIDE:
                shist[:] = hist
                shist1[:] = hist1
                shist2[:] = hist2

    if scans & FLAG_DOJPHIDE:
        hist[:] = shist
        hist1[:] = shist1
        hist2[:] = shist2

    if scans & FLAG_HIST:
        for i in range(_HBINS):
            out.write("% 4d % 5d % 5d\n"
                      % (i - 128, hist1[i], hist2[i]))
        return True

    out.write("Total changes: %d bits, %d bytes\n" % (count, count // 8))

    # Sort the bin indices by descending change count (C qsort + compare()).
    ordered = sorted(range(_HBINS), key=lambda k: hist[k], reverse=True)

    out.write("Histogram of changes to coefficients:\n")
    for i in ordered:
        if hist[i]:
            _print_hist_line(out, i, hist, hist1, hist2, count)

    out.write("Histogram of changes to coefficients (in order):\n")
    for i in range(_HBINS):
        if hist[i]:
            _print_hist_line(out, i, hist, hist1, hist2, count)

    return True


def _print_hist_line(out, i, hist, hist1, hist2, count):
    coef = i - 128
    h = hist[i]
    h1 = hist1[i]
    h2 = hist2[i]
    frac = (float(h) / h1) if h1 else 0.0
    pctcount = (float(h) / count) * 100
    diffpct = ((float(h2 - h1) / h1) * 100) if h1 else 0.0
    out.write(
        "% 4d: % 5d(%.3f) | %6.3f%% || % 7d | % 7d : %7.3f%%\n"
        % (coef, h, frac, pctcount, h1, h2, diffpct)
    )


def _usage(progname, out=sys.stderr):
    out.write("Usage: %s orig.jpg modified.jpg\n" % progname)


def main(argv=None):
    if argv is None:
        argv = sys.argv

    progname = argv[0] if argv else "stegcompare"

    parser = argparse.ArgumentParser(
        prog=progname, add_help=False,
        description="Compare DCT coefficients of two JPEG images.",
    )
    parser.add_argument("-V", dest="version", action="store_true",
                        help="print version and exit")
    parser.add_argument("-h", dest="hist", action="store_true",
                        help="dump the raw coefficient histogram")
    parser.add_argument("-t", dest="scheme", default=None,
                        help="scheme: o=outguess (all coeffs), p=jphide")
    # ``nargs='*'`` so that ``-V`` / ``-h`` work without two file args; the
    # exact-2-files requirement is enforced below (C: ``if (argc != 2)``).
    parser.add_argument("files", nargs="*", metavar="FILE")

    # argparse would exit(2) on parse failure; the C exits(1) via usage().
    try:
        args = parser.parse_args(argv[1:])
    except SystemExit:
        _usage(progname)
        return 1

    if args.version:
        sys.stdout.write("Stegcompare Version %s\n" % VERSION)
        return 1

    if len(args.files) != 2:
        _usage(progname)
        return 1

    scans = FLAG_DOJPHIDE
    if args.hist:
        scans |= FLAG_HIST
    if args.scheme is not None:
        scans &= ~(FLAG_DOOUTGUESS | FLAG_DOJPHIDE)
        # Faithful to the C: the loop stops at the first recognised char
        # (``i < strlen(optarg) && !scans``), so only the first char counts.
        for ch in args.scheme:
            if ch == "o":
                scans |= FLAG_DOOUTGUESS
                break
            elif ch == "p":
                scans |= FLAG_DOJPHIDE
                break
            else:
                _usage(progname)
                return 1

    docompare(args.files[0], args.files[1], scans)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (BrokenPipeError, OSError) as exc:
        # Downstream closed the pipe early (e.g. `... | head`); the original C
        # tool dies on SIGPIPE here.  Redirect to devnull and exit quietly so we
        # don't emit an ugly traceback.
        if isinstance(exc, OSError) and exc.errno not in (None, 22, 32):
            raise
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
        except Exception:  # noqa: BLE001
            pass
        sys.exit(0)
