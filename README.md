# stegdetect-py

A faithful Python port of [Niels Provos' `stegdetect` / `stegbreak`](https://github.com/abeluck/stegdetect)
JPEG steganography detector and cracker. The original tools detect and attack
messages hidden in JPEG images with **JSteg**, **OutGuess**, **JPHide**, and
**F5**, and they let you dictionary/crack passwords protecting those payloads.

This project re-implements the original C source module-by-module (rules engine,
INI config parser, rule preprocessor, crack database, calibration feature
extraction, Fisher discriminant, F5 slow-path detection, and the DCT-based
crackers) on top of a pure-Python JPEG quantized-DCT decoder, so it runs
anywhere Python + NumPy + Pillow are available — no C compiler, no libjpeg.

## Features

| Tool | Purpose | Status |
|------|---------|--------|
| `stegdetect` | Detect JPEG stego (jsteg / outguess / jphide / f5) via χ² statistics | ✅ |
| `stegbreak` | Dictionary/crack passwords for embedded payloads | ✅ |
| `stegcompare` | Compare DCT coefficients of two JPEGs (cover vs. stego) | ✅ |

## Installation

```bash
# from the project root (src-layout package)
pip install -e .

# or, to run the tests as well
pip install -e ".[test]"
```

Runtime dependencies: **Python ≥ 3.9**, **NumPy ≥ 1.21**, **Pillow ≥ 9.0**.

## Command-line usage

### Detect steganography

```bash
stegdetect -q photo.jpg          # quiet mode
stegdetect -s photo.jpg          # also print the detected scheme
stegdetect -t f photo.jpg        # only test the F5 scheme
```

### Crack an embedded password

```bash
stegbreak -t j -f wordlist.txt secret.jpg     # crack jsteg
stegbreak -t opj -f words.txt stego.jpg       # try outguess + jphide + jsteg
stegbreak -V                                   # version
```

Default rules live in `stegdetect_py/stegbreak.conf` (the `[List.Rules:Wordlist]`
section); override with `-r rules.conf`.

### Compare two JPEGs

```bash
stegcompare -t o original.jpg modified.jpg     # outguess mode (all coeffs)
stegcompare original.jpg modified.jpg          # default jphide mode
stegcompare -h original.jpg modified.jpg       # raw coefficient histogram
```

> **Tip:** For a *stable* coefficient comparison of two arbitrary images, use
> `-t o` (OutGuess / all-coefficients mode). The default JPHide mode counts
> coefficients in a way that can depend on pixel content, so two differing
> images may report *"Size of images differs"* and stop early — that mirrors the
> original C behavior.

## Python API

```python
from stegdetect_py import common, detection

jpg = common.load_image("photo.jpg")
dcts, bits = common.prepare_jsteg(jpg)
print(detection.histogram_chi_jsteg(dcts, bits))

# Crack via the library
from stegdetect_py import stegbreak
# stegbreak.main(["-t", "j", "-f", "words.txt", "secret.jpg"])
```

## Project layout

```
stegdetect_python/
├── pyproject.toml          # build + tool config (setuptools, pytest)
├── README.md
├── LICENSE                 # BSD-3-Clause
├── MANIFEST.in
├── .gitignore
├── src/
│   └── stegdetect_py/      # the importable package
│       ├── __init__.py
│       ├── __main__.py     # `python -m stegdetect_py`
│       ├── jpeg.py         # pure-Python quantized-DCT decoder
│       ├── common.py       # DCT prep helpers (prepare_*/load_image)
│       ├── cli.py          # stegdetect entry point
│       ├── detection.py    # χ² detection
│       ├── chi2.py         # χ² CDF
│       ├── break_jsteg.py / break_outguess.py / break_jphide.py
│       ├── rules.py        # John-the-Ripper rule engine
│       ├── rpp.py          # rule preprocessor (range expansion)
│       ├── cfg.py          # INI config parser
│       ├── db.py           # crack database
│       ├── stegbreak.py    # stegbreak entry point + default rules loader
│       ├── stegbreak.conf  # default [List.Rules:Wordlist]
│       ├── extraction.py   # calibration feature extraction
│       ├── discrimination.py # Fisher linear discriminant
│       ├── f5.py           # F5 slow-path detection
│       ├── arc4.py / blowfish.py / md5.py  # ciphers (port of C)
│       ├── mathx.py / util.py              # matrix invert / fgetl helpers
│       ├── dct.py          # IDCT / dequant
│       └── py.typed        # PEP 561 marker
└── tests/                  # pytest suite (77 tests, all passing)
```

## Known limitations

- **No JPEG encoder.** `jpeg.py` is decode-only. Embedded carriers used by the
  test-suite are built *in-memory* from DCT coefficients; there is no way to
  re-emit a stego JPEG from this library.
- **`f5.py` recompression is approximate.** The original C re-encodes using the
  image's own quantization tables via libjpeg. This pure-Python port approximates
  that with Pillow's `qtables` option; results are close but not bit-identical
  to the C tool.
- **JPHide is JPEG-quality sensitive.** Embedding/cracking JPHide works reliably
  at `quality=95`; lower qualities can break the crack (a property inherited
  from the original algorithm).
- **X11 tools not ported.** `xsteg`, `xsteg_xpm`, and `stegdeimage` (the
  original graphical front-ends) are display-only and out of scope for a
  headless port.

## License

BSD-3-Clause. See [LICENSE](LICENSE). The original `stegdetect` is
Copyright (c) 1999–2002 Niels Provos.
