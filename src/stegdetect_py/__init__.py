"""Python port of Niels Provos' stegdetect (JPEG steganography detector).

This package is a faithful re-implementation of the original C stegdetect /
stegbreak tools.  The detection algorithms operate on the *quantized* DCT
coefficients of a JPEG image; see :mod:`stegdetect_py.jpeg` for the
pure-Python coefficient decoder that replaces libjpeg's
``jpeg_read_coefficients``.

Quick start
-----------
    from stegdetect_py import cli
    cli.main(["image.jpg"])                 # run the detector on a file

    from stegdetect_py import common, detection
    jpg = common.load_image("image.jpg")
    dcts, bits = common.prepare_jsteg(jpg)
    print(detection.histogram_chi_jsteg(dcts, bits))
"""

__version__ = "0.1.0"

from .jpeg import (
    decode_coefficients,
    JPEG,
    Component,
    JPEGError,
    JPEGNotSupported,
)
from . import common, detection, chi2, dct, cli

__all__ = [
    "decode_coefficients",
    "JPEG",
    "Component",
    "JPEGError",
    "JPEGNotSupported",
    "common",
    "detection",
    "chi2",
    "dct",
    "cli",
    "__version__",
]
