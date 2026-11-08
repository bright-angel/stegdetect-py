"""
Validation of the stegdetect CLI (stegdetect_py.cli).

Runs without pytest (use:  PYTHONPATH=. python tests/test_cli.py).

Covers:
  * flag parsing (-t / -s / -q / -V / -n)
  * clean image -> "negative"
  * in-memory JSteg embedding -> " jsteg(...)"
  * appended data after EOI -> " appended(...)"
  * F5 fast-path and Invisible-Secrets comment signatures
"""

import io
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
from PIL import Image
import pytest

from stegdetect_py import cli, common
from stegdetect_py.jpeg import decode_coefficients


@pytest.fixture
def tmp():
    """Temporary directory (path string) used by the CLI integration tests."""
    d = tempfile.mkdtemp(prefix="stegdetect_test_")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _make_jpg(size, seed, quality=85, subsampling=0):
    rng = np.random.RandomState(seed)
    im = Image.fromarray(rng.randint(0, 256, (size, size, 3), dtype=np.uint8), "RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, subsampling=subsampling)
    return buf.getvalue()


def _embed_jsteg(jpg, msg_coeffs=1500, seed=7):
    rng = np.random.RandomState(seed)
    bits = []
    width = 8
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
                if (v & 1) == v:
                    continue
                if idx < len(bits):
                    blk[s] = (v & ~1) | bits[idx]
                idx += 1


def test_clean_is_negative(tmp):
    p = os.path.join(tmp, "clean.jpg")
    with open(p, "wb") as f:
        f.write(_make_jpg(512, 4))
    line = cli.detect(p, cli.FLAG_DOOUTGUESS | cli.FLAG_DOJPHIDE |
                      cli.FLAG_DOJSTEG | cli.FLAG_DOINVIS | cli.FLAG_DOF5 |
                      cli.FLAG_DOAPPEND)
    print("clean:", line)
    assert "negative" in line, line
    assert "jsteg" not in line


def test_jsteg_detected_in_memory():
    raw = _make_jpg(512, 4)
    jpg = decode_coefficients(raw)
    _embed_jsteg(jpg)
    line = cli.detect_jpg(jpg, raw, "mem.jpg",
                          cli.FLAG_DOJSTEG | cli.FLAG_DOOUTGUESS | cli.FLAG_DOJPHIDE)
    print("jsteg-embedded:", line)
    assert "jsteg" in line, line


def test_appended_data(tmp):
    p = os.path.join(tmp, "app.jpg")
    with open(p, "wb") as f:
        f.write(_make_jpg(64, 2) + b"HIDDEN-PAYLOAD-AFTER-EOI")
    line = cli.detect(p, cli.FLAG_DOAPPEND)
    print("appended:", line)
    assert "appended" in line, line


def test_f5_fast_signature():
    sig = b"JPEG Encoder Copyright 1998, James R. Weeks and BioElectroMech."
    common.comments = [sig]
    common.commentsize = [len(sig)]
    common.ncomments = 1
    assert cli.check_f5_fast() is True
    common.comments = []
    common.commentsize = []
    common.ncomments = 0
    assert cli.check_f5_fast() is False


def test_invisible_signature():
    payload = b"\x04\x00\x00\x00" + b"ABCD"  # length 4, then 4 bytes
    common.comments = [b"ignored", payload]
    common.commentsize = [7, len(payload)]
    common.ncomments = 2
    assert cli.check_invisible() == 4
    common.comments = []
    common.commentsize = []
    common.ncomments = 0
    assert cli.check_invisible() is None


def test_flag_parsing_version():
    buf = StringIO()
    try:
        with redirect_stdout(buf):
            cli.main(["-V"])
    except SystemExit:
        pass
    out = buf.getvalue()
    print("version output:", out.strip())
    assert "Stegdetect Version" in out


def test_t_flag_selects_jsteg_only(tmp):
    p = os.path.join(tmp, "clean.jpg")
    with open(p, "wb") as f:
        f.write(_make_jpg(512, 4))
    buf = StringIO()
    with redirect_stdout(buf):
        cli.main(["-t", "j", p])
    out = buf.getvalue()
    print("t=j output:", out.strip())
    assert "negative" in out


if __name__ == "__main__":
    tmp = tempfile.mkdtemp(prefix="stegdetect_test_")
    test_clean_is_negative(tmp)
    test_jsteg_detected_in_memory()
    test_appended_data(tmp)
    test_f5_fast_signature()
    test_invisible_signature()
    test_flag_parsing_version()
    test_t_flag_selects_jsteg_only(tmp)
    print("\nALL CLI TESTS OK")
