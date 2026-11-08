import io
import os

import numpy as np
from PIL import Image

from stegdetect_py import common, stegcompare
from stegdetect_py.stegcompare import (
    FLAG_DOJPHIDE,
    FLAG_DOOUTGUESS,
    FLAG_HIST,
)


def _make_jpg(path, seed=5, size=160, quality=95, pixels=None):
    rng = np.random.RandomState(seed)
    im = Image.fromarray(
        rng.randint(0, 256, (size, size, 3), dtype=np.uint8), "RGB"
    )
    if pixels is not None:
        arr = np.asarray(im).copy()
        for (x, y, c, val) in pixels:
            arr[y, x, c] = val
        im = Image.fromarray(arr, "RGB")
    im.save(path, format="JPEG", quality=quality, subsampling=0)


def test_identical_images_no_changes(tmp_path):
    p1 = str(tmp_path / "a.jpg")
    p2 = str(tmp_path / "b.jpg")
    _make_jpg(p1, seed=1)
    _make_jpg(p2, seed=1)  # same RNG -> same pixels -> same file

    out = io.StringIO()
    ok = stegcompare.docompare(p1, p2, FLAG_DOJPHIDE, out=out)
    text = out.getvalue()
    assert ok is True
    assert "Size:" in text
    # Identical files -> zero changed coefficients.
    assert "Total changes: 0 bits, 0 bytes" in text
    assert "Histogram of changes to coefficients:" in text


def test_modified_image_reports_changes(tmp_path):
    p1 = str(tmp_path / "a.jpg")
    p2 = str(tmp_path / "b.jpg")
    # original
    _make_jpg(p1, seed=2)
    # a few pixel edits -> block count stays equal, some coeffs change.
    # Use OutGuess (all-coefficients) mode: prepare_all yields a structural,
    # image-content-independent coefficient count, so the comparison is stable.
    _make_jpg(p2, seed=2, pixels=[(10, 10, 0, 200), (40, 30, 1, 0), (80, 70, 2, 255)])

    out = io.StringIO()
    ok = stegcompare.docompare(p1, p2, FLAG_DOOUTGUESS, out=out)
    text = out.getvalue()
    assert ok is True
    # The two images must still share the same coefficient count.
    assert "Size:" in text
    assert "Size of images differs" not in text
    # Pixel edits must surface as changed coefficients.
    assert "Total changes:" in text
    assert "bits, " in text


def test_hist_flag_raw_dump(tmp_path):
    p1 = str(tmp_path / "a.jpg")
    p2 = str(tmp_path / "b.jpg")
    _make_jpg(p1, seed=3)
    _make_jpg(p2, seed=3)

    out = io.StringIO()
    ok = stegcompare.docompare(p1, p2, FLAG_DOJPHIDE | FLAG_HIST, out=out)
    text = out.getvalue()
    assert ok is True
    # Raw histogram mode prints "<coef> <hist1> <hist2>" lines, no totals.
    assert "Total changes:" not in text
    # Each line: a coefficient value (e.g. -128) followed by two counts.
    assert "-128" in text


def test_outguess_mode_runs(tmp_path):
    p1 = str(tmp_path / "a.jpg")
    p2 = str(tmp_path / "b.jpg")
    _make_jpg(p1, seed=4)
    _make_jpg(p2, seed=4, pixels=[(5, 5, 0, 100)])

    out = io.StringIO()
    ok = stegcompare.docompare(p1, p2, FLAG_DOOUTGUESS, out=out)
    assert ok is True
    assert "Total changes:" in out.getvalue()


def test_version_exit_code(capsys):
    rc = stegcompare.main(["stegcompare", "-V"])
    assert rc == 1
    assert "Stegcompare Version" in capsys.readouterr().out


def test_missing_args_usage(capsys):
    rc = stegcompare.main(["stegcompare"])
    # C usage() -> exit(1)
    assert rc == 1


def test_different_size_reports_difference(tmp_path, capsys):
    p1 = str(tmp_path / "a.jpg")
    p2 = str(tmp_path / "b.jpg")
    _make_jpg(p1, seed=6, size=160)
    _make_jpg(p2, seed=6, size=200)  # different dimensions -> bits differ
    out = io.StringIO()
    ok = stegcompare.docompare(p1, p2, FLAG_DOJPHIDE, out=out)
    assert ok is False
    # The C writes this via warnx() -> stderr.
    assert "Size of images differs" in capsys.readouterr().err
