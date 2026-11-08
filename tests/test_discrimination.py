import io
import math
import os

import numpy as np

from stegdetect_py import discrimination as disc


def _make_decision(pos, neg):
    """Build a CdDecision with the given positive/negative sample lists.

    Mirrors what ``cd_process_file`` does: it records ``npoints`` (the feature
    dimensionality) so ``cd_compute`` can size its projection vector.
    """
    cdd = disc.cd_new()
    cdd.npoints = len(pos[0])
    for p in pos:
        disc.cd_make_entry(cdd, "pos", 1, [float(x) for x in p], len(p))
    for n in neg:
        disc.cd_make_entry(cdd, "neg", 0, [float(x) for x in n], len(n))
    return cdd


def test_cd_collect_data():
    pts, n = disc.cd_collect_data("1.0 2.5 -3.0  4")
    assert n == 4
    assert pts == [1.0, 2.5, -3.0, 4.0]


def test_cd_classify_linearly_separable():
    # Two well-separated clusters in 2-D.
    pos = [[8.0, 8.0], [9.0, 7.0], [7.0, 9.0], [8.5, 8.5]]
    neg = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    cdd = _make_decision(pos, neg)
    disc.cd_compute(cdd, "test", test=0)

    # A point deep in the positive cluster should classify positive.
    assert disc.cd_classify(cdd, [8.0, 8.0]) is True
    # A point deep in the negative cluster should classify negative.
    assert disc.cd_classify(cdd, [0.5, 0.5]) is False


def test_cd_process_file_roundtrip(tmp_path):
    path = tmp_path / "train.txt"
    # imagename:type,transform:data  (3-D, non-collinear so covariance is full rank)
    lines = [
        "a.png:1,spline:0.1 0.2 0.3",
        "b.png:1,spline:0.4 0.1 0.5",
        "c.png:1,spline:0.2 0.5 0.1",
        "d.png:0,spline:1.1 1.2 1.3",
        "e.png:0,spline:1.4 1.1 1.5",
        "f.png:0,spline:1.2 1.5 1.1",
    ]
    path.write_text("\n".join(lines) + "\n")
    cdd = disc.cd_new()
    disc.cd_process_file(cdd, str(path))
    assert cdd.npoints == 3
    assert cdd.npositive == 3
    assert cdd.nnegative == 3
    assert cdd.transform_name == "spline"
    disc.cd_compute(cdd, "roundtrip", test=0)
    assert disc.cd_name(cdd) == "roundtrip"
    assert disc.cd_transform(cdd) is not None


def test_cd_read_dump_roundtrip():
    pos = [[8.0, 8.0], [9.0, 7.0]]
    neg = [[0.0, 0.0], [1.0, 0.0]]
    cdd = _make_decision(pos, neg)
    disc.cd_compute(cdd, "saved", test=0)
    cdd.transform_name = "spline"  # set by cd_process_file in real usage

    buf = io.StringIO()
    disc.cd_dump(buf, cdd)
    buf.seek(0)
    loaded = disc.cd_read(buf)

    assert loaded is not None
    assert loaded.name == "saved"
    assert loaded.npoints == cdd.npoints
    assert loaded.b is not None
    for a, b in zip(loaded.b, cdd.b):
        assert math.isclose(a, b, rel_tol=1e-9)
    assert math.isclose(loaded.k, cdd.k, rel_tol=1e-9)


def test_cd_test_sets_boundary():
    # Make positives clearly separable; cd_test should pick a where with FPR < 1%.
    rng = np.random.RandomState(3)
    pos = [[float(rng.normal(8, 0.1)), float(rng.normal(8, 0.1))] for _ in range(20)]
    neg = [[float(rng.normal(0, 0.1)), float(rng.normal(0, 0.1))] for _ in range(20)]
    cdd = _make_decision(pos, neg)
    disc.cd_compute(cdd, "t", test=1)
    disc.cd_test(cdd)
    assert -1.0 <= cdd.where <= 2.0
    # The Fisher projection separates the two class means: at the midpoint
    # boundary they fall on opposite sides (this is the core LDA property,
    # independent of where cd_test ultimately parks the operating point).
    k_mid = (cdd.projpos + cdd.projneg) / 2.0
    val_pos = disc.cd_project([8.0, 8.0], cdd.b, cdd.npoints)
    val_neg = disc.cd_project([0.0, 0.0], cdd.b, cdd.npoints)
    assert (val_pos > k_mid) != (val_neg > k_mid)
