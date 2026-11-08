"""Faithful Python port of stegdetect's ``discrimination.c``.

This implements the *calibration-based* Fisher linear discriminant that
stegdetect's ``stegcompare``/``xsteg`` tools use to separate stego from cover
images given a set of pre-computed feature vectors (the transforms live in
:mod:`stegdetect_py.extraction`).

The original C relies on the BSD ``<sys/queue.h>`` TAILQ macros.  Those are
plain doubly-linked lists, so we represent each queue as a Python ``list`` and
preserve insertion order.  The public function names mirror the C API
(``cd_init``, ``cd_new``, ``cd_process_file``, ``cd_compute``, ``cd_classify``,
``cd_test``, ``cd_read``, ``cd_dump``, ...) so behaviour matches the reference
binary.

One faithful detail worth calling out: in ``cd_covarest`` the original source
accumulates *both* the positive and negative deviation sums into ``sum1`` while
``sum2`` stays zero, so the pooled covariance is ``sum1 / (count1 + count2 -
2)``.  We reproduce that computation exactly (including the dead ``sum2``) so
numerical output matches.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from . import extraction
from .mathx import matrix_invert
from .util import fgetl

CD_PERCENT = 0.8
MAX_NPOINTS = 1024


class CdEntry:
    """A single labelled feature vector (mirrors ``struct cd_entry``)."""

    __slots__ = ("filename", "positive", "npoints", "points")

    def __init__(self, filename: str, positive: int, points: List[float],
                 npoints: int):
        self.filename = filename
        self.positive = positive
        self.points = points
        self.npoints = npoints


class CdDecision:
    """One Fisher discriminant (mirrors ``struct cd_decision``)."""

    __slots__ = (
        "name", "npoints", "b", "k", "where", "projpos", "projneg",
        "positiveq", "npositive", "negativeq", "nnegative",
        "transform_name", "transform",
    )

    def __init__(self):
        self.name: Optional[str] = None
        self.npoints: int = 0
        self.b: Optional[List[float]] = None
        self.k: float = 0.0
        self.where: float = 0.5
        self.projpos: float = 0.0
        self.projneg: float = 0.0
        self.positiveq: List[CdEntry] = []
        self.npositive: int = 0
        self.negativeq: List[CdEntry] = []
        self.nnegative: int = 0
        self.transform_name: Optional[str] = None
        self.transform = None


# Module-level queue of decisions (mirrors the global TAILQ ``cdq``).
_cdq: List[CdDecision] = []


def cd_init() -> None:
    """Re-initialise the global decision queue."""
    global _cdq
    _cdq = []


def cd_insert(cdd: CdDecision) -> None:
    _cdq.append(cdd)


def cd_iterate(cdd: Optional[CdDecision]) -> Optional[CdDecision]:
    if cdd is None:
        return _cdq[0] if _cdq else None
    idx = _cdq.index(cdd)
    return _cdq[idx + 1] if idx + 1 < len(_cdq) else None


def cd_new() -> CdDecision:
    return CdDecision()


def cd_collect_data(data: str) -> Tuple[List[float], int]:
    """Parse a space-separated run of doubles (mirrors ``cd_collect_data``).

    Returns ``(points, npoints)`` or raises ``ValueError`` on malformed input.
    """
    points: List[float] = []
    for tok in data.split(" "):
        if tok == "":
            continue
        try:
            points.append(float(tok))
        except ValueError:
            raise ValueError("cd_collect_data: bad data token %r" % tok)
        if len(points) >= MAX_NPOINTS:
            break
    return points, len(points)


def cd_make_entry(cdd: CdDecision, name: str, positive: int,
                  points: List[float], npoints: int) -> None:
    entry = CdEntry(name, positive, points, npoints)
    if positive:
        cdd.positiveq.append(entry)
        cdd.npositive += 1
    else:
        cdd.negativeq.append(entry)
        cdd.nnegative += 1


def cd_process_file(cdd: CdDecision, filename: str) -> int:
    """Parse a training file (mirrors ``cd_process_file``).

    Each line has the form ``imagename:type,transform:data`` where ``type`` is
    ``1`` for positive and ``0`` for negative, ``transform`` names the feature
    transform, and ``data`` is the space-separated feature vector.
    """
    npoints = cdd.npoints
    with open(filename, "r") as fin:
        for linenr, raw in enumerate(fin, 1):
            line = raw.rstrip("\n").rstrip("\r")
            if line == "":
                continue
            parts = line.split(":")
            if len(parts) < 3:
                raise ValueError('%s: bad format in line %d' % (filename, linenr))
            imagename, tmp, data = parts[0], parts[1], ":".join(parts[2:])
            subt = tmp.split(",", 1)
            if len(subt) < 2:
                raise ValueError('%s: bad format in line %d' % (filename, linenr))
            type_, tfname = subt[0], subt[1]

            if cdd.transform_name is not None:
                if cdd.transform_name != tfname:
                    raise ValueError(
                        '%s: transform name changed in line %d' % (filename, linenr))
            else:
                cdd.transform = extraction.transform_lookup(tfname)
                if cdd.transform is None:
                    raise ValueError(
                        '%s: unknown transform in line %d' % (filename, linenr))
                cdd.transform_name = tfname

            points, tmpnpoints = cd_collect_data(data)
            if npoints and tmpnpoints != npoints:
                raise ValueError(
                    '%s: require %d data points but got %d in line %d'
                    % (filename, npoints, tmpnpoints, linenr))
            if not npoints:
                npoints = tmpnpoints

            cd_make_entry(cdd, imagename, int(type_), points, npoints)

    if not cdd.npoints:
        cdd.npoints = npoints
    return 0


def cd_meanest(head: List[CdEntry], howmany: int, npoints: int) -> List[float]:
    """Per-dimension mean of the entries in ``head`` (mirrors ``cd_meanest``)."""
    mest = [0.0] * npoints
    for i in range(npoints):
        total = 0.0
        count = 0
        for entry in head:
            total += entry.points[i]
            count += 1
            if howmany and count >= howmany:
                break
        mest[i] = total / count
    return mest


def cd_covarest(cdd: CdDecision, npositive: int, nnegative: int,
                mestpos: List[float], mestneg: List[float]) -> List[List[float]]:
    """Pooled covariance estimate (mirrors ``cd_covarest``).

    Faithful to the original: both the positive and negative deviation products
    are accumulated into ``sum1`` while ``sum2`` remains zero, so the result is
    ``sum1 / (count1 + count2 - 2)``.
    """
    npoints = cdd.npoints
    covarest: List[List[float]] = [[0.0] * npoints for _ in range(npoints)]
    for i in range(npoints):
        for j in range(npoints):
            sum1 = 0.0
            count1 = 0
            for entry in cdd.positiveq:
                sum1 += (entry.points[i] - mestpos[i]) * (entry.points[j] - mestpos[j])
                count1 += 1
                if npositive and count1 >= npositive:
                    break
            sum2 = 0.0
            count2 = 0
            for entry in cdd.negativeq:
                sum1 += (entry.points[i] - mestneg[i]) * (entry.points[j] - mestneg[j])
                count2 += 1
                if nnegative and count2 >= nnegative:
                    break
            covarest[i][j] = (sum1 + sum2) / (count1 + count2 - 2)
    return covarest


def cd_project(a: List[float], b: List[float], npoints: int) -> float:
    """Dot product of two ``npoints``-length vectors (mirrors ``cd_project``)."""
    return sum(a[i] * b[i] for i in range(npoints))


def cd_dump(fout, cdd: CdDecision) -> None:
    fout.write("%s\n" % cdd.name)
    fout.write("%s\n" % cdd.transform_name)
    fout.write("%d" % cdd.npoints)
    for v in cdd.b:
        fout.write(" %f" % v)
    fout.write("\n")
    fout.write("%f %f %f\n" % (cdd.projpos, cdd.projneg, cdd.k))


def cd_read(fin) -> Optional[CdDecision]:
    """Deserialise a decision written by :func:`cd_dump` (mirrors ``cd_read``)."""
    cdd = CdDecision()
    name = fgetl(fin)
    if name is None:
        return None
    cdd.name = name
    tfname = fgetl(fin)
    if tfname is None:
        return None
    cdd.transform_name = tfname
    cdd.transform = extraction.transform_lookup(tfname)
    if cdd.transform is None:
        print("Unknown transforms \"%s\" for \"%s\"" % (tfname, name))
        return None

    line = fgetl(fin)
    if line is None:
        return None
    parts = line.split(" ", 1)
    cdd.npoints = int(parts[0])
    rest = parts[1] if len(parts) > 1 else ""
    try:
        cdd.b, n = cd_collect_data(rest)
    except ValueError:
        print("Format %s: projection data malformed" % name)
        return None
    if cdd.b is None or cdd.npoints != n:
        print("Format %s: projection data malformed" % name)
        return None

    line = fgetl(fin)
    if line is None:
        return None
    vals = line.split()
    if len(vals) != 3:
        print("Format %s: boundary values malformed" % name)
    else:
        cdd.projpos, cdd.projneg, cdd.k = (float(vals[0]), float(vals[1]),
                                           float(vals[2]))
    print("Read detection for %s" % name)
    return cdd


def cd_compute(cdd: CdDecision, name: str, test: int) -> None:
    """Train the Fisher discriminant (mirrors ``cd_compute``)."""
    if cdd.nnegative < 2 or cdd.npositive < 2:
        raise ValueError("Not enough data points for \"%s\"" % name)

    cdd.name = name
    cdd.b = [0.0] * cdd.npoints

    if test:
        npositive = int(cdd.npositive * CD_PERCENT)
        nnegative = int(cdd.nnegative * CD_PERCENT)
    else:
        npositive = cdd.npositive
        nnegative = cdd.nnegative

    mestpos = cd_meanest(cdd.positiveq, npositive, cdd.npoints)
    mestneg = cd_meanest(cdd.negativeq, nnegative, cdd.npoints)

    covarest = cd_covarest(cdd, npositive, nnegative, mestpos, mestneg)
    matrix_invert(covarest, cdd.npoints)

    for i in range(cdd.npoints):
        total = 0.0
        for j in range(cdd.npoints):
            total += covarest[i][j] * (mestpos[j] - mestneg[j])
        cdd.b[i] = total

    cdd.projpos = cd_project(mestpos, cdd.b, cdd.npoints)
    cdd.projneg = cd_project(mestneg, cdd.b, cdd.npoints)
    cdd.k = (cdd.projpos + cdd.projneg) / 2

    cd_setboundary(cdd, cdd.where)


def cd_setboundary(cdd: CdDecision, where: float) -> None:
    """Set the decision boundary at fraction ``where`` (mirrors ``cd_setboundary``)."""
    if cdd.projpos > cdd.projneg:
        k = (cdd.projpos - cdd.projneg) * where + cdd.projneg
    else:
        k = (cdd.projneg - cdd.projpos) * (1 - where) + cdd.projpos
    cdd.k = k


def cd_classify(cdd: CdDecision, points: List[float]) -> bool:
    """Classify a feature vector; returns ``True`` for the positive class."""
    val = cd_project(points, cdd.b, cdd.npoints)
    if cdd.projpos > cdd.projneg:
        return val > cdd.k
    else:
        return val < cdd.k


def cd_test(cdd: CdDecision) -> None:
    """Find a boundary ``where`` keeping the false-positive rate below 1%."""
    saved = False
    for where in [x / 100.0 for x in range(-100, 201, 15)]:
        negcorrect = negfalse = 0
        poscorrect = posfalse = 0

        cd_setboundary(cdd, where)

        npositive = int(cdd.npositive * CD_PERCENT)
        nnegative = int(cdd.nnegative * CD_PERCENT)

        count = 0
        for entry in cdd.negativeq:
            count += 1
            if count <= nnegative:
                continue
            if not cd_classify(cdd, entry.points):
                negcorrect += 1
            else:
                negfalse += 1

        count = 0
        for entry in cdd.positiveq:
            count += 1
            if count <= npositive:
                continue
            if cd_classify(cdd, entry.points):
                poscorrect += 1
            else:
                posfalse += 1

        fprate = negfalse / (negfalse + negcorrect) if (negfalse + negcorrect) else 0.0
        if not saved and fprate < 1:
            saved = True
            cdd.where = where

    if not saved:
        cdd.where = 0.5


def cd_name(cdd: CdDecision) -> Optional[str]:
    return cdd.name


def cd_transform(cdd: CdDecision):
    return cdd.transform
