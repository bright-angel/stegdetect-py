"""
Integration tests for the stegbreak subsystem: cfg parser, rpp rule expansion,
and the full wordlist-crack pipeline (db + rules + crackers).
"""

import io
import os
import numpy as np
from PIL import Image

from stegdetect_py import cfg, rpp, db, rules as rules_mod
from stegdetect_py import break_jsteg, break_outguess, break_jphide, common, arc4
from stegdetect_py.stegbreak import (
    do_wordlist_crack, jsteg_read_jpg, outguess_read_jpg, jphide_read_jpg,
    FLAG_DOJSTEG, FLAG_DOOUTGUESS, FLAG_DOJPHIDE,
)


# ---------------------------------------------------------------------------
# cfg / rpp unit tests
# ---------------------------------------------------------------------------
def test_cfg_parse():
    cfg.cfg_reset()
    cfg_text = """
[Section.One]
key = value
other = 123
[List.Rules:Wordlist]
l
u
c$1
"""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".conf", delete=False) as fh:
        fh.write(cfg_text)
        path = fh.name
    try:
        cfg.cfg_init(path)
        assert cfg.cfg_get_param("section.one", None, "key") == "value"
        assert cfg.cfg_get_param("section.one", None, "other") == "123"
        lst = cfg.cfg_get_list("list.rules", ":wordlist")
        assert lst is not None
        items = lst.items()
        assert items == ["l", "u", "c$1"]
    finally:
        os.unlink(path)


def test_rpp_range_expansion():
    cfg.cfg_reset()
    # Build a tiny config with a single list rule containing a range.
    import tempfile
    cfg_text = "[List.Rules:Wordlist]\nl[0-9]\n"
    with tempfile.NamedTemporaryFile("w", suffix=".conf", delete=False) as fh:
        fh.write(cfg_text)
        path = fh.name
    try:
        cfg.cfg_init(path)
        ctx = rpp.RppContext()
        assert rpp.rpp_init(ctx, rpp.SUBSECTION_WORDLIST) == 0
        rules = []
        while True:
            r = rpp.rpp_next(ctx)
            if r is None:
                break
            # rule "l[0-9]" produces l0..l9
            assert r[0] == "l"
            rules.append(r)
        assert len(rules) == 10
        assert "l0" in rules and "l9" in rules
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Helpers to build an in-memory carrier JPEG
# ---------------------------------------------------------------------------
def _make_jpg(seed=3, size=160):
    rng = np.random.RandomState(seed)
    im = Image.fromarray(
        rng.randint(0, 256, (size, size, 3), dtype=np.uint8), "RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=95, subsampling=0)
    from stegdetect_py.jpeg import decode_coefficients
    return decode_coefficients(buf.getvalue())


def _eligible_refs(jpg):
    refs = []
    for ci, comp in enumerate(jpg.components):
        for bi, blk in enumerate(comp.blocks):
            for si, val in enumerate(blk):
                if (val & 1) == val:
                    continue
                refs.append((ci, bi, si))
    return refs


def embed_jsteg_memory(jpg, plain, password):
    refs = _eligible_refs(jpg)
    n = len(plain)
    width = max(1, n.bit_length())
    header = [((width >> (4 - b)) & 1) for b in range(5)]
    header += [((n >> (width - 1 - b)) & 1) for b in range(width)]
    as_ = arc4.ARC4Stream()
    arc4.arc4_fixedkey(as_, password)
    ks = [arc4.arc4_getbyte(as_) for _ in range(n)]
    cipher = bytes(p ^ k for p, k in zip(plain, ks))
    bits = list(header)
    for byte in cipher:
        for b in range(8):
            bits.append((byte >> (7 - b)) & 1)
    for k, bit in enumerate(bits):
        ci, bi, si = refs[k]
        v = jpg.components[ci].blocks[bi][si]
        jpg.components[ci].blocks[bi][si] = (v & ~1) | bit
    return len(bits)


# ---------------------------------------------------------------------------
# full pipeline end-to-end
# ---------------------------------------------------------------------------
def _run_stegbreak(jpg, prepare_fn, crack_fn, flag, password_str, wordlist_words):
    import tempfile
    import importlib.resources
    # stegbreak.conf ships as package data next to the stegdetect_py package.
    cfg_path = str(importlib.resources.files("stegdetect_py").joinpath("stegbreak.conf"))
    cfg.cfg_reset()
    cfg.cfg_init(cfg_path)
    obj = prepare_fn(jpg)
    db.db_init()
    db.db_insert("test.jpg", flag, obj, crack_fn, "test.jpg")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write("\n".join(wordlist_words) + "\n")
        wl = fh.name
    try:
        found = do_wordlist_crack(wl)
    finally:
        os.unlink(wl)
    return found


def test_stegbreak_jsteg_e2e():
    jpg = _make_jpg(seed=3)
    password = "korejwa"
    n = 40
    rng = np.random.RandomState(7)
    plain = bytes(rng.randint(1, 255, n - 8)) + bytes([0]) + b"korejwa"
    embed_jsteg_memory(jpg, plain, password.encode())

    def prepare(j):
        dcts, bits = common.prepare_jsteg(j)
        return break_jsteg.break_jsteg_prepare(dcts, bits)

    found = _run_stegbreak(jpg, prepare, break_jsteg.crack_jsteg,
                          FLAG_DOJSTEG, password,
                          ["apple", "banana", "korejwa", "zebra"])
    assert found == "korejwa", f"expected to crack 'korejwa', got {found!r}"


def test_stegbreak_outguess_e2e():
    from tests.test_break_outguess import embed_outguess_memory, _find_seed
    jpg = _make_jpg(seed=11)
    password = "opensesame"
    # OutGuess requires a reasonably long plaintext (>= break_outguess.MIN_LEN).
    plaintext = (b"MEET AT THE OLD CLOCK TOWER AT MIDNIGHT BRING THE DOCUMENTS "
                 b"AND TELL NO ONE THE EAGLE HAS LANDED OVER AND OUT ") * 4
    seed = _find_seed(jpg, plaintext, password.encode())
    embed_outguess_memory(jpg, plaintext, password.encode(), seed)

    def prepare(j):
        dcts, bits = common.prepare_normal(j)
        return break_outguess.break_outguess_prepare(dcts, bits)

    found = _run_stegbreak(jpg, prepare, break_outguess.crack_outguess,
                          FLAG_DOOUTGUESS, password,
                          ["hello", "opensesame", "world"])
    assert found == "opensesame", f"expected 'opensesame', got {found!r}"


def test_stegbreak_jphide_e2e():
    from tests.test_break_jphide import embed_jphide_v3_memory
    jpg = _make_jpg(seed=21)
    password = "s3cr3t-jphide"
    embed_jphide_v3_memory(jpg, password.encode(), length=1000)

    def prepare(j):
        mbits = sum(c.hib * c.wib * 64 for c in j.components)
        return break_jphide.break_jphide_prepare(j, mbits)

    found = _run_stegbreak(jpg, prepare, break_jphide.crack_jphide,
                          FLAG_DOJPHIDE, password,
                          ["alpha", "s3cr3t-jphide", "beta"])
    assert found == "s3cr3t-jphide", f"expected 's3cr3t-jphide', got {found!r}"
