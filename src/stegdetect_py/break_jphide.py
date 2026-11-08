"""
JPHide password cracker -- faithful Python port of break_jphide.c.

JPHide hides its message in the LSB (or second bit, for negative
coefficients) of a selection of coefficients chosen by a Blowfish-driven PRNG
keyed on the candidate password.  ``break_jphide_prepare`` reads the first 256
coefficients of the LTAB traversal (plus the 8-byte IV from component 0's first
block).  ``crack_jphide`` tries both the v3 and v5 verification schemes: each
reads an 8-byte length/IV header with the password-derived Blowfish key and
checks that the decrypted header matches an encryption of the IV.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from . import common, blowfish


class Block:
    """An 8-byte Blowfish block, matching the x86 (little-endian) word layout."""

    __slots__ = ("b",)

    def __init__(self, data: bytes = b"\x00\x00\x00\x00\x00\x00\x00\x00"):
        self.b = bytearray(data[:8])

    def words(self) -> List[int]:
        return [int.from_bytes(self.b[0:4], "little"),
                int.from_bytes(self.b[4:8], "little")]

    def set_words(self, w0: int, w1: int) -> None:
        self.b[0:4] = w0.to_bytes(4, "little")
        self.b[4:8] = w1.to_bytes(4, "little")


def _blf_enc(block: Block, ctx: blowfish.BF_KEY) -> None:
    w0, w1 = block.words()
    ws = [w0, w1]
    blowfish.BF_encrypt(ws, ctx)
    block.set_words(ws[0], ws[1])


def _blf_dec(block: Block, ctx: blowfish.BF_KEY) -> None:
    w0, w1 = block.words()
    ws = [w0, w1]
    blowfish.BF_decrypt(ws, ctx)
    block.set_words(ws[0], ws[1])


class JPHObj:
    """Equivalent of ``struct jphobj``."""

    __slots__ = ("bits", "iv", "wib", "hib", "coeff", "_map")

    def __init__(self):
        self.bits = 0
        self.iv = bytearray(8)
        self.wib = [0, 0, 0]
        self.hib = [0, 0, 0]
        self.coeff = [0] * 256
        self._map = []  # (comp, blk_idx, sub) for each of the 256 coeffs


def _comp_block(comp, lh, lw):
    blk_idx = lh * comp.wib + (lw // 64)
    return comp.blocks[blk_idx][lw % 64]


def break_jphide_prepare(jpg, bits: int) -> JPHObj:
    """Build a :class:`JPHObj` from the decoded JPEG (mirrors break_jphide_prepare)."""
    ltab = common.LTAB
    job = JPHObj()
    job.bits = bits
    for i in range(3):
        job.wib[i] = 64 * jpg.components[i].wib - 1
        job.hib[i] = jpg.components[i].hib

    comp0 = jpg.components[0]
    for i in range(8):
        job.iv[i] = comp0.blocks[0][i] & 0xFF

    coef = ltab[0]
    spos = ltab[1]
    lh = 0
    lw = spos - 64
    lt = 0
    for i in range(256):
        lw += 64
        if lw > job.wib[coef]:
            lw = spos
            lh += 1
            if lh >= job.hib[coef]:
                lt += 3
                if ltab[lt] < 0:
                    break
                coef = ltab[lt]
                lw = spos = ltab[lt + 1]
                lh = 0
        c = jpg.components[coef]
        blk_idx = lh * c.wib + (lw // 64)
        sub = lw % 64
        job.coeff[i] = c.blocks[blk_idx][sub]
        job._map.append((coef, blk_idx, sub))

    return job


class _JPHCracker:
    """Holds the traversal/PRNG state (C uses module globals)."""

    def __init__(self):
        self.ctx = None
        self.iv = bytearray(8)
        self.prngn = [0, 0, 0, 0]
        self.prngstate = [Block() for _ in range(4)]
        self.coef = 0
        self.spos = 0
        self.mode = 0
        self.lh = 0
        self.lw = 0
        self.lt = 0
        self.lwib = [0, 0, 0]
        self.lhib = [0, 0, 0]
        self.coeff = []
        self.where = 0

    # --- Blowfish PRNG bit source -------------------------------------------
    def _get_code_bit(self, k: int) -> int:
        n = self.prngn[k] & 0x3F
        self.prngn[k] += 1
        if n == 0:
            _blf_enc(self.prngstate[k], self.ctx)
        byte = self.prngstate[k].b[(n >> 3) & 7]
        a = (byte << (n & 7)) & 0xFF
        return 1 if (a & 0x80) else 0

    # --- coefficient traversal ---------------------------------------------
    def _get_word(self):
        ltab = common.LTAB
        while True:
            self.lw += 64
            if self.lw > self.lwib[self.coef]:
                self.lw = self.spos
                self.lh += 1
                if self.lh >= self.lhib[self.coef]:
                    self.lt += 3
                    if ltab[self.lt] < 0:
                        return None
                    self.coef = ltab[self.lt]
                    self.lw = self.spos = ltab[self.lt + 1]
                    self.mode = ltab[self.lt + 2]
            y = self.coeff[self.where]
            self.where += 1
            if self.coef == 0 and self.lh == 0 and self.lw <= 7:
                continue
            if self.mode < 0:
                if self.mode <= y <= -self.mode:
                    continue
                if not self._get_code_bit(0) and not self._get_code_bit(0):
                    continue
            else:
                if self.mode == 3 and not self._get_code_bit(0):
                    continue
                if -1 <= y <= 1:
                    if self._get_code_bit(0):
                        continue
                    if self.mode and self._get_code_bit(0):
                        continue
                if self.mode > 1 and not self._get_code_bit(0):
                    continue
            return y

    def _get_bit(self):
        y = self._get_word()
        if y is None:
            return -1
        if y < 0:
            y = -y
        if self.mode < 0:
            return (y & 2) >> 1
        return y & 1

    def _get_bytes(self, n: int):
        out = bytearray()
        for _ in range(n):
            v = 0
            for j in range(8):
                b = self._get_bit()
                if b < 0:
                    return None
                v |= (b << j)
            out.append(v)
        return out

    def setup(self, job: JPHObj, inctx: blowfish.BF_KEY) -> None:
        self.ctx = inctx
        self.iv = bytearray(job.iv)
        self.prngn = [0, 0, 0, 0]
        for i in range(4):
            self.prngstate[i] = Block(bytes(job.iv[:8]))
            _blf_enc(self.prngstate[i], self.ctx)
            first = self.iv[0]
            self.iv = self.iv[1:] + bytes([first])
        ltab = common.LTAB
        self.coef = ltab[0]
        self.spos = ltab[1]
        self.mode = ltab[2]
        self.lh = 0
        self.lw = self.spos - 64
        self.lt = 0
        self.lwib = list(job.wib)
        self.lhib = list(job.hib)
        self.coeff = list(job.coeff)
        self.where = 0

    def _iv_enc(self) -> bytes:
        blk = Block(bytes(self.iv[:8]))
        _blf_enc(blk, self.ctx)
        return bytes(blk.b)

    def v3(self, job: JPHObj, inctx: blowfish.BF_KEY) -> bool:
        self.setup(job, inctx)
        raw = self._get_bytes(8)
        if raw is None:
            return False
        blk = Block(bytes(raw))
        _blf_dec(blk, self.ctx)
        d = blk.b
        length = (d[0] << 16) | (d[1] << 8) | d[2]
        if length * 8 >= job.bits:
            return False
        iv_enc = self._iv_enc()
        for i in range(3, 8):
            if d[i] != iv_enc[i]:
                return False
        return True

    def v5(self, job: JPHObj, inctx: blowfish.BF_KEY) -> bool:
        self.setup(job, inctx)
        raw = self._get_bytes(8)
        if raw is None:
            return False
        blk0 = Block(bytes(raw))
        _blf_dec(blk0, self.ctx)
        d0 = blk0.b
        if d0[3] > 3:
            return False
        length = (d0[0] << 16) | (d0[1] << 8) | d0[2]
        if length * 8 >= job.bits:
            return False
        iv_enc = self._iv_enc()
        if d0[5] != iv_enc[5] or d0[6] != iv_enc[6] or d0[7] != iv_enc[7]:
            return False
        raw2 = self._get_bytes(8)
        if raw2 is None:
            return False
        blk1 = Block(bytes(raw2))
        _blf_dec(blk1, self.ctx)
        d1 = blk1.b
        if d1[1] != d0[1] or d1[2] != d0[2]:
            return False
        rlength = (d0[4] << 16) | (d1[0] << 8) | d1[3]
        if rlength and (rlength < length or rlength > 20 * length):
            return False
        iv2 = Block(bytes(self.iv[:8]))
        _blf_enc(iv2, self.ctx)
        e = iv2.b
        if d1[4] != e[4] or d1[5] != e[5] or d1[6] != e[6] or d1[7] != e[7]:
            return False
        return True


def crack_jphide(filename: str, word: str, job: JPHObj) -> Optional[str]:
    """Try ``word``; return the matched scheme ('v3'/'v5') or None."""
    if job is None:
        return None
    # v5 key = iv[0:6] + password
    key5 = bytes(job.iv[:6]) + word.encode()
    ctx5 = blowfish.BF_KEY()
    blowfish.BF_set_key(ctx5, key5)
    if _JPHCracker().v5(job, ctx5):
        return "v5"

    # v3 key = password
    ctx3 = blowfish.BF_KEY()
    blowfish.BF_set_key(ctx3, word.encode())
    if _JPHCracker().v3(job, ctx3):
        return "v3"
    return None
