"""
OutGuess (old v0.13b) password cracker -- faithful Python port of
break_outguess.c.

OutGuess hides its message in the LSBs of all coefficients, but extraction does
not walk them sequentially: a key-derived "iterator" skips around a bitmap of the
coefficient LSBs so that the embedded bytes land at pseudo-random positions.  A
4-byte state header (seed + length) seeds the iterator; the remaining bytes are
the ARC4-encrypted payload (key derived from the candidate password via MD5).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from . import arc4, common

# Mirror the C limits.
DEFAULT_ITER = 256
INIT_SKIPMOD = 32
SKIPMOD_MAX = 4096


class OGObj:
    """Equivalent of ``struct ogobj``: a bitmap of coefficient LSBs."""

    __slots__ = ("bits", "coeff")

    def __init__(self, bits: int):
        self.bits = bits
        self.coeff = [0] * 4096  # 4096 * 32 bits


class Iterator:
    __slots__ = ("as_", "skipmod", "off")

    def __init__(self):
        self.as_ = arc4.ARC4Stream()
        self.skipmod = INIT_SKIPMOD
        self.off = 0


def _test_bit(coeff: List[int], i: int) -> int:
    return (coeff[i >> 5] >> (i & 31)) & 1


def _write_bit(coeff: List[int], i: int, bit: int) -> None:
    word = i >> 5
    if bit & 1:
        coeff[word] |= (1 << (i & 31))
    else:
        coeff[word] &= ~(1 << (i & 31))


def iterator_init(it: Iterator, as_: arc4.ARC4Stream) -> None:
    it.skipmod = INIT_SKIPMOD
    it.as_ = as_.copy()
    derive = bytes(arc4.arc4_getbyte(it.as_) for _ in range(16))
    arc4.arc4_addrandom(it.as_, derive)
    it.off = arc4.arc4_getword(it.as_) % it.skipmod


def iterator_current(it: Iterator) -> int:
    return it.off


def iterator_next(it: Iterator) -> int:
    it.off += (arc4.arc4_getword(it.as_) % it.skipmod) + 1
    return it.off


def iterator_seed(it: Iterator, seed: int) -> None:
    reseed = bytes([seed & 0xFF, (seed >> 8) & 0xFF])
    arc4.arc4_addrandom(it.as_, reseed)


def _skipadj(x: int, y: int) -> float:
    if y > x / 32:
        return 2.0
    return 2.0 - ((x / 32) - y) / (x / 32.0)


def iterator_adapt(it: Iterator, bits: int, datalen: int) -> None:
    it.skipmod = int(_skipadj(bits, bits - it.off) * (bits - it.off) / (8.0 * datalen))
    if it.skipmod > SKIPMOD_MAX:
        it.skipmod = SKIPMOD_MAX


def steg_retrbyte(coeff: List[int], nbits: int, it: Iterator) -> int:
    i = iterator_current(it)
    tmp = 0
    for where in range(nbits):
        tmp |= _test_bit(coeff, i) << where
        i = iterator_next(it)
    return tmp


def steg_putbyte(coeff: List[int], byte: int, it: Iterator) -> None:
    i = iterator_current(it)
    for where in range(8):
        _write_bit(coeff, i, (byte >> where) & 1)
        i = iterator_next(it)


def break_outguess_prepare(dcts: List[int], bits: int) -> OGObj:
    """Pack the LSBs of every eligible coefficient into the bitmap."""
    og = OGObj(bits)
    maxbits = len(og.coeff) * 32
    j = 0
    for i in range(bits):
        if j >= maxbits:
            break
        _write_bit(og.coeff, j, dcts[i] & 1)
        j += 1
    return og


# Cracker globals.
MIN_LEN = 256
MAX_SEED = 55000


def break_outguess(og: OGObj, as_: arc4.ARC4Stream, it: Iterator) -> Optional[bytes]:
    """Return the decrypted payload if the candidate password matches, else None.

    Mirrors break_outguess(): read the 4-byte state header, seed the iterator,
    walk the payload, then XOR with the master keystream and require the result
    to look like a real file (is_random + file_process).
    """
    tas = as_.copy()

    state = [0] * 4
    for k in range(4):
        state[k] = steg_retrbyte(og.coeff, 8, it) ^ arc4.arc4_getbyte(as_)

    seed = (state[1] << 8) | state[0]
    length = (state[3] << 8) | state[2]

    if seed > MAX_SEED or length * 8 >= og.bits // 2 or length < MIN_LEN:
        return None

    iterator_seed(it, seed)

    bits = min(og.bits, len(og.coeff) * 32)
    buf = bytearray(512)
    n = 0
    remaining = length
    while iterator_current(it) < bits and remaining > 0 and n < 512:
        iterator_adapt(it, og.bits, remaining)
        buf[n] = steg_retrbyte(og.coeff, 8, it)
        n += 1
        remaining -= 1

    need = min(MIN_LEN, 512)
    if n < need or not common.is_random(bytes(buf[:n]), n):
        return None

    for i in range(n):
        buf[i] ^= arc4.arc4_getbyte(tas)

    if common.file_process(bytes(buf[:n])) == 0:
        return None

    return bytes(buf[:n])


def crack_outguess(filename: str, word: str, og: OGObj) -> Optional[bytes]:
    """Try ``word``; return the recovered payload on success, else None."""
    if og is None:
        return None
    as_ = arc4.ARC4Stream()
    arc4.arc4_initkey(as_, word.encode())
    it = Iterator()
    iterator_init(it, as_)
    return break_outguess(og, as_, it)
