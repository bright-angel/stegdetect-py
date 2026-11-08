"""
ARC4 stream cipher -- faithful Python port of arc4.c.

stegbreak derives per-password keys with :func:`arc4_fixedkey` (JSteg) or
:func:`arc4_initkey` (OutGuess), both of which mix in MD5.  The exact PRGA and
state-update semantics from the C sources are preserved (including the
``arc4_fixedkey`` reset of ``i = j = 0`` and the quirky ``arc4_skipbytes``).
"""

from __future__ import annotations

from .md5 import md5


class ARC4Stream:
    """Mirrors ``struct arc4_stream`` (s[256], i, j)."""

    __slots__ = ("s", "i", "j")

    def __init__(self):
        self.s = list(range(256))
        self.i = 0
        self.j = 0

    def copy(self) -> "ARC4Stream":
        c = ARC4Stream()
        c.s = list(self.s)
        c.i = self.i
        c.j = self.j
        return c


def arc4_init(as_: ARC4Stream) -> None:
    as_.s = list(range(256))
    as_.i = 0
    as_.j = 0


def arc4_getbyte(as_: ARC4Stream) -> int:
    s = as_.s
    as_.i = (as_.i + 1) & 0xFF
    si = s[as_.i]
    as_.j = (as_.j + si) & 0xFF
    sj = s[as_.j]
    s[as_.i] = sj
    s[as_.j] = si
    return s[(si + sj) & 0xFF]


def arc4_skipbytes(as_: ARC4Stream, skip: int) -> None:
    s = as_.s
    i = as_.i
    j = as_.j
    while skip > 0:
        i = (i + 1) & 0xFF
        si = s[i]
        j = (si + j) & 0xFF
        sj = s[j]
        s[i] = sj
        s[j] = si
        skip -= 1
    as_.i = i
    as_.j = j


def arc4_getword(as_: ARC4Stream) -> int:
    val = arc4_getbyte(as_) << 24
    val |= arc4_getbyte(as_) << 16
    val |= arc4_getbyte(as_) << 8
    val |= arc4_getbyte(as_)
    return val & 0xFFFFFFFF


def arc4_addrandom(as_: ARC4Stream, dat: bytes) -> None:
    s = as_.s
    i = as_.i - 1
    j = as_.j
    ki = 0
    datlen = len(dat)
    for n in range(256):
        i = (i + 1) & 0xFF
        si = s[i]
        j = (j + si + dat[ki]) & 0xFF
        s[i] = s[j]
        s[j] = si
        ki += 1
        if ki >= datlen:
            ki = 0
    as_.i = i
    as_.j = j


def arc4_initkey(as_: ARC4Stream, key: bytes) -> None:
    """MD5(key) -> 16-byte digest mixed in (OutGuess style)."""
    digest = md5(key)
    arc4_init(as_)
    arc4_addrandom(as_, digest)


def arc4_fixedkey(as_: ARC4Stream, key: bytes) -> None:
    """XOR the key into a 5-byte digest, mix in, then reset i=j=0 (JSteg style)."""
    digest = bytearray(5)
    for i, b in enumerate(key):
        digest[i % 5] ^= b
    arc4_init(as_)
    arc4_addrandom(as_, bytes(digest))
    as_.i = 0
    as_.j = 0
