"""
JSteg password cracker -- faithful Python port of break_jsteg.c.

The carrier holds the encrypted message in the least-significant bits of the
JSteg-eligible coefficients (all coefficients whose value is not in {0, 1}), with
a ``jsteg_size`` length header at the very start.  ``break_jsteg_prepare``
extracts the trailing 8 message bytes (the "coeff" field) and the leading 64
bytes (the "header" field).  ``crack_jsteg`` derives an ARC4 keystream from a
candidate password (arc4_fixedkey), skips ``total_bytes - 8`` keystream bytes,
and checks the decrypted tail for the ``korejwa`` / ``cMk`` magic.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from . import arc4, common

JSTEGBYTES = 8
JSTEGHEADER = 64


class JStegObj:
    """Equivalent of ``struct jstegobj``."""

    __slots__ = ("skip", "coeff", "header")

    def __init__(self):
        self.skip = 0
        self.coeff = bytearray(JSTEGBYTES)
        self.header = bytearray(JSTEGHEADER)


def _extract_bytes(dcts: List[int], start_bit: int, nbytes: int) -> bytearray:
    """Read ``nbytes`` little... actually MSB-first bytes from LSBs of dcts.

    Starting at coefficient bit index ``start_bit`` (i.e. dcts[start_bit] holds
    the first LSB), pack each group of 8 LSBs into one byte, MSB first.
    """
    out = bytearray(nbytes)
    pos = start_bit
    for b in range(nbytes):
        acc = 0
        for _ in range(8):
            acc = (acc << 1) | (dcts[pos] & 1)
            pos += 1
        out[b] = acc & 0xFF
    return out


def break_jsteg_prepare(dcts: List[int], bits: int, filename: str = "") -> Optional[JStegObj]:
    """Build a :class:`JStegObj` from the eligible-coefficient stream.

    Mirrors ``break_jsteg_prepare``: locate the message via ``jsteg_size``, then
    copy the trailing 8 bytes (coeff) and the leading 64 bytes (header).
    """
    off = [0]
    nbytes = common.jsteg_size(dcts, bits, off)
    jsbits = nbytes * 8
    obj = JStegObj()
    obj.skip = nbytes - JSTEGBYTES

    if jsbits < (JSTEGBYTES * 8) or off[0] + jsbits > bits:
        return None

    # Trailing 8 bytes of the message.
    obj.coeff = _extract_bytes(dcts, off[0] + jsbits - (JSTEGBYTES * 8), JSTEGBYTES)

    # Leading (up to) 64 bytes of the message.
    maxbits = JSTEGHEADER * 8
    if maxbits > jsbits:
        maxbits = jsbits
    obj.header = _extract_bytes(dcts, off[0], maxbits // 8)

    return obj


def _plain_ok(plain: bytearray) -> bool:
    """Check the decrypted tail for the JSteg magic (break_jsteg)."""
    if plain[1:8] == b"korejwa":
        return True
    if plain[4:7] == b"cMk":
        n = plain[7]
        if n == 4 or n == 5:
            return True
    return False


def break_jsteg(obj: JStegObj, as_: arc4.ARC4Stream) -> bool:
    """Decrypt the trailing 8 bytes with ``as_`` and test for the magic."""
    arc4.arc4_skipbytes(as_, obj.skip)
    plain = bytearray(JSTEGBYTES)
    for i in range(JSTEGBYTES):
        plain[i] = obj.coeff[i] ^ arc4.arc4_getbyte(as_)
    return _plain_ok(plain)


def crack_jsteg(filename: str, word: str, obj: JStegObj) -> Tuple[bool, Optional[bytearray]]:
    """Try ``word`` against the prepared carrier; return (found, header_or_None)."""
    if obj is None:
        return False, None
    as_ = arc4.ARC4Stream()
    arc4.arc4_fixedkey(as_, word.encode())
    if not break_jsteg(obj, as_):
        return False, None
    # Recover and decrypt the header to identify the embedded file type.
    as2 = arc4.ARC4Stream()
    arc4.arc4_fixedkey(as2, word.encode())
    header = bytearray(JSTEGHEADER)
    for i in range(JSTEGHEADER):
        header[i] = obj.header[i] ^ arc4.arc4_getbyte(as2)
    return True, header
