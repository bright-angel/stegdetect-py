"""
MD5 message-digest algorithm -- faithful Python port of the stegdetect md5.c
(RFC 1321 reference implementation).

Exposes :func:`md5` returning the 16-byte digest, matching the ``MD5Init`` /
``MD5Update`` / ``MD5Final`` flow used by arc4.c's ``arc4_initkey``.
"""

from __future__ import annotations

import struct
from typing import List


def _leftrotate(x: int, c: int) -> int:
    return ((x << c) | (x >> (32 - c))) & 0xFFFFFFFF


# Per-round shift amounts (RFC 1321, section 3.4, table).
_SHIFT = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
]

# Binary integer part of sin(i) * 2^32, for i = 1..64.
_K = [int(abs(__import__("math").sin(i + 1)) * (2 ** 32)) & 0xFFFFFFFF for i in range(64)]


def md5(data: bytes) -> bytes:
    """Return the 16-byte MD5 digest of ``data`` (matches MD5Final output)."""
    # Initial state (A,B,C,D).
    a0, b0, c0, d0 = 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476

    # Pre-processing: padding.
    msg = bytearray(data)
    orig_len_bits = (len(msg) * 8) & 0xFFFFFFFFFFFFFFFF
    msg.append(0x80)
    while len(msg) % 64 != 56:
        msg.append(0x00)
    msg += struct.pack("<Q", orig_len_bits)

    for chunk_start in range(0, len(msg), 64):
        chunk = msg[chunk_start:chunk_start + 64]
        M = list(struct.unpack("<16I", chunk))  # 16 little-endian 32-bit words

        A, B, C, D = a0, b0, c0, d0

        for i in range(64):
            if i < 16:
                F = (B & C) | (~B & D)
                g = i
            elif i < 32:
                F = (D & B) | (~D & C)
                g = (5 * i + 1) % 16
            elif i < 48:
                F = B ^ C ^ D
                g = (3 * i + 5) % 16
            else:
                F = C ^ (B | ~D)
                g = (7 * i) % 16

            F = (F + A + _K[i] + M[g]) & 0xFFFFFFFF
            A = D
            D = C
            C = B
            B = (B + _leftrotate(F, _SHIFT[i])) & 0xFFFFFFFF

        a0 = (a0 + A) & 0xFFFFFFFF
        b0 = (b0 + B) & 0xFFFFFFFF
        c0 = (c0 + C) & 0xFFFFFFFF
        d0 = (d0 + D) & 0xFFFFFFFF

    return struct.pack("<4I", a0, b0, c0, d0)


def md5_hex(data: bytes) -> str:
    return md5(data).hex()
