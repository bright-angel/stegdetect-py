"""
End-to-end validation of the OutGuess cracker.

OutGuess hides data in the LSBs of all coefficients but extracts it via a
key-derived skip-iterator over a coefficient bitmap.  We build a carrier in
memory by walking the *same* iterator the cracker uses (so positions line up),
place the 4-byte state header and the ARC4-encrypted payload, then confirm
``crack_outguess`` recovers the password and payload.
"""

import io
import numpy as np
from PIL import Image

from stegdetect_py.jpeg import decode_coefficients
from stegdetect_py import common, arc4, break_outguess


def _eligible_refs(jpg):
    refs = []
    for ci, comp in enumerate(jpg.components):
        for bi, blk in enumerate(comp.blocks):
            for si, val in enumerate(blk):
                if (val & 1) == val:
                    continue
                refs.append((ci, bi, si))
    return refs


def embed_outguess_memory(jpg, plaintext, password, seed):
    """Embed ``plaintext`` (>=256 bytes ASCII) using the OutGuess scheme."""
    refs = _eligible_refs(jpg)
    bits = len(refs)
    n = len(plaintext)
    assert n >= break_outguess.MIN_LEN, "plaintext must be >= MIN_LEN"
    assert n * 8 < bits // 2, "plaintext too long for carrier"

    og = break_outguess.OGObj(bits)

    master = arc4.ARC4Stream()
    arc4.arc4_initkey(master, password)
    master2 = arc4.ARC4Stream()
    arc4.arc4_initkey(master2, password)

    it = break_outguess.Iterator()
    break_outguess.iterator_init(it, master)

    state = [seed & 0xFF, (seed >> 8) & 0xFF, n & 0xFF, (n >> 8) & 0xFF]
    for k in range(4):
        ks = arc4.arc4_getbyte(master)
        stored = state[k] ^ ks
        break_outguess.steg_putbyte(og.coeff, stored, it)
        if break_outguess.iterator_current(it) >= bits:
            return None  # iterator ran off the end

    break_outguess.iterator_seed(it, seed)

    remaining = n
    placed = 0
    while remaining > 0:
        break_outguess.iterator_adapt(it, og.bits, remaining)
        ks = arc4.arc4_getbyte(master2)
        stored = plaintext[placed] ^ ks
        break_outguess.steg_putbyte(og.coeff, stored, it)
        placed += 1
        remaining -= 1
        if break_outguess.iterator_current(it) >= bits:
            return None

    # Write the bitmap LSBs back into the actual coefficients.
    for j in range(bits):
        if break_outguess._test_bit(og.coeff, j):
            ci, bi, si = refs[j]
            v = jpg.components[ci].blocks[bi][si]
            jpg.components[ci].blocks[bi][si] = (v & ~1) | 1
        else:
            ci, bi, si = refs[j]
            v = jpg.components[ci].blocks[bi][si]
            jpg.components[ci].blocks[bi][si] = (v & ~1) | 0
    return bits


def _make_jpg(seed):
    rng = np.random.RandomState(seed)
    im = Image.fromarray(rng.randint(0, 256, (160, 160, 3), dtype=np.uint8), "RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92, subsampling=0)
    return decode_coefficients(buf.getvalue())


def _find_seed(jpg, plaintext, password):
    for seed in range(1, 55000):
        if embed_outguess_memory(jpg, plaintext, password, seed) is not None:
            return seed
    raise RuntimeError("no seed produced an in-bounds embedding")


def test_outguess_roundtrip():
    jpg = _make_jpg(5)
    password = b"letmein123"
    plaintext = (b"MEET AT THE OLD CLOCK TOWER AT MIDNIGHT BRING THE DOCUMENTS "
                 b"AND TELL NO ONE THE EAGLE HAS LANDED OVER AND OUT ") * 4  # >256 bytes
    assert len(plaintext) >= 256

    seed = _find_seed(jpg, plaintext, password)

    dcts, bits = common.prepare_normal(jpg)
    og = break_outguess.break_outguess_prepare(dcts, bits)
    payload = break_outguess.crack_outguess("test.jpg", password.decode(), og)
    assert payload is not None, "cracker failed to recover the OutGuess payload"
    assert payload.startswith(b"MEET AT THE OLD CLOCK"), payload[:32]
    print("OutGuess roundtrip: OK (seed=%d)" % seed)


if __name__ == "__main__":
    test_outguess_roundtrip()
    print("\nALL OUTGUESS CRACKER TESTS OK")
