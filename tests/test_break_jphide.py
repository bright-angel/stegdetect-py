"""
End-to-end validation of the JPHide cracker (v3 scheme).

We build a JPHide carrier in memory: prepare the 256-coefficient selection, run
the Blowfish-PRNG traversal to discover which coefficients carry the first 8
message bytes, and set their LSBs so the decrypted header matches the v3
signature (length field + encryption of the rotated IV).  ``crack_jphide`` must
then report a v3 match.
"""

import io
import numpy as np
from PIL import Image

from stegdetect_py.jpeg import decode_coefficients
from stegdetect_py import blowfish, break_jphide


def _make_jpg(seed):
    rng = np.random.RandomState(seed)
    im = Image.fromarray(rng.randint(0, 256, (160, 160, 3), dtype=np.uint8), "RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=95, subsampling=0)
    return decode_coefficients(buf.getvalue())


def _mbits(jpg):
    return sum(c.hib * c.wib * 64 for c in jpg.components)


def _blf_enc_bytes(data8, ctx):
    from stegdetect_py.break_jphide import Block, _blf_enc
    blk = Block(data8)
    _blf_enc(blk, ctx)
    return bytes(blk.b)


def embed_jphide_v3_memory(jpg, password, length=1000):
    bits = _mbits(jpg)
    job = break_jphide.break_jphide_prepare(jpg, bits)

    ctx3 = blowfish.BF_KEY()
    blowfish.BF_set_key(ctx3, password)

    cr = break_jphide._JPHCracker()
    cr.setup(job, ctx3)

    # Discover the 64 selected coefficients (8 header bytes) and their mode.
    positions = []
    for _ in range(64):
        y = cr._get_word()
        if y is None:
            raise RuntimeError("traversal terminated before 64 bits")
        positions.append((cr.where - 1, cr.mode))

    # Build the v3 header plaintext and encrypt it with the password key.
    iv_enc = _blf_enc_bytes(bytes(cr.iv[:8]), ctx3)
    header_plain = bytes([(length >> 16) & 0xFF, (length >> 8) & 0xFF,
                          length & 0xFF]) + iv_enc[3:8]
    header_cipher = _blf_enc_bytes(header_plain, ctx3)

    for j in range(64):
        idx, mode = positions[j]
        bit = (header_cipher[j >> 3] >> (j & 7)) & 1
        y = job.coeff[idx]
        if mode >= 0:
            job.coeff[idx] = (y & ~1) | bit
        else:
            job.coeff[idx] = (y & ~2) | (bit << 1)

    # Write the modified coefficients back into the image.
    for i, (coef, blk_idx, sub) in enumerate(job._map):
        jpg.components[coef].blocks[blk_idx][sub] = job.coeff[i]
    return bits


def test_jphide_v3_roundtrip():
    jpg = _make_jpg(7)
    password = b"s3cr3t-jphide"
    bits = embed_jphide_v3_memory(jpg, password, length=1000)

    job2 = break_jphide.break_jphide_prepare(jpg, bits)
    scheme = break_jphide.crack_jphide("test.jpg", password.decode(), job2)
    assert scheme == "v3", "expected v3 match, got %r" % (scheme,)

    # Wrong password must not match.
    wrong = break_jphide.crack_jphide("test.jpg", "nope", job2)
    assert wrong is None
    print("JPHide v3 roundtrip: OK")


if __name__ == "__main__":
    test_jphide_v3_roundtrip()
    print("\nALL JPHIDE CRACKER TESTS OK")
