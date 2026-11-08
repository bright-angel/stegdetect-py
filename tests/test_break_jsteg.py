"""
End-to-end validation of the JSteg cracker.

Real JSteg-carrier JPEGs can only be produced by the external jsteg/outguess
tools, which are not part of this repository.  To validate the cracker we build
a carrier *in memory*: decode a JPEG to its coefficient representation, simulate
the JSteg embedding (length header + ARC4-encrypted payload placed in the LSBs
of the eligible coefficients), then confirm ``crack_jsteg`` recovers the password
and magic.  The embedding scheme matches what break_jsteg.c assumes.
"""

from stegdetect_py import common, arc4, break_jsteg


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
    """Simulate JSteg embedding by patching coefficients in memory.

    ``plain`` is the message bytes whose last 8 bytes must carry the magic
    (``[0] + b'korejwa'``).  Returns the number of eligible coefficients used.
    """
    refs = _eligible_refs(jpg)
    n = len(plain)
    width = max(1, (n).bit_length())
    # length header: 5-bit width field, then width-bit length value.
    header_bits = []
    for b in range(5):
        header_bits.append((width >> (4 - b)) & 1)
    for b in range(width):
        header_bits.append((n >> (width - 1 - b)) & 1)

    # ARC4 keystream from the password (reset i=j=0, like arc4_fixedkey).
    as_ = arc4.ARC4Stream()
    arc4.arc4_fixedkey(as_, password)
    ks = [arc4.arc4_getbyte(as_) for _ in range(n)]
    cipher = bytes(p ^ k for p, k in zip(plain, ks))

    # Full bit stream: header then ciphertext (MSB first per byte).
    bits = list(header_bits)
    for byte in cipher:
        for b in range(8):
            bits.append((byte >> (7 - b)) & 1)

    if len(bits) > len(refs):
        raise ValueError("not enough eligible coefficients to embed %d bits" % len(bits))
    for k, bit in enumerate(bits):
        ci, bi, si = refs[k]
        v = jpg.components[ci].blocks[bi][si]
        jpg.components[ci].blocks[bi][si] = (v & ~1) | bit
    return len(bits)


def test_jsteg_roundtrip():
    import io, numpy as np
    from PIL import Image
    from stegdetect_py.jpeg import decode_coefficients

    rng = np.random.RandomState(3)
    im = Image.fromarray(rng.randint(0, 256, (160, 160, 3), dtype=np.uint8), "RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90, subsampling=0)
    jpg = decode_coefficients(buf.getvalue())

    password = b"correct horse battery staple"
    # Message: arbitrary header + last 8 bytes = [0] + 'korejwa'.
    n = 40
    plain = bytes(rng.randint(1, 255, n - 8)) + bytes([0]) + b"korejwa"
    embed_jsteg_memory(jpg, plain, password)

    dcts, bits = common.prepare_jsteg(jpg)
    obj = break_jsteg.break_jsteg_prepare(dcts, bits)
    assert obj is not None, "prepare returned None (bad size)"

    found, header = break_jsteg.crack_jsteg("test.jpg", password.decode(), obj)
    assert found, "cracker failed to find the correct password"
    assert header is not None

    # A wrong password must not validate.
    wrong, _ = break_jsteg.crack_jsteg("test.jpg", "wrongpassword", obj)
    assert not wrong

    print("JSteg roundtrip: OK (password found, wrong rejected)")


def test_jsteg_cmk_magic():
    import io, numpy as np
    from PIL import Image
    from stegdetect_py.jpeg import decode_coefficients

    rng = np.random.RandomState(9)
    im = Image.fromarray(rng.randint(0, 256, (160, 160, 3), dtype=np.uint8), "RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90, subsampling=0)
    jpg = decode_coefficients(buf.getvalue())

    password = b"opensesame"
    n = 40
    # cMk magic lives in the LAST 8 bytes: tail[4:7]='cMk', tail[7]=4 (or 5).
    plain = bytes(rng.randint(1, 255, n - 8)) + b"\x00\x00\x00\x00" + b"cMk" + bytes([4])
    embed_jsteg_memory(jpg, plain, password)

    dcts, bits = common.prepare_jsteg(jpg)
    obj = break_jsteg.break_jsteg_prepare(dcts, bits)
    assert obj is not None
    found, _ = break_jsteg.crack_jsteg("test.jpg", password.decode(), obj)
    assert found, "cracker failed to detect cMk magic"
    print("JSteg cMk magic: OK")


if __name__ == "__main__":
    test_jsteg_roundtrip()
    test_jsteg_cmk_magic()
    print("\nALL JSTEG CRACKER TESTS OK")
