"""
Validation of the cryptographic primitives (md5, arc4, blowfish) used by
stegbreak.

* md5 / blowfish are standard algorithms and are checked against published
  test vectors (RFC 1321 / the canonical Blowfish all-zero-key vector).
* arc4 (Provos' arc4.c) is a *non-standard* RC4 variant: its key-schedule
  (arc4_addrandom) leaves i=255/j=final and the PRGA continues from there
  rather than from i=j=0.  It therefore does NOT match the standard RC4
  vector, and is instead validated for determinism and XOR round-trip
  consistency -- the properties the crackers actually depend on.
"""

from stegdetect_py import md5, arc4, blowfish


def test_md5_vectors():
    assert md5.md5_hex(b"") == "d41d8cd98f00b204e9800998ecf8427e"
    assert md5.md5_hex(b"abc") == "900150983cd24fb0d6963f7d28e17f72"
    assert md5.md5_hex(
        b"abcdefghijklmnopqrstuvwxyz") == "c3fcd3d76192e4007dfb496cca67e13b"
    print("MD5 vectors: OK")


def test_blowfish_vector():
    # BF(key = 8 zero bytes, block = 0) -> 0x4EF99745 6198DD78
    key = blowfish.BF_KEY()
    blowfish.BF_set_key(key, b"\x00" * 8)
    block = [0, 0]
    blowfish.BF_encrypt(block, key)
    ct = (block[0] << 32) | block[1]
    assert ct == 0x4EF997456198DD78, hex(ct)
    # Round-trip: decrypt restores the original block.
    blowfish.BF_decrypt(block, key)
    assert block == [0, 0], block
    print("Blowfish vector + roundtrip: OK")


def _arc4_stream(key, kind, n):
    as_ = arc4.ARC4Stream()
    if kind == "fixed":
        arc4.arc4_fixedkey(as_, key)
    else:
        arc4.arc4_initkey(as_, key)
    return bytes(arc4.arc4_getbyte(as_) for _ in range(n))


def test_arc4_deterministic_and_roundtrip():
    key = b"secret-password"
    # Determinism.
    s1 = _arc4_stream(key, "fixed", 32)
    s2 = _arc4_stream(key, "fixed", 32)
    assert s1 == s2
    # Different password -> different stream (overwhelmingly likely).
    s3 = _arc4_stream(b"secret-password2", "fixed", 32)
    assert s1 != s3
    # Stream-cipher round trip: keystream XOR encrypts, same keystream decrypts.
    plaintext = b"attack at dawn -- the eagle flies at midnight"
    ks = _arc4_stream(key, "initkey", len(plaintext))
    cipher = bytes(p ^ k for p, k in zip(plaintext, ks))
    assert cipher != plaintext
    ks2 = _arc4_stream(key, "initkey", len(plaintext))
    dec = bytes(c ^ k for c, k in zip(cipher, ks2))
    assert dec == plaintext
    # initkey and fixedkey must differ for the same raw password.
    assert _arc4_stream(key, "initkey", 32) != _arc4_stream(key, "fixed", 32)
    print("ARC4 deterministic + roundtrip: OK")


if __name__ == "__main__":
    test_md5_vectors()
    test_blowfish_vector()
    test_arc4_deterministic_and_roundtrip()
    print("\nALL CRYPTO TESTS OK")
