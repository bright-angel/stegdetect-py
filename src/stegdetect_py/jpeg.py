"""
Pure-Python JPEG decoder that reproduces the *quantized* DCT coefficients
returned by libjpeg's ``jpeg_read_coefficients``.

The detection algorithms in stegdetect operate on the *quantized* DCT
coefficients (the integer values present in the entropy-coded data, before
dequantization / IDCT).  This module parses a JPEG bitstream, decodes the
Huffman entropy data for a baseline / extended-sequential (8-bit) Huffman
stream, and produces, for every image component, a grid of 8x8 blocks where
each block is a flat list of 64 coefficients stored in *zig-zag (scan) order*
-- exactly the layout used by libjpeg's coefficient buffers and by the
original stegdetect C code (``dctcompbuf[comp][row][col][i]`` with ``i`` in
zig-zag order).

Only the information needed by stegdetect is extracted:
  * per-component quantized coefficient blocks (hib x wib x 64)
  * quantization tables (zig-zag order)
  * COM (comment) segments, APPn presence bitmask, JFIF version
  * raw bytes that follow the EOI marker (used for "appended" detection)

Progressive and 12-bit JPEGs are not supported (the original tool relies on
libjpeg which supports them, but they are extremely rare for steganalysis
targets); an informative ``JPEGNotSupported`` is raised instead.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


class JPEGError(Exception):
    pass


class JPEGNotSupported(JPEGError):
    pass


class ScanEnded(Exception):
    """Raised internally when an entropy scan terminates early (EOI / marker)."""
    def __init__(self, marker: int):
        self.marker = marker
        super().__init__("scan ended at marker 0xFF%02X" % marker)


# Standard JPEG zig-zag scan order.  zz[s] = row*8 + col for scan position s.
ZZ = [
    0,  1,  8, 16,  9,  2,  3, 10,
   17, 24, 32, 25, 18, 11,  4,  5,
   12, 19, 26, 33, 40, 48, 41, 34,
   27, 20, 13,  6,  7, 14, 21, 28,
   35, 42, 49, 56, 57, 50, 43, 36,
   29, 22, 15, 23, 30, 37, 44, 51,
   58, 59, 52, 45, 38, 31, 39, 46,
   53, 60, 61, 54, 47, 55, 62, 63,
]

# Inverse: given (row,col) encoded as idx = row*8+col, the scan position.
ZZ_INV = [0] * 64
for _s, _v in enumerate(ZZ):
    ZZ_INV[_v] = _s


class HuffmanTable:
    """Canonical JPEG Huffman decode table (class=0 DC, class=1 AC)."""

    def __init__(self, bits: bytes, values: bytes):
        # bits[0] is the count for length 1, ..., bits[15] for length 16
        self.bits = bits
        self.values = values
        self._build()

    def _build(self):
        # Build, for each code length, the first code and first value index.
        self.mincode = [-1] * 17
        self.maxcode = [-1] * 17
        self.valptr = [0] * 17
        code = 0
        k = 0
        for l in range(1, 17):
            if self.bits[l - 1]:
                self.valptr[l] = k
                self.mincode[l] = code
                code += self.bits[l - 1]
                self.maxcode[l] = code - 1
                k += self.bits[l - 1]
            else:
                self.maxcode[l] = -1
            code <<= 1
        self.last_length = 16
        while self.last_length > 0 and self.maxcode[self.last_length] < 0:
            self.last_length -= 1

    def decode(self, br: "BitReader") -> int:
        code = 0
        for l in range(1, 17):
            bit = br.get_bit()
            code = (code << 1) | bit
            if self.maxcode[l] >= 0 and self.mincode[l] <= code <= self.maxcode[l]:
                idx = self.valptr[l] + (code - self.mincode[l])
                if idx >= len(self.values):
                    raise JPEGError("huffman index out of range")
                return self.values[idx]
        raise JPEGError("huffman decode failed (no code found)")


class BitReader:
    """Bit reader over the entropy-coded segment with 0xFF stuffing handling.

    When a 0xFF is read, the following byte decides the meaning:
      * 0x00  -> escaped 0xFF (a real data byte of 0xFF)
      * other -> a marker; raises ScanEnded with that marker value.
    """

    def __init__(self, data: bytes, start: int, end: int):
        self.data = data
        self.pos = start
        self.end = end
        self.bitbuf = 0
        self.bitcnt = 0

    def _next_byte(self) -> int:
        if self.pos >= self.end:
            raise ScanEnded(0xD9)  # treat premature end as EOI
        b = self.data[self.pos]
        self.pos += 1
        if b != 0xFF:
            return b
        # Marker territory: read next byte.
        if self.pos >= self.end:
            raise ScanEnded(0xD9)
        nxt = self.data[self.pos]
        self.pos += 1
        if nxt == 0x00:
            return 0xFF  # stuffed 0xFF
        # A real marker.
        raise ScanEnded(nxt)

    def get_bit(self) -> int:
        if self.bitcnt == 0:
            self.bitbuf = self._next_byte()
            self.bitcnt = 8
        self.bitcnt -= 1
        return (self.bitbuf >> self.bitcnt) & 1

    def get_bits(self, n: int) -> int:
        v = 0
        for _ in range(n):
            v = (v << 1) | self.get_bit()
        return v

    def align(self):
        """Discard remaining bits in the current byte (move to byte boundary)."""
        self.bitcnt = 0

    def read_restart(self) -> int:
        """Read a restart marker (FFD0..FFD7). Assumes byte alignment."""
        self.align()
        if self.pos + 2 > self.end:
            raise ScanEnded(0xD9)
        if self.data[self.pos] != 0xFF:
            raise JPEGError("expected restart marker")
        m = self.data[self.pos + 1]
        self.pos += 2
        if not (0xD0 <= m <= 0xD7):
            raise JPEGError("not a restart marker: 0xFF%02X" % m)
        return m


def _receive_and_extend(br: BitReader, s: int) -> int:
    """Receive s bits and extend to a signed value (JPEG "extend" function)."""
    if s == 0:
        return 0
    v = br.get_bits(s)
    if v < (1 << (s - 1)):
        v += (-1 << s) + 1
    return v


@dataclass
class Component:
    cid: int
    h: int
    v: int
    qt: int
    # Grid of blocks: blocks[row * wib + col] is a list of 64 ints (zig-zag order)
    blocks: List[List[int]] = field(default_factory=list)
    wib: int = 0
    hib: int = 0
    quant: Optional[List[int]] = None


@dataclass
class JPEG:
    width: int = 0
    height: int = 0
    components: List[Component] = field(default_factory=list)
    quant_tables: Dict[int, List[int]] = field(default_factory=dict)
    comments: List[bytes] = field(default_factory=list)
    markers: int = 0            # bitmask: (1 << (APPn - APP0)) for each APPn present
    jfif: Optional[Tuple[int, int]] = None
    trailing: bytes = b""       # bytes after EOI (appended data)
    restart_interval: int = 0   # from DRI marker (0 = none)

    def component_by_id(self, cid: int) -> Component:
        for c in self.components:
            if c.cid == cid:
                return c
        raise JPEGError("component id %d not found" % cid)


def _parse_markers(data: bytes) -> dict:
    """Walk the marker segments. Returns a dict with parsed structures and the
    offset where the entropy (SOS) data begins."""
    if data[:2] != b"\xff\xd8":
        raise JPEGError("not a JPEG file (missing SOI)")

    info = {
        "dqts": {},          # qt index -> 64 list (zig-zag order)
        "sof": None,         # (precision, height, width, [comp tuples])
        "dht": [],           # list of (class, id, HuffmanTable)
        "sos": None,         # (ncomp, [(selector, dc, ac)], sos_data_start, sos_data_len)
        "comments": [],
        "markers": 0,
        "jfif": None,
        "eoi_pos": None,
        "restart_interval": 0,
    }

    pos = 2
    n = len(data)
    # Track whether we are in the marker stream (before SOS entropy data).
    while pos < n:
        if data[pos] != 0xFF:
            # Skip stray byte (shouldn't normally happen before SOS).
            pos += 1
            continue
        # Skip fill bytes.
        while pos < n and data[pos] == 0xFF:
            pos += 1
        if pos >= n:
            break
        marker = data[pos]
        pos += 1

        if marker == 0xD9:  # EOI
            info["eoi_pos"] = pos - 2
            break
        if 0xD0 <= marker <= 0xD7:  # restart marker in stream (unexpected here)
            continue
        if marker == 0x01 or (0xD0 <= marker <= 0xD7):
            continue

        # Read segment length (Lp), includes the 2 length bytes themselves.
        if pos + 2 > n:
            break
        seglen = struct.unpack(">H", data[pos:pos + 2])[0]
        seg_start = pos
        seg_end = pos + seglen  # exclusive

        if marker == 0xDA:  # SOS
            p = pos + 2
            ns = data[p]
            p += 1
            comps = []
            for _ in range(ns):
                sel = data[p]
                dcac = data[p + 1]
                p += 2
                comps.append((sel, dcac >> 4, dcac & 0xF))
            # Skip Ss, Se, Ah/Al
            p += 3
            info["sos"] = (ns, comps, seg_end, None)
            info["sos_data_start"] = seg_end
            # Entropy data runs until EOI (or end).  We'll locate EOI later.
            return info

        elif marker == 0xFE:  # COM - comment segment
            # seg_start points at the 2 length bytes; skip them to get the text.
            info["comments"].append(data[seg_start + 2:seg_end])

        elif 0xE0 <= marker <= 0xEF:  # APPn
            info["markers"] |= 1 << (marker - 0xE0)
            # Skip the 2 length bytes to reach the payload.
            appdata = data[seg_start + 2:seg_end]
            if marker == 0xE0 and appdata[:5] == b"JFIF\x00":
                if len(appdata) >= 7:
                    jmajor = appdata[5]
                    jminor = appdata[6]
                    info["jfif"] = (jmajor, jminor)

        elif marker == 0xDB:  # DQT
            p = pos + 2
            while p < seg_end:
                pq_tq = data[p]
                p += 1
                prec = pq_tq >> 4
                tq = pq_tq & 0x0F
                count = 64 if prec == 0 else 64 * 2
                if p + count > seg_end:
                    break
                if prec == 0:
                    qt = list(data[p:p + 64])
                else:
                    qt = list(struct.unpack(">64H", data[p:p + 128]))
                info["dqts"][tq] = qt
                p += count

        elif marker == 0xC4:  # DHT
            p = pos + 2
            while p < seg_end:
                tc_th = data[p]
                p += 1
                cls = tc_th >> 4
                th = tc_th & 0x0F
                bits = data[p:p + 16]
                p += 16
                nvals = sum(bits)
                values = data[p:p + nvals]
                p += nvals
                info["dht"].append((cls, th, HuffmanTable(bytes(bits), bytes(values))))

        elif 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            # SOF0..SOF15.  We support baseline (0xC0) and extended-sequential
            # Huffman (0xC1).  Progressive (0xC2) and 12-bit are not supported.
            if marker not in (0xC0, 0xC1):
                raise JPEGNotSupported(
                    "SOF marker 0xFF%02X not supported (only baseline/"
                    "extended-sequential Huffman 8-bit)" % marker)
            p = pos + 2
            precision = data[p]
            p += 1
            height, width = struct.unpack(">HH", data[p:p + 4])
            p += 4
            nc = data[p]
            p += 1
            comps = []
            for _ in range(nc):
                cid = data[p]
                hv = data[p + 1]
                qt = data[p + 2]
                p += 3
                comps.append((cid, hv >> 4, hv & 0x0F, qt))
            info["sof"] = (precision, height, width, comps)
            if precision != 8:
                raise JPEGNotSupported("only 8-bit precision supported")

        elif marker == 0xDD:  # DRI - define restart interval
            if pos + 4 <= seg_end:
                info["restart_interval"] = struct.unpack(">H", data[pos + 2:pos + 4])[0]

        # Skip to end of this segment.
        pos = seg_end
    return info


def _locate_eoi(data: bytes, start: int) -> int:
    """Find the first FFD9 (EOI) starting at `start`. Returns its offset, or
    len(data) if not found."""
    i = start
    n = len(data)
    while i + 1 < n:
        if data[i] == 0xFF and data[i + 1] == 0xD9:
            return i
        i += 1
    return n


def decode_coefficients(data: bytes) -> JPEG:
    """Decode a JPEG and return a :class:`JPEG` with quantized coefficients."""
    info = _parse_markers(data)
    if info["sof"] is None:
        raise JPEGError("no SOF marker found")
    if info["sos"] is None:
        raise JPEGError("no SOS marker found")

    precision, height, width, sof_comps = info["sof"]
    ns, sos_comps, sos_data_start, _ = info["sos"]

    # Map component selector -> (dc_table, ac_table) from SOS.
    comp_tables = {}
    for sel, dc, ac in sos_comps:
        comp_tables[sel] = (dc, ac)

    # Build Huffman lookup: (class, id) -> table
    htables: Dict[Tuple[int, int], HuffmanTable] = {}
    for cls, th, tbl in info["dht"]:
        htables[(cls, th)] = tbl

    # Build component objects.
    jpg = JPEG(
        width=width,
        height=height,
        comments=info["comments"],
        markers=info["markers"],
        jfif=info["jfif"],
        restart_interval=info["restart_interval"],
    )
    for cid, h, v, qt in sof_comps:
        comp = Component(cid=cid, h=h, v=v, qt=qt)
        comp.quant = info["dqts"].get(qt)
        jpg.components.append(comp)

    # Determine MCU grid.
    Hmax = max(c.h for c in jpg.components)
    Vmax = max(c.v for c in jpg.components)
    mcu_cols = (width + 8 * Hmax - 1) // (8 * Hmax)
    mcu_rows = (height + 8 * Vmax - 1) // (8 * Vmax)

    for c in jpg.components:
        c.wib = mcu_cols * c.h
        c.hib = mcu_rows * c.v
        c.blocks = [None] * (c.wib * c.hib)  # type: ignore

    # Locate EOI / end of entropy data.
    eoi_pos = _locate_eoi(data, sos_data_start)
    jpg.trailing = data[eoi_pos + 2:] if eoi_pos + 2 <= len(data) else b""
    ent_end = eoi_pos if eoi_pos < len(data) else len(data)

    br = BitReader(data, sos_data_start, ent_end)

    # DC predictors per component (reset at restarts).
    pred = [0] * len(jpg.components)

    restart_interval = jpg.restart_interval
    mcu_index = 0
    try:
        for mcu_row in range(mcu_rows):
            for mcu_col in range(mcu_cols):
                # Restart handling: consume an RST marker and reset predictors.
                if restart_interval and mcu_index > 0 and \
                        mcu_index % restart_interval == 0:
                    br.read_restart()
                    pred = [0] * len(jpg.components)
                for ci, comp in enumerate(jpg.components):
                    dc_tbl = htables[(0, comp_tables[comp.cid][0])]
                    ac_tbl = htables[(1, comp_tables[comp.cid][1])]
                    for sub_row in range(comp.v):
                        for sub_col in range(comp.h):
                            block = _decode_block(br, dc_tbl, ac_tbl, pred, ci)
                            row = mcu_row * comp.v + sub_row
                            col = mcu_col * comp.h + sub_col
                            comp.blocks[row * comp.wib + col] = block
                mcu_index += 1
    except ScanEnded:
        # Premature end: leave remaining blocks as zeros (None -> zeros).
        pass

    # Fill any undecoded blocks with zeros.
    for c in jpg.components:
        for i in range(len(c.blocks)):
            if c.blocks[i] is None:
                c.blocks[i] = [0] * 64

    return jpg


def _decode_block(br: BitReader, dc_tbl: HuffmanTable,
                  ac_tbl: HuffmanTable, pred: List[int], ci: int) -> List[int]:
    block = [0] * 64
    # DC
    t = dc_tbl.decode(br)
    diff = _receive_and_extend(br, t)
    pred[ci] += diff
    block[0] = pred[ci]
    # AC
    k = 1
    while k < 64:
        rs = ac_tbl.decode(br)
        r = rs >> 4
        s = rs & 0x0F
        if s == 0:
            if r == 15:
                k += 16
                continue
            else:
                break  # EOB
        else:
            k += r
            if k >= 64:
                break
            block[k] = _receive_and_extend(br, s)
            k += 1
    return block
