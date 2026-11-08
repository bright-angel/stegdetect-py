"""
stegbreak -- faithful Python port of stegbreak.c.

Brute-forces the passphrase of JSteg / OutGuess / JPHide steganographic JPEGs by
running a wordlist through a set of John-the-Ripper style mangling rules.  The
rule set is read from a config file (``[List.Rules:Wordlist]`` section); the
crackers themselves live in :mod:`stegdetect_py.break_jsteg`,
:mod:`stegdetect_py.break_outguess` and :mod:`stegdetect_py.break_jphide`.

Usage (mirrors stegbreak.1)::

    stegbreak [-V] [-r <rules>] [-f <wordlist>] [-t <schemes>] file.jpg ...
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from . import common, db, cfg, rpp, rules as rules_mod
from . import break_jsteg, break_outguess, break_jphide

# Scheme flags (stegbreak.c).
FLAG_DOOUTGUESS = 0x0001
FLAG_DOJPHIDE = 0x0002
FLAG_DOJSTEG = 0x0004

# Default rules file (RULES_NAME).  Looks next to this module, then in CWD.
_DEFAULT_RULES = "stegbreak.conf"

# Default wordlist, mirroring the C default "/usr/share/dict/words".
_DEFAULT_WORDLIST = "/usr/share/dict/words"

quiet = False
convert = False

HANDLERS = [
    {
        "type": FLAG_DOJPHIDE, "extension": ".jph",
        "crack": break_jphide.crack_jphide,
        "read": "jphide_read_jpg",
    },
    {
        "type": FLAG_DOOUTGUESS, "extension": ".og",
        "crack": break_outguess.crack_outguess,
        "read": "outguess_read_jpg",
    },
    {
        "type": FLAG_DOJSTEG, "extension": ".jsg",
        "crack": break_jsteg.crack_jsteg,
        "read": "jsteg_read_jpg",
    },
]


# ---------------------------------------------------------------------------
# JPEG -> prepared carrier objects
# ---------------------------------------------------------------------------
def jsteg_read_jpg(path: str):
    jpg = common.load_image(path)
    dcts, bits = common.prepare_jsteg(jpg)
    return break_jsteg.break_jsteg_prepare(dcts, bits, path)


def outguess_read_jpg(path: str):
    jpg = common.load_image(path)
    dcts, bits = common.prepare_normal(jpg)
    return break_outguess.break_outguess_prepare(dcts, bits)


def jphide_read_jpg(path: str):
    jpg = common.load_image(path)
    mbits = sum(c.hib * c.wib * 64 for c in jpg.components)
    return break_jphide.break_jphide_prepare(jpg, mbits)


_READERS = {
    "jsteg": jsteg_read_jpg,
    "outguess": outguess_read_jpg,
    "jphide": jphide_read_jpg,
}


def _has_extension(name: str, ext: str) -> bool:
    return name.lower().endswith(ext)


def doinsert(path: str, scans: int) -> int:
    """Load ``path`` into the crack database for the enabled schemes.

    Returns 1 if at least one carrier was inserted, 0 otherwise.  Mirrors
    doinsert()/process_loop() in stegbreak.c (the short-file ``-c`` convert path
    is omitted -- we always decode from the JPEG directly).
    """
    if not _has_extension(path, ".jpg") and not _has_extension(path, ".jpeg"):
        if convert:
            return 0
        for h in HANDLERS:
            if _has_extension(path, h["extension"]):
                reader = _READERS[h["read"].split("_")[0]]
                try:
                    obj = reader(path)
                except Exception as exc:  # noqa: BLE001
                    if not quiet:
                        sys.stderr.write(f"{path}: cannot read {h['extension']}: {exc}\n")
                    continue
                db.db_insert(path, h["type"], obj, h["crack"], path)
                return 1
        return 0

    res = 0
    for h in HANDLERS:
        if not (scans & h["type"]):
            continue
        reader = _READERS[h["read"].split("_")[0]]
        try:
            obj = reader(path)
        except Exception as exc:  # noqa: BLE001
            if not quiet:
                sys.stderr.write(f"{path}: cannot prepare {h['type']:#x}: {exc}\n")
            continue
        if obj is None:
            if not quiet:
                sys.stderr.write(f"{path}: not a {_scheme_name(h['type'])} carrier\n")
            continue
        db.db_insert(path, h["type"], obj, h["crack"], path)
        res = 1
    return res


# ---------------------------------------------------------------------------
# Wordlist crack loop
# ---------------------------------------------------------------------------
def do_wordlist_crack(wordlist_path: str) -> Optional[str]:
    ctx = rpp.RppContext()
    if rpp.rpp_init(ctx, rpp.SUBSECTION_WORDLIST):
        sys.stderr.write("No wordlist mode rules found in rules file\n")
        return None

    rule_count = rpp.rules_count(ctx, -1)

    eng = rules_mod.RulesEngine(16)
    last = " " * 17
    found_word: Optional[str] = None

    rule = rpp.rpp_next(ctx)
    while rule is not None:
        sub = rules_mod.rules_reject(rule)
        if sub is None:
            rule = rpp.rpp_next(ctx)
            continue

        if wordlist_path in (None, "-"):
            wf = sys.stdin
        else:
            try:
                wf = open(wordlist_path, "r", encoding="utf-8", errors="replace")
            except OSError as exc:
                sys.stderr.write(f"fopen: {wordlist_path}: {exc}\n")
                return found_word

        try:
            for raw in wf:
                line = raw.rstrip("\n").rstrip("\r")
                if not line:
                    continue
                if line.startswith("#") or line.startswith(";"):
                    if line.startswith("#!comment"):
                        continue
                    continue
                word = eng.apply(line, sub, -1)
                if word is None:
                    continue
                if word == last:
                    continue
                last = word
                hit = db.db_crack(word)
                if hit is not None:
                    entry, res = hit
                    found_word = word
                    _report_found(entry, word, res)
                    wf.close()
                    db.db_flush()
                    return found_word
        finally:
            if wf is not sys.stdin:
                wf.close()

        rule = rpp.rpp_next(ctx)

    db.db_flush()
    return found_word


def _scheme_name(type_: int) -> str:
    return {FLAG_DOJSTEG: "jsteg", FLAG_DOOUTGUESS: "outguess",
            FLAG_DOJPHIDE: "jphide"}.get(type_, "unknown")


def _report_found(entry, word: str, res) -> None:
    scheme = _scheme_name(entry.type)
    if entry.type == FLAG_DOJSTEG:
        header = res[1]
        extra = f" (header: {bytes(header).hex()})" if header else ""
        msg = f"{entry.filename}: {scheme} found \"{word}\"{extra}"
    elif entry.type == FLAG_DOJPHIDE:
        msg = f"{entry.filename}: {scheme} ({res}) found \"{word}\""
    else:
        payload = res
        extra = f" ({len(payload)} bytes)" if payload else ""
        msg = f"{entry.filename}: {scheme} found \"{word}\"{extra}"
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _default_rules_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, _DEFAULT_RULES)
    if os.path.exists(cand):
        return cand
    if os.path.exists(_DEFAULT_RULES):
        return _DEFAULT_RULES
    return cand


def main(argv=None) -> int:
    global quiet, convert

    parser = argparse.ArgumentParser(
        prog="stegbreak",
        description="Brute-force JSteg/OutGuess/JPHide steganographic JPEGs.")
    parser.add_argument("-V", action="store_true", help="print version and exit")
    parser.add_argument("-r", dest="rules", default=None, help="rules configuration file")
    parser.add_argument("-f", dest="wordlist", default=None, help="wordlist file")
    parser.add_argument("-t", dest="schemes", default=None,
                        help="schemes to crack: o=outguess, p=jphide, j=jsteg")
    parser.add_argument("-q", action="store_true", dest="quiet", help="quiet")
    parser.add_argument("-c", action="store_true", dest="convert",
                        help="convert mode (pre-process images)")
    parser.add_argument("files", nargs="*", help="JPEG files to crack")
    args = parser.parse_args(argv)

    if args.V:
        sys.stdout.write("Stegbreak Version 0.6 (Python port)\n")
        return 0

    quiet = args.quiet
    convert = args.convert

    # Faithful to stegbreak.c: scans starts at 0 and only falls back to all
    # schemes when -t is omitted entirely.
    scans = 0
    if args.schemes:
        for ch in args.schemes:
            if ch == "o":
                scans |= FLAG_DOOUTGUESS
            elif ch == "p":
                scans |= FLAG_DOJPHIDE
            elif ch == "j":
                scans |= FLAG_DOJSTEG
            else:
                sys.stderr.write(f"unknown scheme '{ch}'\n")
                return 1
    if not scans:
        scans = FLAG_DOOUTGUESS | FLAG_DOJPHIDE | FLAG_DOJSTEG

    rules_name = args.rules or _default_rules_path()
    wordlist = args.wordlist or _DEFAULT_WORDLIST

    if not convert:
        try:
            cfg.cfg_init(rules_name)
        except OSError as exc:
            sys.stderr.write(f"{exc}\n")
            return 1
        db.db_init()

    if not args.files:
        sys.stderr.write("usage: stegbreak [-V] [-r <rules>] [-f <wordlist>] "
                         "[-t <schemes>] file.jpg ...\n")
        return 1

    n = 0
    for path in args.files:
        if doinsert(path, scans):
            n += 1

    if not convert and n:
        start = time.time()
        found = do_wordlist_crack(wordlist)
        now = int(time.time() - start)
        sys.stderr.write(
            f"Processed {n} files. Time: {now}s\n")
        if found:
            return 0

    return 1 if not convert else 0


if __name__ == "__main__":
    raise SystemExit(main())
