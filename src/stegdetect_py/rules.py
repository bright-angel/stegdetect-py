"""
John-the-Ripper style rules engine -- faithful Python port of rules.c.

This expands a wordlist entry into many candidate passwords by applying a list of
rules.  Each rule is a string of single-character commands (lower-case, upper-case,
reverse, append, insert, class substitution, rejects, memory, ...).  The full
command set from the C source is implemented, including the character/class
conversion tables and the position/length codes.

The C source relies on three macros:
    #define RULE  (*rule++)   -- consume the next rule char
    #define LAST  (*(rule-1)) -- the char just consumed
    #define NEXT  (*rule)      -- peek at the next rule char (no consume)
We model them with ``nxt()`` (consume) and ``peek()`` (no consume).  Rejecting a
candidate returns ``None`` from ``apply``.
"""

from __future__ import annotations

from typing import List, Optional

RULE_WORD_SIZE = 128
RULE_BUF = RULE_WORD_SIZE * 2          # size of the in/out buffers in C
INVALID_LENGTH = 0x7F

# Conversion source keyboard layout (the C CONV_* strings).
CONV_SOURCE = ("`1234567890-=\\qwertyuiop[]asdfghjkl;'zxcvbnm,./"
               "~!@#$%^&*()_+|QWERTYUIOP{}ASDFGHJKL:\"ZXCVBNM<>?")
CONV_SHIFT = ("~!@#$%^&*()_+|QWERTYUIOP{}ASDFGHJKL:\"ZXCVBNM<>?"
             "`1234567890-=\\qwertyuiop[]asdfghjkl;'zxcvbnm,./")
CONV_INVERT = ("`1234567890-=\\QWERTYUIOP[]aSDFGHJKL;'ZXCVBNM,./"
              "~!@#$%^&*()_+|qwertyuiop{}asdfghjkl:\"zxcvbnm<>?")
CONV_VOWELS = ("`1234567890-=\\QWeRTYuioP[]aSDFGHJKL;'ZXCVBNM,./"
              "~!@#$%^&*()_+|QWeRTYuioP{}aSDFGHJKL:\"ZXCVBNM<>?")
CONV_RIGHT = ("1234567890-=\\\\wertyuiop[]]sdfghjkl;''xcvbnm,./\\"
             "!@#$%^&*()_+||WERTYUIOP{}}SDFGHJKL:\"\"XCVBNM<>?|")
CONV_LEFT = ("``1234567890-=qqwertyuiop[aasdfghjkl;zzxcvbnm,."
            "~~!@#$%^&*()_+QQWERTYUIOP{AASDFGHJKL:ZZXCVBNM<>")

CHARS_LOWER = "abcdefghijklmnopqrstuvwxyz"
CHARS_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHARS_DIGITS = "0123456789"

# Error codes.
RULES_ERROR_NONE = 0
RULES_ERROR_END = 1
RULES_ERROR_UNKNOWN = 2
RULES_ERROR_POSITION = 3
RULES_ERROR_CLASS = 4
RULES_ERROR_REJECT = 5
RULES_ERRORS = [None, "Unexpected end of rule", "Unknown command",
                "Invalid position code", "Unknown character class code",
                "Unknown rule reject flag"]


def _build_conv(src: str, dst: str) -> List[int]:
    conv = list(range(0x100))
    for s, d in zip(src, dst):
        conv[ord(s)] = ord(d)
    return conv


class RulesEngine:
    def __init__(self, max_length: int = RULE_WORD_SIZE - 1):
        self.max_length = max_length
        self.classes = [None] * 0x100
        self.length = [INVALID_LENGTH] * 0x100
        self._init_classes()
        self._init_convs()
        self._init_length(max_length)
        self.debug = 0
        self.errno = RULES_ERROR_NONE

    # ---- tables -----------------------------------------------------------
    def _init_class(self, name: str, valid: str) -> None:
        tbl = [0] * 0x100
        for c in valid:
            tbl[ord(c)] = 1
        self.classes[ord(name)] = tbl
        if "a" <= name <= "z":
            inv = chr(ord(name) & ~0x20)
            itbl = [1] * 0x100
            for c in valid:
                itbl[ord(c)] = 0
            self.classes[ord(inv)] = itbl

    def _init_classes(self) -> None:
        self._init_class("?", "?")
        self._init_class("v", "aeiouAEIOU")
        self._init_class("c", "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ")
        self._init_class("w", " \t")
        self._init_class("p", ".,:;'\"?!`")
        self._init_class("s", "$%^&*()-_+=|\\<>[]{}#@/~")
        self._init_class("l", CHARS_LOWER)
        self._init_class("u", CHARS_UPPER)
        self._init_class("d", CHARS_DIGITS)
        self._init_class("a", CHARS_LOWER + CHARS_UPPER)
        self._init_class("x", CHARS_LOWER + CHARS_UPPER + CHARS_DIGITS)

    def _init_convs(self) -> None:
        self.conv_source = list(range(0x100))
        self.conv_shift = _build_conv(CONV_SOURCE, CONV_SHIFT)
        self.conv_invert = _build_conv(CONV_SOURCE, CONV_INVERT)
        self.conv_vowels = _build_conv(CONV_SOURCE, CONV_VOWELS)
        self.conv_right = _build_conv(CONV_SOURCE, CONV_RIGHT)
        self.conv_left = _build_conv(CONV_SOURCE, CONV_LEFT)
        self.conv_tolower = _build_conv(CHARS_UPPER, CHARS_LOWER)
        self.conv_toupper = _build_conv(CHARS_LOWER, CHARS_UPPER)

    def _init_length(self, max_length: int) -> None:
        self.length = [INVALID_LENGTH] * 0x100
        for c in range(ord("0"), ord("9") + 1):
            self.length[c] = c - ord("0")
        for c in range(ord("a"), ord("z") + 1):
            self.length[c] = c - (ord("a") - 10)
        for c in range(ord("A"), ord("Z") + 1):
            self.length[c] = c - (ord("A") - 10)
        self.length[ord("*")] = max_length
        self.length[ord("-")] = max_length - 1
        self.length[ord("+")] = max_length + 1

    # ---- helpers ----------------------------------------------------------
    def _conv(self, conv, s: str) -> str:
        return "".join(chr(conv[ord(c)]) for c in s)

    def _class_match(self, charclass: str, c: str) -> bool:
        if charclass.startswith("?"):
            cls = self.classes[ord(charclass[1])]
            if cls is None:
                return False
            return cls[ord(c)] == 1
        return c == charclass[0]

    # ---- core -------------------------------------------------------------
    def apply(self, word: str, rule: str, split: int = -1) -> Optional[str]:
        """Apply a single rule to ``word``; return the result or None on reject."""
        self.errno = RULES_ERROR_NONE
        s = word[:RULE_WORD_SIZE]
        memory_empty = True
        memory = ""
        which = 0
        buf2 = ""
        rule = rule or ""

        ri = 0
        n = len(rule)

        def nxt():
            nonlocal ri
            if ri >= n:
                return "\0"
            ch = rule[ri]
            ri += 1
            return ch

        def peek():
            if ri >= n:
                return "\0"
            return rule[ri]

        def position():
            # Mirror the POSITION macro: consume one char, look up its length.
            c = nxt()
            pos = self.length[ord(c)]
            if pos == INVALID_LENGTH:
                self.errno = RULES_ERROR_POSITION
                return None
            return pos

        while ri < n:
            if not s:
                if self.debug:
                    self.errno = RULES_ERROR_END
                return None
            cmd = nxt()
            if cmd in (":", " ", "\t"):
                continue
            if cmd == "<":
                pos = position()
                if pos is None:
                    return None
                if len(s) < pos:
                    pass          # keep
                else:
                    return None   # reject
                continue
            if cmd == ">":
                pos = position()
                if pos is None:
                    return None
                if len(s) > pos:
                    pass
                else:
                    return None
                continue
            if cmd == "l":
                s = self._conv(self.conv_tolower, s)[:RULE_BUF]
                continue
            if cmd == "u":
                s = self._conv(self.conv_toupper, s)[:RULE_BUF]
                continue
            if cmd == "c":
                s = self._cap(s, False)
                continue
            if cmd == "C":
                s = self._cap(s, True)
                continue
            if cmd == "r":
                s = s[::-1]
                continue
            if cmd == "d":
                s = (s + s)[:RULE_BUF]
                continue
            if cmd == "f":
                s = (s + s[::-1])[:RULE_BUF]
                continue
            if cmd == "p":
                s = self._plural(s)[:RULE_BUF]
                continue
            if cmd == "$":
                v = nxt()
                if v == "\0":
                    self.errno = RULES_ERROR_END
                    return None
                s = (s + v)[:RULE_BUF]
                continue
            if cmd == "^":
                v = nxt()
                if v == "\0":
                    self.errno = RULES_ERROR_END
                    return None
                s = (v + s)[:RULE_BUF]
                continue
            if cmd == "x":
                p1 = position()
                if p1 is None:
                    return None
                p2 = position()
                if p2 is None:
                    return None
                if p1 < len(s):
                    s = s[p1:p1 + p2]
                else:
                    s = ""
                continue
            if cmd == "i":
                pos = position()
                if pos is None:
                    return None
                v = nxt()
                if v == "\0":
                    self.errno = RULES_ERROR_END
                    return None
                if pos < len(s):
                    s = (s[:pos] + v + s[pos:])[:RULE_BUF]
                else:
                    s = (s + v)[:RULE_BUF]
                continue
            if cmd == "o":
                pos = position()
                if pos is None:
                    return None
                v = nxt()
                if v == "\0":
                    self.errno = RULES_ERROR_END
                    return None
                if pos < len(s):
                    s = s[:pos] + v + s[pos + 1:]
                continue
            if cmd == "s":
                spec = nxt()
                if spec == "\0":
                    self.errno = RULES_ERROR_END
                    return None
                if spec == "?":
                    clsname = nxt()
                    if clsname == "\0":
                        self.errno = RULES_ERROR_END
                        return None
                    charclass = "?" + clsname
                else:
                    charclass = spec
                repl = peek()          # NEXT: do not consume yet
                if repl == "\0":
                    self.errno = RULES_ERROR_END
                    return None
                s = "".join(repl if self._class_match(charclass, c) else c
                            for c in s)
                nxt()                  # VALUE: consume the replacement char
                continue
            if cmd == "@":
                spec = nxt()
                if spec == "\0":
                    self.errno = RULES_ERROR_END
                    return None
                if spec == "?":
                    clsname = nxt()
                    if clsname == "\0":
                        self.errno = RULES_ERROR_END
                        return None
                    cls = "?" + clsname
                else:
                    cls = spec
                s = "".join(c for c in s if not self._class_match(cls, c))
                continue
            if cmd == "!":
                spec = nxt()
                if spec == "\0":
                    self.errno = RULES_ERROR_END
                    return None
                if spec == "?":
                    clsname = nxt()
                    if clsname == "\0":
                        self.errno = RULES_ERROR_END
                        return None
                    cls = "?" + clsname
                else:
                    cls = spec
                if any(self._class_match(cls, c) for c in s):
                    return None
                continue
            if cmd == "/":
                spec = nxt()
                if spec == "\0":
                    self.errno = RULES_ERROR_END
                    return None
                if spec == "?":
                    clsname = nxt()
                    if clsname == "\0":
                        self.errno = RULES_ERROR_END
                        return None
                    cls = "?" + clsname
                else:
                    cls = spec
                if not any(self._class_match(cls, c) for c in s):
                    return None
                continue
            if cmd == "=":
                pos = position()
                if pos is None:
                    return None
                spec = nxt()
                if spec == "\0":
                    self.errno = RULES_ERROR_END
                    return None
                if pos >= len(s):
                    if spec == "?":
                        nxt()         # SKIP_CLASS
                    return None
                if spec == "?":
                    clsname = nxt()
                    if clsname == "\0":
                        self.errno = RULES_ERROR_END
                        return None
                    cls = "?" + clsname
                    if not self._class_match(cls, s[pos]):
                        return None
                else:
                    if s[pos] != spec:
                        return None
                continue
            if cmd == "[":
                s = s[1:] if s else ""
                continue
            if cmd == "]":
                s = s[:-1] if s else ""
                continue
            if cmd == "t":
                s = self._conv(self.conv_invert, s)
                continue
            if cmd == "T":
                pos = position()
                if pos is None:
                    return None
                if pos < len(s):
                    s = s[:pos] + chr(self.conv_invert[ord(s[pos])]) + s[pos + 1:]
                continue
            if cmd == "(":
                spec = nxt()
                if spec == "\0":
                    self.errno = RULES_ERROR_END
                    return None
                if spec == "?":
                    clsname = nxt()
                    if clsname == "\0":
                        self.errno = RULES_ERROR_END
                        return None
                    cls = "?" + clsname
                else:
                    cls = spec
                if not s or not self._class_match(cls, s[0]):
                    return None
                continue
            if cmd == ")":
                spec = nxt()
                if spec == "\0":
                    self.errno = RULES_ERROR_END
                    return None
                if spec == "?":
                    clsname = nxt()
                    if clsname == "\0":
                        self.errno = RULES_ERROR_END
                        return None
                    cls = "?" + clsname
                else:
                    cls = spec
                if not s or not self._class_match(cls, s[-1]):
                    return None
                continue
            if cmd == "'":
                pos = position()
                if pos is None:
                    return None
                s = s[:pos]
                continue
            if cmd == "%":
                pos = position()
                if pos is None:
                    return None
                spec = nxt()
                if spec == "\0":
                    self.errno = RULES_ERROR_END
                    return None
                if spec == "?":
                    clsname = nxt()
                    if clsname == "\0":
                        self.errno = RULES_ERROR_END
                        return None
                    cls = "?" + clsname
                else:
                    cls = spec
                count = sum(1 for c in s if self._class_match(cls, c))
                if count < pos:
                    return None
                continue
            if cmd == "D":
                pos = position()
                if pos is None:
                    return None
                if pos < len(s):
                    s = s[:pos] + s[pos + 1:]
                continue
            if cmd == "{":
                s = (s[1:] + s[0]) if s else ""
                continue
            if cmd == "}":
                s = (s[-1] + s[:-1]) if s else ""
                continue
            if cmd == "S":
                s = self._conv(self.conv_shift, s)
                continue
            if cmd == "V":
                s = self._conv(self.conv_vowels, s)
                continue
            if cmd == "R":
                s = self._conv(self.conv_right, s)
                continue
            if cmd == "L":
                s = self._conv(self.conv_left, s)
                continue
            if cmd == "P":
                s = self._past(s)[:RULE_BUF]
                continue
            if cmd == "I":
                s = self._ing(s)[:RULE_BUF]
                continue
            if cmd == "M":
                memory = s[:self.max_length]
                memory_empty = False
                continue
            if cmd == "Q":
                ref = memory if not memory_empty else word
                if ref[:self.max_length] == s[:self.max_length]:
                    return None
                continue
            if cmd in ("1", "2", "+"):
                res = self._single_crack(cmd, s, word, split, buf2, which)
                if res is None:
                    return None
                s, buf2, which = res
                continue
            # Unknown command.
            self.errno = RULES_ERROR_UNKNOWN
            return None

        if not s:
            return None
        # End-of-rule single-crack recombination (the C switch on `which`).
        if which == 1:
            s = (s + buf2)[:RULE_BUF]
        elif which == 2:
            s = (buf2 + s)[:RULE_BUF]
        s = s[:self.max_length]
        return s

    def _cap(self, s: str, inverse: bool) -> str:
        if not s:
            return s
        if inverse:
            out = s[0].lower() + "".join(c.upper() for c in s[1:])
            if out[:2] == "mC":
                out = out[:2] + out[2].lower() + out[3:]
        else:
            out = s[0].upper() + "".join(c.lower() for c in s[1:])
            if out[:2] == "Mc":
                out = out[:2] + out[2].upper() + out[3:]
        return out

    def _plural(self, s: str) -> str:
        if not s or len(s) < 2:
            return s
        last = s[-1]
        if last in "hsx":
            return s + "es"
        if last == "f" and s[-2] != "f":
            return s[:-1] + "ves"
        if len(s) > 2 and last == "e" and s[-2] == "f":
            return s[:-2] + "ves"
        if len(s) > 2 and last == "y":
            if s[-2] in "aeiou":
                return s + "s"
            return s[:-1] + "ies"
        return s + "s"

    def _past(self, s: str) -> str:
        if not s or len(s) < 3:
            return s
        last = s[-1]
        if last == "d" and s[-2] == "e":
            return s
        if last == "y":
            return s[:-1] + "ied"
        if last in "bgp" and s[-2] not in "bgp":
            return s + last + "ed"
        if last == "e":
            return s + "d"
        return s + "ed"

    def _ing(self, s: str) -> str:
        if not s or len(s) < 3:
            return s
        last = s[-1]
        if last == "g" and s[-2] == "n" and s[-3] == "i":
            return s
        if last in "aeiou":
            return s[:-1] + "ing"
        if last in "bgp" and s[-2] not in "bgp":
            return s + last + "ing"
        return s + "ing"

    def _single_crack(self, cmd, s, word, split, buf2, which):
        if split < 0:
            self.errno = RULES_ERROR_UNKNOWN
            return None
        if cmd == "1":
            if not split:
                return None
            if which:
                buf2 = s
            else:
                buf2 = word[split:]
            s = word[:split]
            which = 1
            return s, buf2, which
        if cmd == "2":
            if not split:
                return None
            if which:
                buf2 = s
            else:
                buf2 = word[:split]
            s = word[split:]
            which = 2
            return s, buf2, which
        if cmd == "+":
            if which == 1:
                s = (s + buf2)[:RULE_BUF]
            elif which == 2:
                s = (buf2 + s)[:RULE_BUF]
            else:
                self.errno = RULES_ERROR_UNKNOWN
                return None
            which = 0
            return s, buf2, which
        return None


def rules_reject(rule: str) -> Optional[str]:
    """Mirror rules_reject(rule, NULL): return the (sub)rule to apply, or None.

    A leading ``-`` followed by ``c`` / ``8`` / ``s`` (and the char after it) is
    stripped; a leading ``-`` with any other char makes the whole rule invalid;
    an empty or all-skipped rule returns the (possibly empty) remainder so it can
    be applied as a no-op.  Mirrors the C pointer arithmetic (``rule - 1``).
    """
    if rule is None:
        return None
    i = 0
    n = len(rule)
    while i < n:
        c = rule[i]
        if c in (":", " ", "\t"):
            i += 1
            continue
        if c == "-":
            if i + 1 >= n:
                return None
            nxt = rule[i + 1]
            if nxt in ("c", "8", "s"):
                i += 2
                continue
            return None
        # default: return the rule from this command onward.
        return rule[i:]
    return rule[i - 1:] if i > 0 else ""


def expand_rules(word: str, rules_text: str, max_length: int = 16,
                 split: int = -1) -> List[str]:
    """Apply every non-empty rule line in ``rules_text`` to ``word``.

    Lines beginning with ``#`` or ``;`` and the ``#!comment`` pragma are ignored,
    matching stegbreak's wordlist parsing.  Each surviving rule is first passed
    through :func:`rules_reject` (which drops reject-only rules).
    """
    eng = RulesEngine(max_length)
    out = []
    for line in rules_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if line.startswith("#!comment"):
            continue
        sub = rules_reject(line)
        if sub is None:
            continue
        res = eng.apply(word, sub, split)
        if res:
            out.append(res)
    return out
