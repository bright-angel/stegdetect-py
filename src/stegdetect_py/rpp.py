"""
Rule preprocessor -- faithful Python port of rpp.c.

This expands the rule list stored in the config file into individual rules.  A
rule may contain ``[...]`` character ranges (with ``-`` for ranges and ``\\`` for
literals); each range is expanded into the cartesian product of its characters,
so ``l[0-9]`` becomes ``l0 l1 ... l9`` and ``l[a-z]`` becomes 26 rules.

The section a rule list is read from is ``SECTION_RULES`` + ``SUBSECTION_*``
(e.g. ``[List.Rules:Wordlist]``).
"""

from __future__ import annotations

from typing import List, Optional

import stegdetect_py.cfg as cfg

ARCH_BITS = 8
RULE_BUFFER_SIZE = 128
RULE_RANGES_MAX = 16

SECTION_RULES = "list.rules"
SUBSECTION_WORDLIST = ":wordlist"


class RppRange:
    __slots__ = ("pos", "index", "count", "chars", "_seen")

    def __init__(self, pos: int):
        self.pos = pos          # index into the output buffer
        self.index = 0          # current selection
        self.count = 0          # number of chars
        self.chars: List[str] = []
        self._seen = set()      # dedupe mask

    def add(self, c: str) -> None:
        if c in self._seen:
            return
        self._seen.add(c)
        self.chars.append(c)
        self.count += 1


class RppContext:
    __slots__ = ("input", "output", "count", "ranges")

    def __init__(self):
        self.input = None       # current CfgLine node
        self.output = ""
        self.count = -1         # -1 means "must process the next input rule"
        self.ranges: List[RppRange] = []


def rpp_init(ctx: RppContext, subsection: Optional[str]) -> int:
    lst = cfg.cfg_get_list(SECTION_RULES, subsection)
    if lst is not None and lst.head is not None:
        ctx.input = lst.head
        ctx.count = -1
        ctx.ranges = []
        return 0
    return 1


def _rpp_add_char(rng: RppRange, c: str) -> None:
    rng.add(c)


def rpp_process_rule(ctx: RppContext) -> None:
    inp = ctx.input.data
    output: List[str] = []
    ctx.ranges = []
    ctx.count = 0
    c1 = ""
    i = 0
    n = len(inp)
    while i < n and len(output) < RULE_BUFFER_SIZE - 1:
        ch = inp[i]
        if ch == "\\":
            i += 1
            if i < n:
                output.append(inp[i])
                i += 1
            continue
        if ch == "[":
            if ctx.count >= RULE_RANGES_MAX:
                output.append(ch)
                i += 1
                continue
            i += 1  # consume '['
            rng = RppRange(len(output))
            output.append("\0")  # placeholder filled in by rpp_next
            ctx.ranges.append(rng)
            ctx.count += 1
            while i < n and inp[i] != "]":
                c = inp[i]
                if c == "\\":
                    i += 1
                    if i < n:
                        c1 = inp[i]
                        _rpp_add_char(rng, c1)
                        i += 1
                elif c == "-":
                    i += 1
                    if i < n:
                        c2 = inp[i]
                        if c1 and rng.count:
                            if ord(c1) > ord(c2):
                                for code in range(ord(c1) - 1, ord(c2) - 1, -1):
                                    _rpp_add_char(rng, chr(code))
                            else:
                                for code in range(ord(c1) + 1, ord(c2) + 1):
                                    _rpp_add_char(rng, chr(code))
                        c1 = c2
                        i += 1
                    else:
                        i += 1
                else:
                    c1 = c
                    _rpp_add_char(rng, c1)
                    i += 1
            if i < n:
                i += 1  # consume ']'
            continue
        output.append(ch)
        i += 1
    ctx.output = "".join(output)


def rpp_next(ctx: RppContext) -> Optional[str]:
    # Faithful mirror of the C rpp_next():
    #   int index;
    #   if (ctx->count < 0) { if (!ctx->input) return NULL; rpp_process_rule(ctx); }
    #   if ((index = ctx->count - 1) >= 0) {
    #       do { ...fill output... } while (index--);
    #       index = ctx->count - 1;
    #       do { ...advance odometer (carry)... } while (index--);
    #   }
    #   if (index < 0) { ctx->input = ctx->input->next; ctx->count = -1; }
    #   return ctx->output;
    # Note: for a rule with no ranges, count == 0 so index == -1; the body is
    # skipped and the rule is returned as-is exactly once (then count is reset to
    # -1 so the next call processes the following input line).
    if ctx.count < 0:
        if ctx.input is None:
            return None
        rpp_process_rule(ctx)

    index = ctx.count - 1
    if index >= 0:
        # place the current selection of every range into the output buffer
        out = list(ctx.output)
        i = index
        while True:
            rng = ctx.ranges[i]
            out[rng.pos] = rng.chars[rng.index]
            if i <= 0:
                break
            i -= 1
        ctx.output = "".join(out)

        # advance the odometer (least-significant range first, carry left)
        index = ctx.count - 1
        while True:
            rng = ctx.ranges[index]
            rng.index += 1
            if rng.index < rng.count:
                break
            rng.index = 0
            if index <= 0:
                index = -1
                break
            index -= 1

    if index < 0:
        ctx.input = ctx.input.next
        ctx.count = -1

    return ctx.output


def rules_count(ctx: RppContext, split: int = -1) -> int:
    """Count the number of usable (non-reject) rules; rewinds the context."""
    from stegdetect_py.rules import rules_reject, RulesEngine

    eng = RulesEngine(16)
    count = 0
    while True:
        rule = rpp_next(ctx)
        if rule is None:
            break
        sub = rules_reject(rule)
        if sub is None:
            continue
        if eng.apply("", sub, split) is not None:
            count += 1
    rpp_init(ctx, SUBSECTION_WORDLIST)
    return count
