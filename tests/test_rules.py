"""Tests for the John-the-Ripper style rules engine (rules.py).

These assert the documented/expected behaviour of each rule command, derived from
the original rules.c switch statement.  The conversion-table constants are also
checked character-for-character against the C CONV_* macros to catch any
transcription error, since there is no C toolchain here to diff against.
"""

from stegdetect_py.rules import (
    RulesEngine, expand_rules, rules_reject, RULE_WORD_SIZE,
    CONV_SOURCE, CONV_SHIFT, CONV_INVERT, CONV_VOWELS, CONV_RIGHT, CONV_LEFT,
)


# --- The C CONV_* literals, copied verbatim from rules.c for comparison ------
C_CONV_SOURCE = ("`1234567890-=\\qwertyuiop[]asdfghjkl;'zxcvbnm,./"
                 "~!@#$%^&*()_+|QWERTYUIOP{}ASDFGHJKL:\"ZXCVBNM<>?")
C_CONV_SHIFT = ("~!@#$%^&*()_+|QWERTYUIOP{}ASDFGHJKL:\"ZXCVBNM<>?"
               "`1234567890-=\\qwertyuiop[]asdfghjkl;'zxcvbnm,./")
C_CONV_INVERT = ("`1234567890-=\\QWERTYUIOP[]aSDFGHJKL;'ZXCVBNM,./"
                "~!@#$%^&*()_+|qwertyuiop{}asdfghjkl:\"zxcvbnm<>?")
C_CONV_VOWELS = ("`1234567890-=\\QWeRTYuioP[]aSDFGHJKL;'ZXCVBNM,./"
                "~!@#$%^&*()_+|QWeRTYuioP{}aSDFGHJKL:\"ZXCVBNM<>?")
C_CONV_RIGHT = ("1234567890-=\\\\wertyuiop[]]sdfghjkl;''xcvbnm,./\\"
               "!@#$%^&*()_+||WERTYUIOP{}}SDFGHJKL:\"\"XCVBNM<>?|")
C_CONV_LEFT = ("``1234567890-=qqwertyuiop[aasdfghjkl;zzxcvbnm,."
              "~~!@#$%^&*()_+QQWERTYUIOP{AASDFGHJKL:ZZXCVBNM<>")


def test_conv_tables_match_c_source():
    assert CONV_SOURCE == C_CONV_SOURCE
    assert CONV_SHIFT == C_CONV_SHIFT
    assert CONV_INVERT == C_CONV_INVERT
    assert CONV_VOWELS == C_CONV_VOWELS
    assert CONV_RIGHT == C_CONV_RIGHT
    assert CONV_LEFT == C_CONV_LEFT


def test_basic_case_and_reverse():
    e = RulesEngine()
    assert e.apply("AbC", "l") == "abc"
    assert e.apply("AbC", "u") == "ABC"
    assert e.apply("hello", "r") == "olleh"
    assert e.apply("hello", "d") == "hellohello"
    assert e.apply("ab", "f") == "abba"


def test_capitalize():
    e = RulesEngine()
    assert e.apply("hello", "c") == "Hello"
    assert e.apply("HELLO", "c") == "Hello"
    assert e.apply("mcdonald", "c") == "McDonald"
    assert e.apply("hello", "C") == "hELLO"
    assert e.apply("mcdonald", "C") == "mCdONALD"


def test_append_prepend():
    e = RulesEngine()
    assert e.apply("foo", "$!") == "foo!"
    assert e.apply("foo", "$1") == "foo1"
    assert e.apply("foo", "^#") == "#foo"


def test_extract_insert_overwrite():
    e = RulesEngine()
    assert e.apply("abcdef", "x12") == "bc"
    assert e.apply("abcdef", "x03") == "abc"
    assert e.apply("abc", "i1Z") == "aZbc"
    assert e.apply("abc", "i0Z") == "Zabc"
    assert e.apply("abc", "i9Z") == "abcZ"
    assert e.apply("abc", "o1Z") == "aZc"


def test_substitute_purge():
    e = RulesEngine()
    # substitute all lowercase with 'a'
    assert e.apply("hello", "s?la") == "aaaaa"
    # substitute literal 'o' with 'O'
    assert e.apply("hello", "soO") == "hellO"
    # @ purges (removes) chars of the class; an empty result is rejected
    # (the C rejects any rule whose output becomes empty).
    assert e.apply("hello", "@?l") is None
    assert e.apply("HeLLo", "@?u") == "eo"


def test_rejects():
    e = RulesEngine()
    # ! rejects if a char matches
    assert e.apply("hello", "!?d") == "hello"
    assert e.apply("abc123", "!?d") is None
    # / rejects if NO char matches
    assert e.apply("HeLLo", "/?l") == "HeLLo"
    assert e.apply("HELLO", "/?l") is None
    # = rejects unless char at pos matches
    assert e.apply("Hello", "=0?u") == "Hello"
    assert e.apply("hello", "=0?u") is None
    # ( rejects unless the FIRST char matches
    assert e.apply("Hello", "(?u") == "Hello"
    assert e.apply("hello", "(?u") is None
    # ) rejects unless the LAST char matches
    assert e.apply("Hello", ")?l") == "Hello"
    assert e.apply("HellO", ")?l") is None


def test_truncation_and_deletion():
    e = RulesEngine()
    assert e.apply("abcdef", "[") == "bcdef"
    assert e.apply("abcdef", "]") == "abcde"
    assert e.apply("abcdef", "'3") == "abc"
    assert e.apply("a1b2c3", "%3?d") == "a1b2c3"
    assert e.apply("a1b2c3", "%4?d") is None
    assert e.apply("abc", "D1") == "ac"


def test_rotate_and_convs():
    e = RulesEngine()
    assert e.apply("abc", "{") == "bca"
    assert e.apply("abc", "}") == "cab"
    # invert: verbs 'b'/'B' toggle; non-letters unchanged
    assert e.apply("b", "t") == "B"
    assert e.apply("B", "t") == "b"
    assert e.apply("a1", "t") == "a1"
    assert e.apply("b", "T0") == "B"


def test_plural_past_ing():
    e = RulesEngine()
    assert e.apply("cat", "p") == "cats"
    assert e.apply("bus", "p") == "buses"
    assert e.apply("box", "p") == "boxes"
    assert e.apply("baby", "p") == "babies"
    assert e.apply("elf", "p") == "elves"
    assert e.apply("life", "p") == "lives"
    # past
    assert e.apply("walk", "P") == "walked"
    assert e.apply("bake", "P") == "baked"
    assert e.apply("try", "P") == "tried"
    assert e.apply("bog", "P") == "bogged"
    assert e.apply("play", "P") == "plaied"
    # ing
    assert e.apply("walk", "I") == "walking"
    assert e.apply("make", "I") == "making"
    assert e.apply("bog", "I") == "bogging"
    assert e.apply("play", "I") == "playing"


def test_memory_rules():
    e = RulesEngine()
    # M stores the word; Q rejects if the current == stored
    assert e.apply("foo", "M Q") is None
    # transform then Q: keep since transformed != stored ('l' changes "BAR")
    assert e.apply("BAR", "M l Q") == "bar"
    # M with no following Q is a no-op
    assert e.apply("bar", "M") == "bar"


def test_length_codes_and_rejects():
    e = RulesEngine()
    # '>' keeps only if length > pos; '<' keeps only if length < pos
    assert e.apply("abc", ">2") == "abc"      # len 3 > 2
    assert e.apply("ab", ">2") is None        # len 2 not > 2
    assert e.apply("ab", "<3") == "ab"        # len 2 < 3
    assert e.apply("abcd", "<3") is None      # len 4 not < 3


def test_single_crack_split():
    e = RulesEngine()
    # split < 0 -> single-crack commands error out
    assert e.apply("user:pass", "1") is None
    assert e.apply("user:pass", "2") is None
    assert e.apply("user:pass", "+") is None
    # with a real split point, the rule recombines the two halves; '1'/'2'
    # alone are identity (left+right == original, separator stays on the right)
    assert e.apply("user:pass", "1", 5) == "user:pass"
    assert e.apply("user:pass", "2", 5) == "user:pass"
    # transform applied to the LEFT half after '1', then recombined with right
    assert e.apply("user:pass", "1 c", 5) == "User:pass"
    assert e.apply("user:pass", "2 c", 5) == "user:Pass"
    # 'l' transforms the whole word, then '1' resets to the original halves
    assert e.apply("user:pass", "l1", 5) == "user:pass"


def test_rules_reject_filter():
    assert rules_reject("l") == "l"
    assert rules_reject("-c") == "c"
    assert rules_reject("-cl") == "l"
    assert rules_reject("-x") is None
    assert rules_reject("") == ""
    assert rules_reject("::") == ":"


def test_expand_rules_integration():
    # expand_rules applies every rule line, ignoring comments and blank lines,
    # and drops reject-only rules.
    rules_text = """
# this is a comment
l
u
; another comment
#!comment something
-c
foobar-not-a-rule
"""
    out = expand_rules("hello", rules_text)
    # 'l' -> hello, 'u' -> HELLO, '-c' -> 'c' (capitalize) -> Hello
    assert "hello" in out
    assert "HELLO" in out
    assert "Hello" in out
    # the invalid '-x' style rule is skipped (here '-c' is accepted as 'c')
    assert all(isinstance(x, str) and x for x in out)


def test_max_length_truncation():
    e = RulesEngine(max_length=4)
    # a long duplicate is truncated to max_length at the end
    assert e.apply("abcdef", "d") == "abcd"
    e2 = RulesEngine(max_length=16)
    assert e2.apply("x" * 100, "d") == "x" * 16
