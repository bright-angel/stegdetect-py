"""
In-memory database of target files -- faithful Python port of db.c.

Each ``db_insert`` registers a prepared carrier image together with the scheme
type and the matching ``crack_*`` function.  ``db_crack(word)`` tries the given
candidate password against every registered carrier and returns the first hit.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

FLAG_DOJPHIDE = 0x0002
FLAG_DOJSTEG = 0x0004
FLAG_DOOUTGUESS = 0x0001


class DBEntry:
    __slots__ = ("filename", "type", "obj", "crack_fn", "name")

    def __init__(self, filename, type_, obj, crack_fn, name):
        self.filename = filename
        self.type = type_
        self.obj = obj
        self.crack_fn = crack_fn          # (filename, word, obj) -> result
        self.name = name


_entries: List[DBEntry] = []


def db_init() -> None:
    _entries.clear()


def db_insert(filename, type_, obj, crack_fn: Callable, name: str = "") -> None:
    _entries.append(DBEntry(filename, type_, obj, crack_fn, name))


def _is_success(type_: int, res: Any) -> bool:
    if type_ == FLAG_DOJSTEG:
        # crack_jsteg returns (found: bool, header)
        return bool(res and res[0])
    # crack_outguess returns payload or None; crack_jphide returns 'v3'/'v5'/None
    return res is not None


def db_crack(word: str) -> Optional[Tuple[DBEntry, Any]]:
    for entry in _entries:
        res = entry.crack_fn(entry.filename, word, entry.obj)
        if _is_success(entry.type, res):
            return entry, res
    return None


def db_flush() -> None:
    # The C version flushes pending database writes; nothing persistent here.
    pass


def db_count() -> int:
    return len(_entries)
