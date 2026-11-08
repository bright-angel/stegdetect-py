"""
Config-file parser -- faithful Python port of cfg.c.

This is the John-the-Ripper style configuration reader used by stegbreak to load
its rule sets.  The file format is an INI dialect:

    [Section.Name]            # section header (case-insensitive, stored lower)
    key = value               # parameter
    [List.Section]            # a section whose body is a list of lines
    some rule line            # list entry

Sections whose name begins with ``list.`` collect their (non-comment, non-blank)
body lines into an ordered list; all other sections collect ``key = value``
parameters.
"""

from __future__ import annotations

from typing import Dict, List, Optional


def strlwr(s: str) -> str:
    return s.lower()


def trim(s: str) -> str:
    s = s.lstrip(" \t")
    if not s:
        return s
    end = len(s) - 1
    while end >= 0 and s[end] in (" ", "\t"):
        end -= 1
    return s[:end + 1]


class CfgParam:
    __slots__ = ("name", "value")

    def __init__(self, name: str, value: str):
        self.name = name
        self.value = value


class CfgLine:
    __slots__ = ("data", "number", "next")

    def __init__(self, data: str, number: int):
        self.data = data
        self.number = number
        self.next = None


class CfgList:
    __slots__ = ("head", "tail")

    def __init__(self):
        self.head = None
        self.tail = None

    def items(self) -> List[str]:
        out = []
        node = self.head
        while node:
            out.append(node.data)
            node = node.next
        return out


class CfgSection:
    __slots__ = ("name", "params", "list", "next")

    def __init__(self, name: str):
        self.name = name
        self.params: Dict[str, str] = {}
        self.list: Optional[CfgList] = None
        self.next = None


class CfgDB:
    """In-memory store of parsed sections (mirrors the C ``cfg_database``)."""

    def __init__(self):
        self.sections: List[CfgSection] = []
        self._current: Optional[CfgSection] = None

    # -- building ----------------------------------------------------------
    def add_section(self, name: str) -> CfgSection:
        sec = CfgSection(strlwr(name))
        sec.next = self._current
        self.sections.append(sec)
        self._current = sec
        if sec.name.startswith("list."):
            sec.list = CfgList()
        return sec

    def add_line(self, line: str, number: int) -> None:
        if self._current is None or self._current.list is None:
            return
        entry = CfgLine(line, number)
        lst = self._current.list
        if lst.tail:
            lst.tail.next = entry
            lst.tail = entry
        else:
            lst.head = lst.tail = entry

    def add_param(self, name: str, value: str) -> None:
        if self._current is None:
            return
        self._current.params[strlwr(trim(name))] = trim(value)

    def process_line(self, line: str, number: int) -> int:
        line = trim(line)
        if not line or line[0] in ("#", ";"):
            return 0
        if line[0] == "[":
            p = line.find("]")
            if p == -1:
                return 1
            self.add_section(trim(line[1:p]))
        elif self._current and self._current.list is not None:
            self.add_line(line, number)
        elif self._current and "=" in line:
            name, value = line.split("=", 1)
            self.add_param(name, value)
        else:
            return 1
        return 0

    def init(self, name: str) -> None:
        if self.sections:
            return
        try:
            with open(name, "r", encoding="utf-8", errors="replace") as fh:
                for number, raw in enumerate(fh, 1):
                    line = raw.rstrip("\n").rstrip("\r")
                    if self.process_line(line, number):
                        raise ValueError(f"Error in {name} at line {number}")
        except OSError as exc:
            raise OSError(f"fopen: {name}") from exc

    # -- lookup ------------------------------------------------------------
    @staticmethod
    def _match(name: str, section: str, subsection: Optional[str]) -> bool:
        p1 = 0
        p2 = 0
        # match the section prefix
        while p1 < len(name) and p2 < len(section) and name[p1] == section[p2].lower():
            p1 += 1
            p2 += 1
        if p2 < len(section):
            return False
        if subsection:
            q = 0
            while p1 < len(name) and q < len(subsection) and name[p1] == subsection[q].lower():
                p1 += 1
                q += 1
            if p1 < len(name) or q < len(subsection):
                return False
        return True

    def get_section(self, section: str, subsection: Optional[str]) -> Optional[CfgSection]:
        for sec in self.sections:
            if self._match(sec.name, section, subsection):
                return sec
        return None

    def get_list(self, section: str, subsection: Optional[str]) -> Optional[CfgList]:
        sec = self.get_section(section, subsection)
        return sec.list if sec else None

    def get_param(self, section: str, subsection: Optional[str], param: str) -> Optional[str]:
        sec = self.get_section(section, subsection)
        if not sec:
            return None
        return sec.params.get(param.lower())

    def get_int(self, section: str, subsection: Optional[str], param: str) -> int:
        val = self.get_param(section, subsection, param)
        return int(val) if val is not None else -1

    def get_bool(self, section: str, subsection: Optional[str], param: str) -> int:
        val = self.get_param(section, subsection, param)
        if val is None:
            return 0
        if val[0] in ("y", "Y", "t", "T"):
            return 1
        return int(val)


# Module-level database mirroring the C global ``cfg_database``.
_db = CfgDB()


def cfg_reset() -> None:
    """Clear the parsed configuration (used to reload a different file)."""
    global _db
    _db = CfgDB()


def cfg_init(name: str) -> None:
    _db.init(name)


def cfg_get_list(section: str, subsection: Optional[str]) -> Optional[CfgList]:
    return _db.get_list(section, subsection)


def cfg_get_param(section: str, subsection: Optional[str], param: str) -> Optional[str]:
    return _db.get_param(section, subsection, param)


def cfg_get_section(section: str, subsection: Optional[str]) -> Optional[CfgSection]:
    return _db.get_section(section, subsection)


def cfg_get_int(section: str, subsection: Optional[str], param: str) -> int:
    return _db.get_int(section, subsection, param)


def cfg_get_bool(section: str, subsection: Optional[str], param: str) -> int:
    return _db.get_bool(section, subsection, param)
