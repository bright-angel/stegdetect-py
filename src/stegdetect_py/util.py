"""Faithful Python port of stegdetect's ``util.c``.

Only :func:`fgetl` (read one line, trimming ``\\n`` and a trailing ``\\r``) is
needed by the ported code.  It mirrors the original ``fgetl(char *s, int size,
FILE *stream)`` but operates on a Python text ``stream`` instead of a C
``FILE *``.
"""


def fgetl(stream):
    """Read a single line from ``stream`` (a text-mode file object).

    Returns the line *without* the trailing newline and without a trailing
    carriage return, or ``None`` at end-of-file -- matching the contract of
    the C ``fgetl`` (which returns ``NULL`` / the buffer).
    """
    line = stream.readline()
    if line == "":           # EOF
        return None
    if line.endswith("\n"):
        line = line[:-1]
    if line.endswith("\r"):
        line = line[:-1]
    return line
