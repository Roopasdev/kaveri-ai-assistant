"""Guards for everything the model writes or reads (midstones 5 and 9).

SQL: the model may only ever run ONE read-only SELECT. Checked here before the
database sees it, and again by the database itself (`readonly_engine`).
"""
from __future__ import annotations

import re

BLOCKED = ("insert", "update", "delete", "drop", "alter", "create", "replace", "truncate",
           "attach", "detach", "pragma", "vacuum", "reindex", "grant", "revoke", "merge", "copy")
_BLOCKED_RE = re.compile(r"\b(" + "|".join(BLOCKED) + r")\b", re.I)
_STRING_RE = re.compile(r"'(?:[^']|'')*'")
_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.S)


class UnsafeSQL(ValueError):
    """The SQL is not one read-only SELECT. Never sent to the database."""


def check_sql(sql: str) -> str:
    if not sql or not sql.strip():
        raise UnsafeSQL("No SQL was written.")
    # Look at the statement without its string literals and comments, so a
    # keyword hidden in a comment is still caught and 'drop' inside a quoted
    # value does not block an honest query.
    bare = _COMMENT_RE.sub(" ", _STRING_RE.sub("''", sql)).strip()
    if ";" in bare.rstrip(";"):
        raise UnsafeSQL("Only one statement is allowed.")
    if not re.match(r"^(select|with)\b", bare, re.I):
        raise UnsafeSQL("Only SELECT queries are allowed.")
    bad = _BLOCKED_RE.search(bare)
    if bad:
        raise UnsafeSQL(f"'{bad.group(1).upper()}' is not allowed.")
    return sql
