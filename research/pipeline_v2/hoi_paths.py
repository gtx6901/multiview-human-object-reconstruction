"""Resolve asset paths from configs/paths.json.

Scripts stay flat in scripts/; assets live in data/, assets/{human,helmet,backpack}/,
runtime/, renders/ and eval/. A bare name such as ``ROOT/'underarm_weight_amount_fit_v3'``
is looked up in those groups in order; anything not found is treated as a new output
and created under runs/.
"""
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
_CFG = json.loads((_PROJECT / "configs" / "paths.json").read_text(encoding="utf-8"))
_BASE = (_PROJECT / _CFG.get("root", ".")).resolve()
_SEARCH = [_BASE] + [(_BASE / s).resolve() for s in _CFG.get("search", [])]
_ALIAS = _CFG.get("alias", {})
_OUT = (_BASE / _CFG.get("output", "runs")).resolve()


class _Resolver:
    """Path-like object that searches the configured asset groups."""

    __slots__ = ()

    def __truediv__(self, key):
        name = str(key).replace("\\", "/")
        head, sep, tail = name.partition("/")
        head = _ALIAS.get(head, head)
        for base in _SEARCH:
            candidate = base / head
            if candidate.exists():
                return candidate / tail if sep else candidate
        out = _OUT / head
        return out / tail if sep else out

    def __rtruediv__(self, other):
        return Path(other) / str(_BASE)

    def __fspath__(self):
        return str(_BASE)

    def __str__(self):
        return str(_BASE)

    def __repr__(self):
        return f"<project {_BASE}>"

    @property
    def parent(self):
        return self

    @property
    def name(self):
        return _BASE.name

    def exists(self):
        return True

    def is_dir(self):
        return True

    def resolve(self):
        return _BASE

    def absolute(self):
        return _BASE

    def joinpath(self, *parts):
        out = self
        for part in parts:
            out = out / part
        return out

    def glob(self, pattern):
        return _BASE.glob(pattern)

    def rglob(self, pattern):
        return _BASE.rglob(pattern)

    def iterdir(self):
        return _BASE.iterdir()


ROOT = _Resolver()


def P(*parts):
    """Resolve several path components with the same group search."""
    out = ROOT
    for part in parts:
        out = out / part
    return out


def project_dir():
    return _BASE


def runs_dir():
    return _OUT
