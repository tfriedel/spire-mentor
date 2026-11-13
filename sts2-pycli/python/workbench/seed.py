"""Seed canonicalization shared by both backends.

The C# engine's :class:`RunRngSet` hashes the seed string as-is, while the
Rust port's ``RunManager(string_seed=…)`` canonicalizes first (uppercase +
``O`` → ``0`` + ``I`` → ``1``) before hashing. For non-canonical user input
the two engines would otherwise compute different ``Seed`` ints and diverge
from step zero — same input, totally different runs.

The workbench applies :func:`canonicalize_seed` once at session start so
both backends see the same canonical string. Idempotent: a string that is
already canonical (e.g. a game-generated 10-char alphanumeric like
``"Y3X2YZSBQJ"``) is returned unchanged.
"""

from __future__ import annotations


def canonicalize_seed(seed: str) -> str:
    """Normalize a user-entered seed to the form the game would generate.

    Matches ``sts2_core::canonicalize_seed`` in the Rust port: trim,
    uppercase, and fold the visually ambiguous letters ``O`` → ``0`` and
    ``I`` → ``1``.
    """
    return seed.strip().upper().replace("O", "0").replace("I", "1")


# ──────────────────────────────────────────────────────────────────────────────
# Seed hashing (GH #792). A pure-Python reference of the game's seed hash per
# build, independent of the Rust engine, so the parity tests can pin
# Python == Rust == C# oracle for one seed string. The engine's own
# `sts2_rs.seed_hash` / `run_seed_hash` stay the implementation every harness
# path uses; these exist to cross-check it.
# ──────────────────────────────────────────────────────────────────────────────

_M32 = 0xFFFFFFFF
_M64 = 0xFFFFFFFFFFFFFFFF


def _i32(x: int) -> int:
    x &= _M32
    return x - (1 << 32) if x & 0x80000000 else x


def deterministic_hash_code_old(text: str) -> int:
    """v0.107.1 ``StringHelper.GetDeterministicHashCode`` (v0.111.0's
    ``GetDeterministicHashCodeOld``, ``StringHelper.cs:158``) — a signed int.

    Iterates UTF-16 code units, like the C# ``str[i]`` indexer.
    """
    units = text.encode("utf-16-le")
    chars = [units[i] | (units[i + 1] << 8) for i in range(0, len(units), 2)]
    num = 352654597
    num2 = num
    i = 0
    while i < len(chars):
        num = _i32(((num << 5) + num) ^ chars[i])
        if i == len(chars) - 1:
            break
        num2 = _i32(((num2 << 5) + num2) ^ chars[i + 1])
        i += 2
    return _i32(num + num2 * 1566083941)


_P1 = 0x9E3779B185EBCA87
_P2 = 0xC2B2AE3D27D4EB4F
_P3 = 0x165667B19E3779F9
_P4 = 0x85EBCA77C2B2AE63
_P5 = 0x27D4EB2F165667C5


def _rotl(x: int, r: int) -> int:
    return ((x << r) | (x >> (64 - r))) & _M64


def _round(acc: int, lane: int) -> int:
    acc = (acc + lane * _P2) & _M64
    return (_rotl(acc, 31) * _P1) & _M64


def _merge(acc: int, val: int) -> int:
    acc ^= _round(0, val)
    return (acc * _P1 + _P4) & _M64


def xxh64(data: bytes, seed: int = 0) -> int:
    """XxHash64, the reference algorithm (``System.IO.Hashing.XxHash64``)."""
    n = len(data)
    p = 0
    if n >= 32:
        v1 = (seed + _P1 + _P2) & _M64
        v2 = (seed + _P2) & _M64
        v3 = seed & _M64
        v4 = (seed - _P1) & _M64
        while p + 32 <= n:
            v1 = _round(v1, int.from_bytes(data[p : p + 8], "little"))
            v2 = _round(v2, int.from_bytes(data[p + 8 : p + 16], "little"))
            v3 = _round(v3, int.from_bytes(data[p + 16 : p + 24], "little"))
            v4 = _round(v4, int.from_bytes(data[p + 24 : p + 32], "little"))
            p += 32
        h = (_rotl(v1, 1) + _rotl(v2, 7) + _rotl(v3, 12) + _rotl(v4, 18)) & _M64
        for v in (v1, v2, v3, v4):
            h = _merge(h, v)
    else:
        h = (seed + _P5) & _M64
    h = (h + n) & _M64
    while p + 8 <= n:
        h ^= _round(0, int.from_bytes(data[p : p + 8], "little"))
        h = (_rotl(h, 27) * _P1 + _P4) & _M64
        p += 8
    if p + 4 <= n:
        h ^= (int.from_bytes(data[p : p + 4], "little") * _P1) & _M64
        h = (_rotl(h, 23) * _P2 + _P3) & _M64
        p += 4
    while p < n:
        h ^= (data[p] * _P5) & _M64
        h = (_rotl(h, 11) * _P1) & _M64
        p += 1
    h ^= h >> 33
    h = (h * _P2) & _M64
    h ^= h >> 29
    h = (h * _P3) & _M64
    h ^= h >> 32
    return h


#: Builds with ``ulong`` seeds and the XxHash64 string hash (v0.111.0 on;
#: v0.113.0 left the seeding unchanged).
_U64_SEED_BUILDS = frozenset({"v0.111.0", "v0.113.0"})


def seed_hash(text: str, game_version: str = "v0.107.1") -> int:
    """``StringHelper.GetDeterministicHashCode(text)`` as the unsigned value
    every seed site mixes in: ``(uint)`` of the 32-bit hash in v0.107.1,
    ``XxHash64(utf8, 0)`` in v0.111.0 (``StringHelper.cs:140``).

    Mirrors ``sts2_core::GameVersion::seed_hash``.
    """
    if game_version in _U64_SEED_BUILDS:
        return xxh64(text.encode("utf-8"), 0)
    return deterministic_hash_code_old(text) & _M32


def run_seed_hash(seed: str, game_version: str = "v0.107.1") -> int:
    """``RunRngSet.Seed`` for an (already canonical) seed string.

    v0.111.0 keeps the old 32-bit hash for a save-migrated ``"old"``-prefixed
    seed (``RunRngSet.cs:107``). Mirrors ``GameVersion::run_seed_hash``.
    """
    if game_version in _U64_SEED_BUILDS and seed.startswith("old"):
        return deterministic_hash_code_old(seed[3:]) & _M32
    return seed_hash(seed, game_version)


# The largest integer a browser's ``JSON.parse`` holds exactly (2^53 - 1).
_JSON_SAFE_MAX = (1 << 53) - 1


def json_exact_seed(seed: int) -> int | str:
    """A run seed as the browser can read it back exactly (GH #878).

    v0.111.0 run seeds are ``ulong``s, and ``JSON.parse`` rounds any number
    past 2^53. Those go out as a decimal string; every smaller seed (all of
    v0.107.1's ``uint``s) stays a number, so its bytes don't change. Matches
    ``seed_json`` in ``sts2-workbench/src/game.rs``.
    """
    return seed if 0 <= seed <= _JSON_SAFE_MAX else str(seed)
