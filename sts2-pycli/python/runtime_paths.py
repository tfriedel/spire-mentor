"""Frozen-aware path resolution for the packaged advisor exe (GH #380).

The PyInstaller onedir build of ``realgame_advisor`` relocates two kinds of
path that are ``__file__``-relative in a dev checkout:

- read-only data (``localization_eng/``, ``data/ppo/``) lands under the
  bundle's ``_internal`` dir (``sys._MEIPASS``);
- writes (logs, config) must leave the install dir entirely -- the mod
  ships under ``<game>/mods/SpireMentor/``, which is ``Program Files`` on a
  default Steam install and not user-writable.

Every function returns the historical dev path unless ``sys.frozen`` is
set, so behavior on a source checkout is byte-identical.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    """True when running from a PyInstaller bundle."""
    return bool(getattr(sys, "frozen", False))


def bundle_root() -> Path:
    """Where PyInstaller datas land (onedir: ``<dist>/SimAdvisor/_internal``)."""
    return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))


def data_root() -> Path:
    """Root containing ``localization_eng/`` and ``data/ppo/``.

    Dev: the ``sts2-pycli`` checkout dir (this file lives in
    ``sts2-pycli/python/``). Frozen: the bundle root, where the spec's
    ``datas`` entries replicate the same relative layout.
    """
    if is_frozen():
        return bundle_root()
    return Path(__file__).resolve().parent.parent


def workbench_dir() -> Path:
    """The browser-only workbench build the control plane hosts.

    Dev: ``web/dist`` (whatever was built last; hosted only when it is the
    static build). Frozen: the mod zip's ``mods/SpireMentor/workbench/``, the
    sibling of the ``SimAdvisor/`` dir this exe lives in.
    """
    if is_frozen():
        return Path(sys.executable).resolve().parent.parent / "workbench"
    return Path(__file__).resolve().parent.parent / "web" / "dist"


def app_data_dir() -> Path:
    """Writable per-user dir for the packaged advisor's logs/config.

    Shared with the C# side, which writes ``advisor_stdout.log`` and
    ``SpireMentor.conf`` under the same ``%LOCALAPPDATA%\\SpireMentor``.
    """
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "SpireMentor"
