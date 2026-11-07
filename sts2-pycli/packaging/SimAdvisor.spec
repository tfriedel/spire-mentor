# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the packaged SpireMentor advisor (GH #380).
#
# onedir, not onefile: onefile unpacks to temp on EVERY launch, and cold
# start is the whole point of dropping torch (45 s -> seconds). Console
# stays on: SimAdvisorProcess redirects stdout/stderr to its log, and a
# manual double-click shows the output instead of nothing.
#
# Build via scripts/build_advisor_exe.ps1 (dedicated .venv-advisor synced
# with `uv sync --only-group advisor-exe`, so torch cannot leak in even
# before the excludes below).

import glob
import os
import sys

SPEC_DIR = os.path.abspath(SPECPATH)
PYCLI = os.path.dirname(SPEC_DIR)

# The declared champions only -- retired nets stay out of the bundle -- read
# from data/ppo/champions.toml, the same manifest the advisor reads at runtime.
# Never a filename list here: that would be a separate champion resolver, and
# one a promotion could leave behind while every existence check still passed.
#
# The manifest ships INTO the bundle too, so the frozen exe resolves champions
# by name rather than by "newest file present in _internal/data/ppo" -- which is
# what let a retired net win in #339.
sys.path.insert(0, os.path.join(PYCLI, "python"))
import champions  # noqa: E402

# `onnx_path` honors this override for every character, which would bundle one
# net five times. Fine at runtime, never what a release build wants.
if os.environ.get("STS2_WB_PPO_ONNX"):
    raise SystemExit("unset STS2_WB_PPO_ONNX before building: it overrides every champion")

datas = [(str(champions.manifest_path()), "data/ppo")]
for character in champions.characters():
    # Raises FileNotFoundError naming the file if the manifest is ahead of the
    # tree -- do NOT let a release build silently fall back to an older net.
    src = champions.onnx_path(character)
    mismatch = champions.verify(character)
    if mismatch:
        # The gate swept a specific file; a re-export is a different artifact.
        raise SystemExit(f"champion hash mismatch for {character!r}: {mismatch}")
    print(f"bundling champion: {character} -> {os.path.basename(src)}")
    datas.append((src, "data/ppo"))
# Instinct, the e2e student (realgame_instinct): its weights, at the path the
# advisor resolves (`runtime_paths.data_root() / WEIGHTS_RELPATH`). Required: a
# release that silently lost it would answer every Instinct request with the
# search. The public export ships no weights (`data/e2e` is private), so a build
# from it opts out explicitly with SIMADVISOR_NO_INSTINCT=1; the mod then shows
# no Search / Instinct switch (the advisor advertises no Instinct).
import realgame_instinct  # noqa: E402

instinct = os.path.join(PYCLI, str(realgame_instinct.WEIGHTS_RELPATH))
if os.path.isfile(instinct):
    print(f"bundling Instinct: {os.path.basename(instinct)}")
    datas.append((instinct, str(realgame_instinct.WEIGHTS_RELPATH.parent).replace(os.sep, "/")))
elif os.environ.get("SIMADVISOR_NO_INSTINCT") == "1":
    print("no Instinct weights: building without Instinct (SIMADVISOR_NO_INSTINCT=1)")
else:
    raise SystemExit(
        f"no Instinct weights at {instinct} (set SIMADVISOR_NO_INSTINCT=1 to build without it)"
    )
loc = glob.glob(os.path.join(PYCLI, "localization_eng", "*.json"))
if not loc:
    raise SystemExit("no localization_eng/*.json found")
# Gzipped: the game reads every .json under a mod's folder as a mod manifest
# and logs a stack trace for each one that is not. The reader that runs in
# the bundle (sts2_hint/canonical_maps.py) falls back to <name>.json.gz.
import gzip  # noqa: E402
import shutil  # noqa: E402

loc_gz_dir = os.path.join(PYCLI, "build", "localization_eng_gz")
shutil.rmtree(loc_gz_dir, ignore_errors=True)
os.makedirs(loc_gz_dir)
for p in loc:
    gz = os.path.join(loc_gz_dir, os.path.basename(p) + ".gz")
    with open(p, "rb") as src, gzip.GzipFile(gz, "wb", mtime=0) as dst:
        shutil.copyfileobj(src, dst)
    datas.append((gz, "localization_eng"))

a = Analysis(
    [os.path.join(PYCLI, "python", "realgame_advisor.py")],
    pathex=[os.path.join(PYCLI, "python")],
    binaries=[],
    datas=datas,
    # PyO3 extension; modulegraph can miss it behind the lazy imports.
    hiddenimports=["sts2_rs"],
    hookspath=[],
    runtime_hooks=[],
    # Torch is the point (the .venv-advisor build env does not even have
    # it); the rest are belt-and-braces against a build from the dev venv.
    #
    # numpy + onnxruntime joined that list once the run policy moved to
    # `ppo.rust_policy` (tract, via the sts2_rs the advisor already loads
    # for combat MCTS). They are ~54 MB together and were the last reason
    # the bundle carried a numeric stack at all. Excluding rather than
    # merely omitting matters: `ppo.schema` / `ppo.encoder` still import
    # numpy, and a dev-venv build would otherwise pull them back in through
    # any lazy import the modulegraph happens to see.
    excludes=[
        "torch",
        "numpy",
        "onnxruntime",
        "torchvision",
        "torchaudio",
        "wandb",
        "scipy",
        "sklearn",
        "umap",
        "joblib",
        "pandas",
        "matplotlib",
        "fastapi",
        "uvicorn",
        "starlette",
        "pydantic",
        "mcp",
        "google",
        "IPython",
        "tkinter",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="SimAdvisor",
    debug=False,
    strip=False,
    upx=False,
    console=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="SimAdvisor",
)
