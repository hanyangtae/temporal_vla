#!/usr/bin/env python3
"""Single CLI for seen18 GR00T N1.6 latent-steering analyses.

Subcommands live in analyses/*.py (each exposes NAME/HELP/add_args/run) and share
helpers from core/ (io, schemes, metrics, geometry, plot3d). List commands:

  python seen18.py --help
  python seen18.py <cmd> --help
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyses  # noqa: E402


def _load_modules():
    mods = {}
    for m in pkgutil.iter_modules(analyses.__path__):
        if m.name.startswith("_"):
            continue
        mod = importlib.import_module(f"analyses.{m.name}")
        if hasattr(mod, "NAME") and hasattr(mod, "run") and hasattr(mod, "add_args"):
            mods[mod.NAME] = mod
    return dict(sorted(mods.items()))


def main(argv=None) -> None:
    mods = _load_modules()
    p = argparse.ArgumentParser(prog="seen18", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, mod in mods.items():
        sp = sub.add_parser(name, help=getattr(mod, "HELP", ""), description=mod.__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
        mod.add_args(sp)
    args = p.parse_args(argv)
    mods[args.cmd].run(args)


if __name__ == "__main__":
    main()
