#!/usr/bin/env python3
"""Download SEVIRI Level 1.5, FCI Level 1c and the MSG cloud mask for the scenes of the scene list.

  python scripts/download.py                       # all 76 scenes of data/scenes.csv
  python scripts/download.py --use validation      # only the validation scenes
  python scripts/download.py --scenes Scene_2025021912 Scene_2026050508

Needs EUMETSAT_KEY and EUMETSAT_SECRET in the environment. About 1.2 GB per scene (mostly FCI)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seviri22.config import add_common_args, config_from_args, ensure_dirs, scene_list  # noqa: E402
from seviri22 import datastore  # noqa: E402


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--use", choices=["training", "validation"], default=None)
    ap.add_argument("--scenes", nargs="*", default=None, help="scene names (Scene_YYYYMMDDHH)")
    args = ap.parse_args()
    cfg = config_from_args(args)
    ensure_dirs(cfg)
    scenes = args.scenes or scene_list(cfg, args.use)
    failed = []
    for i, s in enumerate(scenes, 1):
        sdir = os.path.join(cfg["paths"]["scenes_dir"], s)
        print(f"[{i}/{len(scenes)}] {s}")
        try:
            got = datastore.resolve(s[6:], cfg)
            datastore.fetch(*got["seviri"], sdir)
            datastore.fetch(*got["fci"], sdir)
            datastore.fetch(*got["clm"], os.path.join(sdir, "clm_grib"))
        except Exception as exc:
            print(f"  failed: {type(exc).__name__}: {exc}")
            failed.append(s)
    print(f"done; {len(scenes) - len(failed)} of {len(scenes)} scenes complete" +
          (f"; failed: {', '.join(failed)}" if failed else ""))


if __name__ == "__main__":
    main()
