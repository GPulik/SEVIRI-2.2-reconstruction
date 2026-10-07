#!/usr/bin/env python3
"""Build the cloud masks and the per-scene pixel caches (predictors, PSF-matched FCI target, classes).

  python scripts/prepare.py                         # all scenes of the scene list
  python scripts/prepare.py --use training --delete-fci

Each cache (<cache_dir>/<scene>.npz) holds the valid pixels (SZA below the limit, finite predictors and
target): vis16, refl39, temp, vis06, bt108, tgt, cid, and their flat indices on the north-up SEVIRI grid.
Existing caches are kept; delete one to rebuild it (for example after changing --rsr-platform)."""
import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seviri22.config import add_common_args, config_from_args, ensure_dirs, scene_list  # noqa: E402
from seviri22.psf import build_kernels, scale_kernel, fwhm  # noqa: E402
from seviri22 import scene as sc  # noqa: E402
from seviri22.cloudmask import build_mask  # noqa: E402


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--use", choices=["training", "validation"], default=None)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--delete-fci", action="store_true", help="delete a scene's FCI files once its cache is written")
    args = ap.parse_args()
    cfg = config_from_args(args)
    ensure_dirs(cfg)
    p = cfg["psf"]
    k_ew, k_ns = build_kernels(cfg["paths"]["mtf_xls"], p["mtf_sheet"], p["fci_km_per_px"])
    k_ew, k_ns = scale_kernel(k_ew, p["scale"]), scale_kernel(k_ns, p["scale"])
    print(f"PSF FWHM {fwhm(k_ew):.2f} km east-west, {fwhm(k_ns):.2f} km north-south; "
          f"calibration {cfg['seviri']['calib_mode']}; 3.9 um response {cfg['seviri']['rsr_platform']}")
    scenes = args.scenes or scene_list(cfg, args.use)
    for i, s in enumerate(scenes, 1):
        out = sc.cache_path(cfg, s)
        if os.path.exists(out):
            print(f"[{i}/{len(scenes)}] {s}: cache exists")
            continue
        sdir = os.path.join(cfg["paths"]["scenes_dir"], s)
        mpath = os.path.join(cfg["paths"]["masks_dir"], f"{s}_mask.nc")
        t0 = time.time()
        try:
            if not os.path.exists(mpath):
                clm = sorted(glob.glob(os.path.join(sdir, "clm_grib", "*")))
                if not clm:
                    raise FileNotFoundError("no CLM file (run download.py)")
                orient = build_mask(clm[0], sc.seviri_files(sdir), mpath)
                print(f"  mask oriented: {orient}")
            z = sc.extract(sdir, mpath, cfg, k_ew, k_ns)
            np.savez(out + ".part.npz", **z)
            os.replace(out + ".part.npz", out)
            print(f"[{i}/{len(scenes)}] {s}: {z['idx'].size:,} valid pixels ({time.time() - t0:.0f} s)")
            if args.delete_fci:
                for f in sc.fci_files(sdir):
                    os.remove(f)
        except Exception as exc:
            print(f"[{i}/{len(scenes)}] {s}: failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
