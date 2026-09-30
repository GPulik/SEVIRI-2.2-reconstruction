#!/usr/bin/env python3
"""Test of the mapping operator on the 1.6 um band, which both instruments carry (Table S4, Figure S4).

For each validation scene the FCI 1.6 um field is mapped onto the SEVIRI grid with the PSF (at several
widths, shape kept), the grid-cell mean and the nearest FCI pixel, and compared with the SEVIRI 1.6 um
reflectance of the pixel cache. Reported per class: squared correlation and RMSE about a linear fit, so the
calibration difference between the two instruments does not enter.

  python scripts/operator_test.py --scales 0.6 0.8 0.9 1.0 1.1 1.2 1.4 1.6
Needs the FCI files of the validation scenes (keep them, i.e. prepare.py without --delete-fci)."""
import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seviri22.config import add_common_args, config_from_args, ensure_dirs, scene_list  # noqa: E402
from seviri22.psf import build_kernels, scale_kernel, fwhm  # noqa: E402
from seviri22 import scene as sc  # noqa: E402

CLS = {"All": lambda z: np.ones(z["cid"].size, bool), "All Cloud": lambda z: z["cid"] == 2,
       "Ice Cloud": lambda z: (z["cid"] == 2) & (z["bt108"] < 233.15),
       "Land": lambda z: z["cid"] == 1, "Ocean": lambda z: z["cid"] == 0}


def sums(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m].astype(np.float64), y[m].astype(np.float64)
    return np.array([x.size, x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum()])


def fit_stats(a):
    n, sx, sy, sxx, syy, sxy = a
    vx, vy, cxy = sxx / n - (sx / n) ** 2, syy / n - (sy / n) ** 2, sxy / n - sx * sy / n ** 2
    r2 = cxy ** 2 / (vx * vy)
    return dict(n=int(n), r2=r2, rmse_fit=float(np.sqrt(max(vy * (1 - r2), 0.0))))


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--scales", nargs="*", type=float, default=[0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6])
    ap.add_argument("--scenes", nargs="*", default=None)
    args = ap.parse_args()
    cfg = config_from_args(args)
    ensure_dirs(cfg)
    p = cfg["psf"]
    k_ew, k_ns = build_kernels(cfg["paths"]["mtf_xls"], p["mtf_sheet"], p["fci_km_per_px"])
    ops = ["nearest", "grid_mean"] + [f"psf_{s:.2f}" for s in args.scales]
    pooled = {}
    rows = []
    for i, s in enumerate(args.scenes or scene_list(cfg, "validation"), 1):
        z = sc.load_cache(cfg, s, ("idx", "shape", "cid", "bt108", "vis16"))
        sdir = os.path.join(cfg["paths"]["scenes_dir"], s)
        try:
            fci_src = sc.fci_files(sdir)
        except FileNotFoundError:
            print(f"[{i}] {s}: no FCI files, skipped")
            continue
        with tempfile.TemporaryDirectory() as ld:
            fci = []
            for f in fci_src:
                shutil.copy2(f, ld); fci.append(os.path.join(ld, os.path.basename(f)))
            area = sc.seviri_area(sc.seviri_files(sdir))
            for op in ops:
                if op.startswith("psf_"):
                    s_ = float(op[4:])
                    field = sc.fci_on_seviri(fci, area, scale_kernel(k_ew, s_), scale_kernel(k_ns, s_), "nir_16", "psf")
                else:
                    field = sc.fci_on_seviri(fci, area, None, None, "nir_16", op)
                x = field.ravel()[z["idx"]]
                for c, f in CLS.items():
                    m = f(z)
                    a = sums(x[m], z["vis16"][m])
                    pooled[(op, c)] = pooled.get((op, c), 0) + a
                    rows.append(dict(scene=s[6:], operator=op, cls=c, **fit_stats(a)))
        print(f"[{i}] {s}")
    od = cfg["paths"]["out_dir"]
    pd.DataFrame(rows).to_csv(os.path.join(od, "operator_test_per_scene.csv"), index=False)
    out = []
    for (op, c), a in pooled.items():
        s_ = float(op[4:]) if op.startswith("psf_") else np.nan
        out.append(dict(operator=op, scale=s_, fwhm_ew_km=fwhm(scale_kernel(k_ew, s_)) if op.startswith("psf_") else np.nan,
                        cls=c, **fit_stats(a)))
    T = pd.DataFrame(out)
    T.to_csv(os.path.join(od, "operator_test.csv"), index=False)
    print(T[T.cls == "All Cloud"].to_string(index=False))


if __name__ == "__main__":
    main()
