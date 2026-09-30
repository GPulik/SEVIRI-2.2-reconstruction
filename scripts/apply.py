#!/usr/bin/env python3
"""Synthetic 2.25 um reflectance and Cloud Phase RGB for any SEVIRI Level 1.5 native file.

  python scripts/apply.py MSG3-SEVI-MSG15-0100-NA-20250219121243.695000000Z-NA.nat
  python scripts/apply.py MSG2-SEVI-...nat --rsr-platform Meteosat-9      # e.g. IODC, Meteosat-9

Options: --model (default <models_dir>/rf.joblib), --rsr-platform (spectral response for the 3.9 um
reflectance; use the platform that acquired the file), --calib-mode (solar-channel calibration; default
MEIRINK-2023, as for training), --out (directory).
Writes <name>_nir22_synthetic.nc (reflectance fraction, north-up SEVIRI grid, NaN where SZA exceeds the
limit or a predictor is missing) and <name>_cloud_phase_rgb.png.
Outside the Meteosat-10 0 deg configuration used for training the result is not validated."""
import argparse
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seviri22.config import add_common_args, config_from_args, FEATURES  # noqa: E402
from seviri22.scene import read_seviri  # noqa: E402
from seviri22.models import predict_chunked  # noqa: E402
from seviri22.rgb import cloud_phase_rgb, save_png  # noqa: E402


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("nat", nargs="+", help="SEVIRI Level 1.5 native file(s) of one scene")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    cfg = config_from_args(args)
    model = joblib.load(args.model or os.path.join(cfg["paths"]["models_dir"], "rf.joblib"))
    f, area, t0 = read_seviri(args.nat, cfg)
    ok = f["sza"] < cfg["seviri"]["sza_limit_deg"]
    for k in FEATURES:
        ok &= np.isfinite(f[k])
    g = np.full(ok.shape, np.nan, np.float32)
    g[ok] = predict_chunked(model, np.column_stack([f[k][ok] for k in FEATURES]).astype(np.float32))
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, os.path.splitext(os.path.basename(args.nat[0]))[0])
    import xarray as xr
    xr.Dataset({"nir22_synthetic": (("y", "x"), g)},
               attrs={"description": "synthetic 2.25 um reflectance (fraction) reconstructed from SEVIRI",
                      "start_time": str(t0), "orientation": "north-up, west-left (satpy array reversed in both axes)",
                      "area_proj4": getattr(area, "proj_str", str(area)),
                      "area_extent": list(getattr(area, "area_extent", [])),
                      "calib_mode": str(cfg["seviri"]["calib_mode"]), "rsr_platform": cfg["seviri"]["rsr_platform"]}
               ).to_netcdf(stem + "_nir22_synthetic.nc")
    save_png(cloud_phase_rgb(f["vis16"], g, f["vis06"]) * ok[..., None], stem + "_cloud_phase_rgb.png")
    print(f"{int(ok.sum()):,} pixels; median synthetic 2.25 um {np.nanmedian(g):.3f}; written {stem}_*")


if __name__ == "__main__":
    main()
