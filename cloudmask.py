"""EUMETSAT MSG cloud mask (CLM, GRIB) on the SEVIRI grid.

The GRIB array orientation is not fixed across readers, so the mask is oriented against the scene's own
10.8 um brightness temperature: the right orientation puts the cold pixels under the cloud class."""
import os

import numpy as np

FLIPS = {"as stored": lambda a: a, "flip rows": lambda a: a[::-1, :],
         "flip columns": lambda a: a[:, ::-1], "flip both": lambda a: a[::-1, ::-1]}


def read_clm_grib(path):
    """CLM GRIB -> float32 classes 0-3 (0 clear water, 1 clear land, 2 cloud, 3 not processed), NaN elsewhere."""
    import eccodes
    with open(path, "rb") as fh:
        gid = eccodes.codes_grib_new_from_file(fh)
        try:
            nx, ny = eccodes.codes_get(gid, "Nx"), eccodes.codes_get(gid, "Ny")
            vals = np.asarray(eccodes.codes_get_values(gid), np.float64)
            miss = eccodes.codes_get(gid, "missingValue")
        finally:
            eccodes.codes_release(gid)
    a = vals.reshape(ny, nx)
    return np.where((a == miss) | ~np.isin(a, (0, 1, 2, 3)), np.nan, a).astype(np.float32)


def orientation_scores(mask, bt):
    """Median 10.8 um BT of clear pixels minus that of cloudy pixels (K), for each orientation."""
    out = {}
    for k, f in FLIPS.items():
        m = f(mask)
        ok = np.isfinite(m) & np.isfinite(bt)
        clr, cld = ok & ((m == 0) | (m == 1)), ok & (m == 2)
        out[k] = np.nan if clr.sum() < 1000 or cld.sum() < 1000 else float(np.median(bt[clr]) - np.median(bt[cld]))
    return out


def choose_orientation(scores, min_contrast=8.0, min_margin=4.0):
    v = sorted(((s, k) for k, s in scores.items() if np.isfinite(s)), reverse=True)
    if not v:
        return None, "no scorable orientation"
    if v[0][0] < min_contrast:
        return None, f"best contrast {v[0][0]:.1f} K < {min_contrast} K"
    if len(v) > 1 and v[0][0] - v[1][0] < min_margin:
        return None, f"ambiguous: {v[0][1]} {v[0][0]:.1f} K vs {v[1][1]} {v[1][0]:.1f} K"
    return v[0][1], "ok"


def build_mask(clm_path, nat_files, out_path):
    """Orient the CLM against the scene's 10.8 um BT (satpy orientation) and write it as NetCDF."""
    import xarray as xr
    from satpy import Scene
    raw = read_clm_grib(clm_path)
    s = Scene(filenames=nat_files, reader="seviri_l1b_native")
    s.load(["IR_108"])
    bt = np.asarray(s["IR_108"].values, np.float32)
    if raw.shape != bt.shape:
        raise ValueError(f"CLM {raw.shape} and SEVIRI {bt.shape} grids differ")
    k, why = choose_orientation(orientation_scores(raw, bt))
    if k is None:
        raise ValueError(f"{clm_path}: mask orientation undecided ({why})")
    m = FLIPS[k](raw).copy()
    tmp = out_path + ".part.nc"
    xr.Dataset({"cloud_mask": (("y", "x"), m)},
               attrs={"source": os.path.basename(clm_path), "orientation_applied": k,
                      "classes": "0 clear water, 1 clear land, 2 cloud, 3 not processed"}).to_netcdf(tmp)
    os.replace(tmp, out_path)
    return k
