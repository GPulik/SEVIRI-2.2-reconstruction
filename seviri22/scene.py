"""Reading one scene: SEVIRI predictors, the FCI 2.25 um target on the SEVIRI grid, and the cloud mask.

Array convention: satpy returns the SEVIRI full disc south-up / east-left; every array here is turned
north-up / west-left with [::-1, ::-1]. The pixel cache stores the flattened indices (`idx`) of the valid
pixels on that grid, so any field can be put back with `to2d`."""
import glob
import os
import shutil
import tempfile

import numpy as np

from .config import FEATURES
from .psf import apply_psf_nan

FLIP = (slice(None, None, -1), slice(None, None, -1))
SOLAR = ["VIS006", "IR_016"]
THERMAL = ["IR_039", "IR_108", "IR_134"]


def _files(scene_dir, pattern):
    return sorted(glob.glob(os.path.join(scene_dir, pattern)))


def seviri_files(scene_dir):
    f = _files(scene_dir, "*MSG*.nat")
    if len(f) != 1:
        raise FileNotFoundError(f"{scene_dir}: expected one SEVIRI .nat file, found {len(f)}")
    return f


def fci_files(scene_dir):
    f = _files(scene_dir, "*MTI1*BODY*.nc") or _files(scene_dir, "*MTI1*.nc")
    if not f:
        raise FileNotFoundError(f"{scene_dir}: no FCI files")
    return f


def read_seviri(nat_files, cfg):
    """SEVIRI predictors on the north-up grid.

    Solar channels use cfg['seviri']['calib_mode'] (MEIRINK-2023 by default); satpy accepts that mode only
    for the solar channels, so the thermal channels are read in a second scene with the nominal calibration.
    The 3.9 um solar reflectance is computed with pyspectral for the spectral response of
    cfg['seviri']['rsr_platform']."""
    from satpy import Scene
    from pyorbital.astronomy import get_alt_az
    from pyspectral.near_infrared_reflectance import Calculator

    sv = cfg["seviri"]
    mode = sv.get("calib_mode") or "NOMINAL"
    sol = Scene(filenames=nat_files, reader="seviri_l1b_native", reader_kwargs={"calib_mode": mode})
    sol.load(SOLAR)
    th = Scene(filenames=nat_files, reader="seviri_l1b_native")
    th.load(THERMAL)
    area = th["IR_108"].attrs["area"]
    t0 = th.start_time
    lons, lats = area.get_lonlats()
    lons = np.where(np.abs(lons) > 360, np.nan, lons)
    lats = np.where(np.abs(lats) > 90, np.nan, lats)
    sza = 90.0 - np.degrees(get_alt_az(t0, np.nan_to_num(lons), np.nan_to_num(lats))[0])
    sza = np.where(np.isfinite(lons), sza, np.nan)
    del lons, lats
    calc = Calculator(sv["rsr_platform"], "seviri", "IR_039")
    refl39 = calc.reflectance_from_tbs(sza, th["IR_039"].values, th["IR_108"].values,
                                       tb_ir_co2=th["IR_134"].values)
    bt108 = np.asarray(th["IR_108"].values, np.float64)[FLIP]
    out = {
        "vis16": (np.asarray(sol["IR_016"].values)[FLIP] / 100.0).astype(np.float32),
        "vis06": (np.asarray(sol["VIS006"].values)[FLIP] / 100.0).astype(np.float32),
        "refl39": np.asarray(refl39)[FLIP].astype(np.float32),
        "temp": ((bt108 - sv["temp_min_k"]) / (sv["temp_max_k"] - sv["temp_min_k"])).astype(np.float32),
        "bt108": bt108.astype(np.float32),
        "sza": np.asarray(sza)[FLIP].astype(np.float32),
    }
    return out, area, t0


def seviri_area(nat_files):
    from satpy import Scene
    s = Scene(filenames=nat_files, reader="seviri_l1b_native")
    s.load(["IR_108"])
    return s["IR_108"].attrs["area"]


def fci_on_seviri(fci_paths, area, k_ew, k_ns, channel="nir_22", operator="psf"):
    """An FCI channel mapped onto the SEVIRI grid (north-up), as reflectance fraction.

    operator: 'psf' (Equations (3)-(4): PSF convolution on the 1 km grid, then nearest sample),
              'grid_mean' (satpy bucket_avg: mean of the FCI pixels whose centres fall in each SEVIRI cell),
              'nearest' (one FCI pixel)."""
    from satpy import Scene
    mtg = Scene(filenames=fci_paths, reader="fci_l1c_nc")
    mtg.load([channel])
    if operator == "psf":
        native = np.asarray(mtg[channel].values, dtype=np.float64)
        mtg["mapped"] = mtg[channel].copy(data=apply_psf_nan(native, k_ew, k_ns))
        del native
        r = mtg.resample(area, resampler="nearest", datasets=["mapped"])["mapped"].values
    elif operator == "grid_mean":
        r = mtg.resample(area, resampler="bucket_avg", datasets=[channel])[channel].values
    elif operator == "nearest":
        r = mtg.resample(area, resampler="nearest", datasets=[channel])[channel].values
    else:
        raise ValueError(operator)
    return (np.asarray(r)[FLIP] / 100.0).astype(np.float32)


def read_mask(path):
    """Cloud mask written by cloudmask.build_mask, turned north-up; 0 ocean, 1 land, 2 cloud, 3 unprocessed."""
    import xarray as xr
    with xr.open_dataset(path) as ds:
        ds.load()
        a = np.asarray(ds[[v for v in ds.data_vars if "mask" in v][0]].values)
    return a[FLIP]


def extract(scene_dir, mask_path, cfg, k_ew, k_ns, with_target=True):
    """Valid pixels of one scene as flat arrays (the pixel cache). Valid = SZA below the limit and finite
    predictors (and finite target when with_target)."""
    with tempfile.TemporaryDirectory() as ld:        # local copies: faster than network drives
        nat = []
        for f in seviri_files(scene_dir):
            shutil.copy2(f, ld); nat.append(os.path.join(ld, os.path.basename(f)))
        feats, area, t0 = read_seviri(nat, cfg)
        tgt = None
        if with_target:
            fci = []
            for f in fci_files(scene_dir):
                shutil.copy2(f, ld); fci.append(os.path.join(ld, os.path.basename(f)))
            tgt = fci_on_seviri(fci, area, k_ew, k_ns)
    mask = feats["sza"] < cfg["seviri"]["sza_limit_deg"]
    for k in FEATURES:
        mask &= np.isfinite(feats[k])
    if tgt is not None:
        mask &= np.isfinite(tgt)
    idx = np.flatnonzero(mask.ravel()).astype(np.int64)
    out = {k: feats[k].ravel()[idx] for k in FEATURES + ["bt108"]}
    if tgt is not None:
        out["tgt"] = tgt.ravel()[idx]
    if mask_path and os.path.exists(mask_path):
        cid = read_mask(mask_path)
        out["cid"] = np.nan_to_num(np.asarray(cid, np.float64).ravel()[idx], nan=3).astype(np.int8)
    out.update(idx=idx, shape=np.array(mask.shape, np.int64),
               start_time=np.array(str(t0)), rsr_platform=np.array(cfg["seviri"]["rsr_platform"]),
               calib_mode=np.array(str(cfg["seviri"]["calib_mode"])))
    return out


def to2d(z, values):
    shape = tuple(int(s) for s in z["shape"])
    out = np.full(shape[0] * shape[1], np.nan, np.float32)
    out[z["idx"]] = values
    return out.reshape(shape)


def cache_path(cfg, scene):
    return os.path.join(cfg["paths"]["cache_dir"], f"{scene}.npz")


def load_cache(cfg, scene, keys=None):
    with np.load(cache_path(cfg, scene)) as z:
        return {k: z[k] for k in (keys or z.files)}
