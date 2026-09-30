"""Validation statistics (Equations (S1)-(S5)) with exact pooling through running sums, and the
evaluation classes of Section 2.3."""
import numpy as np
from scipy.ndimage import uniform_filter

from .config import CLASSES


def class_masks(cid, bt108_k):
    t = bt108_k - 273.15
    return {"All": np.ones(cid.size, bool), "All Cloud": cid == 2,
            "Warm Cloud": (cid == 2) & (t > 0.0),
            "Cold Cloud": (cid == 2) & (t <= 0.0) & (t >= -40.0),
            "Ice Cloud": (cid == 2) & (t < -40.0),
            "Land": cid == 1, "Ocean": cid == 0}


def sums(target, pred):
    """n, sum x, sum y, sum x2, sum y2, sum xy, SSE, SAE (x = target, y = prediction)."""
    x, y = target.astype(np.float64), pred.astype(np.float64)
    return np.array([x.size, x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum(),
                     ((y - x) ** 2).sum(), np.abs(y - x).sum()])


def stats(a):
    n, sx, sy, sxx, syy, sxy, sse, sae = a
    if n < 2:
        return None
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    sst = sxx - sx * sx / n
    return dict(n=int(n), r2=1.0 - sse / sst, rmse=float(np.sqrt(sse / n)), mae=sae / n,
                bias=(sy - sx) / n, slope=slope, intercept=(sy - slope * sx) / n)


def nan_uniform_filter(arr, size):
    m = ~np.isnan(arr)
    cnt = uniform_filter(m.astype(np.float32), size=size, mode="constant", cval=0.0)
    sm = uniform_filter(np.where(m, arr, 0.0), size=size, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, sm / cnt, np.nan)


def rel_sums(t_sm, p_sm, floor):
    """Sums for the relative RMSE of Equation (S5): smoothed fields, pixels with smoothed target > floor."""
    keep = np.isfinite(t_sm) & np.isfinite(p_sm) & (t_sm > floor)
    t, p = t_sm[keep].astype(np.float64), p_sm[keep].astype(np.float64)
    return np.array([((p - t) ** 2).sum(), t.sum(), t.size])


def rel_value(a):
    return np.nan if a[2] == 0 else np.sqrt(a[0] / a[2]) / (a[1] / a[2]) * 100.0


def scene_scores(z, pred, to2d, window=3, floor=0.02):
    """Per-class sums for one scene: {class: (sums, rel_sums)}."""
    t_sm = nan_uniform_filter(to2d(z, z["tgt"]), window).ravel()[z["idx"]]
    p_sm = nan_uniform_filter(to2d(z, pred), window).ravel()[z["idx"]]
    cm = class_masks(z["cid"], z["bt108"])
    out = {}
    for c in CLASSES:
        m = cm[c]
        if m.sum() >= 2:
            out[c] = (sums(z["tgt"][m], pred[m]), rel_sums(t_sm[m], p_sm[m], floor))
    return out
