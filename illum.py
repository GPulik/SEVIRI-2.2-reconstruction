"""Illumination conventions of the solar reflectances.

satpy gives the 0.6 and 1.6 um SEVIRI reflectances and the FCI 2.25 um reflectance as 100 pi L d^2 / E0, with no
division by cos(theta); the pyspectral 3.9 um reflectance is divided by cos(theta). The published models use
these values as they are ("published"). Two alternatives, both needing the solar zenith angle in the cache:

  norm_all    0.6 and 1.6 um inputs and the target divided by cos(theta), so that all four inputs and the target are
              reflectance factors; the prediction is multiplied by cos(theta) again before scoring, so scores stay
              on the published scale
  unnorm_39   the 3.9 um input multiplied by cos(theta), so that all inputs and the target carry the same
              cos(theta) factor
"""
import numpy as np

MODES = ("published", "norm_all", "unnorm_39")


def convert(z, mode):
    """(cache dict with converted predictors/target, factor to multiply predictions with or None).
    Pixel-wise, so it works on a whole cache or on a subset of its pixels."""
    if mode not in MODES:
        raise ValueError(f"illumination mode {mode!r}, expected one of {MODES}")
    if mode == "published":
        return z, None
    if "sza" not in z:
        raise KeyError("the cache has no solar zenith angle: run scripts/add_sza.py (or rebuild it with prepare.py)")
    c = np.cos(np.deg2rad(np.asarray(z["sza"], np.float64))).astype(np.float32)
    if c.size and c.min() < 0.1:
        raise ValueError(f"cos(SZA) down to {c.min():.3f}: the caches should hold SZA below 82 deg only")
    out = dict(z)
    if mode == "norm_all":
        out["vis06"] = z["vis06"] / c
        out["vis16"] = z["vis16"] / c
        if "tgt" in z:
            out["tgt"] = z["tgt"] / c
        return out, c
    out["refl39"] = z["refl39"] * c
    return out, None
