"""SEVIRI point-spread function from the pre-launch MTF, and the NaN-aware forward operator.

The line-spread function of each scan direction is the inverse cosine transform of the detector-averaged
MTF (Equation (1)); the two form a separable kernel (Equation (2)) sampled on the 1 km FCI grid. The FCI
field is convolved with it, with a validity mask so that space and missing data do not bias the mean
(Equation (3)), and then sampled at the SEVIRI pixel centres (Equation (4))."""
import numpy as np
from scipy.ndimage import convolve1d


def read_mtf(xls, sheet, cols):
    """Frequencies and detector-averaged MTF from one block of the EUMETSAT MTF workbook."""
    import pandas as pd
    d = pd.read_excel(xls, sheet_name=sheet, header=None)
    fr = pd.to_numeric(d.iloc[4:, cols[0]], errors="coerce").values
    dts = [pd.to_numeric(d.iloc[4:, c], errors="coerce").values for c in cols[1:]]
    m = np.isfinite(fr)
    return fr[m], np.nanmean(np.vstack([x[m] for x in dts]), axis=0)


def lsf_from_mtf(freq, mtf, dx_km, half_km=12.0):
    """Zero-phase line-spread function on a grid of spacing dx_km (Equation (1))."""
    x = np.arange(-half_km, half_km + dx_km, dx_km)
    df = np.median(np.diff(freq))
    p = np.zeros_like(x)
    for fk, mk in zip(freq, mtf):
        p += mk * np.cos(2 * np.pi * fk * x) * df
    p = np.clip(p, 0, None)
    s = p.sum()
    return p / s if s > 0 else p


def trim(k, rel=1e-4):
    keep = k > k.max() * rel
    i0 = int(np.argmax(keep))
    i1 = len(keep) - int(np.argmax(keep[::-1]))
    kk = k[i0:i1]
    return kk / kk.sum()


def build_kernels(mtf_xls, sheet="NIR1.6", dx_km=1.0):
    """East-west and north-south kernels (columns 0-3 and 5-8 of the workbook sheet)."""
    fE, mE = read_mtf(mtf_xls, sheet, (0, 1, 2, 3))
    fN, mN = read_mtf(mtf_xls, sheet, (5, 6, 7, 8))
    return trim(lsf_from_mtf(fE, mE, dx_km)), trim(lsf_from_mtf(fN, mN, dx_km))


def scale_kernel(k, s):
    """Stretch (s > 1) or compress (s < 1) a kernel in width, keeping its shape; renormalised."""
    if s == 1.0:
        return k
    n = len(k)
    x = np.arange(n) - (n - 1) / 2.0
    m = int(np.ceil(n * max(s, 1.0)))
    m += (m + 1) % 2
    xn = np.arange(m) - (m - 1) / 2.0
    kn = np.interp(xn / s, x, k, left=0.0, right=0.0)
    t = kn.sum()
    return kn / t if t > 0 else kn


def fwhm(k, dx=1.0):
    n = len(k)
    x = (np.arange(n) - (n - 1) / 2.0) * dx
    half = k.max() / 2.0
    above = np.where(k >= half)[0]
    if above.size < 2:
        return float("nan")
    i, j = above[0], above[-1]
    lo = x[i] if i == 0 else np.interp(half, [k[i - 1], k[i]], [x[i - 1], x[i]])
    hi = x[j] if j == n - 1 else np.interp(half, [k[j + 1], k[j]], [x[j + 1], x[j]])
    return float(hi - lo)


def apply_psf_nan(img, k_ew, k_ns):
    """NaN-aware normalised convolution (Equation (3)): PSF-weighted mean over valid pixels only."""
    v = np.isfinite(img).astype(np.float64)
    d0 = np.where(v > 0, img, 0.0)
    num = convolve1d(convolve1d(d0, k_ew, axis=1, mode="constant"), k_ns, axis=0, mode="constant")
    del d0
    den = convolve1d(convolve1d(v, k_ew, axis=1, mode="constant"), k_ns, axis=0, mode="constant")
    del v
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den <= 1e-6] = np.nan
    return out
