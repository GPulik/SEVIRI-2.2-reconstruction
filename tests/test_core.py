"""Tests of the parts that need no satellite data. Run: python -m pytest tests  (or python tests/test_core.py)."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seviri22 import psf, models, metrics, cloudmask, rgb  # noqa: E402
from seviri22.config import load_config, FEATURES  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_config_and_override():
    cfg = load_config(os.path.join(ROOT, "config.yaml"), {"seviri.rsr_platform": "Meteosat-9"})
    assert cfg["seviri"]["rsr_platform"] == "Meteosat-9"
    assert cfg["seviri"]["calib_mode"] == "MEIRINK-2023"
    assert os.path.isabs(cfg["paths"]["cache_dir"])
    assert cfg["training"]["per_scene"] * 37 == 66_277_064


def test_kernel_from_gaussian_mtf():
    # the MTF of a Gaussian of standard deviation s is exp(-2 pi^2 s^2 f^2); the recovered LSF must match
    s = 1.8
    f = np.linspace(0, 1.0, 400)
    k = psf.trim(psf.lsf_from_mtf(f, np.exp(-2 * np.pi ** 2 * s ** 2 * f ** 2), 1.0))
    assert abs(psf.fwhm(k) - 2.3548 * s) < 0.1
    assert abs(k.sum() - 1) < 1e-12


def test_scale_kernel():
    x = np.arange(-12, 13)
    k = np.exp(-0.5 * (x / 2.0) ** 2); k /= k.sum()
    assert np.allclose(psf.scale_kernel(k, 1.0), k)
    for s in (0.8, 1.2, 1.4):                            # width scales with s, shape kept
        assert abs(psf.fwhm(psf.scale_kernel(k, s)) / psf.fwhm(k) - s) < 0.03, s
    assert abs(psf.scale_kernel(k, 1.3).sum() - 1) < 1e-12


def test_psf_nan_aware():
    rng = np.random.default_rng(0)
    img = rng.random((40, 50))
    img[:5] = np.nan
    k = np.array([0.25, 0.5, 0.25])
    out = psf.apply_psf_nan(img, k, k)
    i, j = 20, 20                                         # interior: plain weighted mean
    assert abs(out[i, j] - (np.outer(k, k) * img[i - 1:i + 2, j - 1:j + 2]).sum()) < 1e-12
    i = 5                                                 # next to NaN rows: mean over valid pixels only
    w = np.outer(k, k); blk = img[i - 1:i + 2, j - 1:j + 2]; v = np.isfinite(blk)
    assert abs(out[i, j] - (w * np.where(v, blk, 0)).sum() / (w * v).sum()) < 1e-12
    assert np.isnan(out[:4]).all()
    const = np.full((30, 30), 0.3)
    assert np.allclose(psf.apply_psf_nan(const, k, k)[2:-2, 2:-2], 0.3)


def test_select_pool():
    cid = np.array([0, 0, 1, 2, 3, 0])
    u = np.array([0.01, 0.9, 0.5, 0.5, 0.0, 0.04])
    assert models.select_pool(cid, u, 0.05).tolist() == [0, 2, 3, 5]
    assert len(models.select_pool(cid, u, 0.05, np.random.default_rng(0), 2)) == 2


def _fake_cache(rng, H=60, W=70):
    n = H * W
    idx = np.sort(rng.choice(n, n // 2, replace=False))
    z = {k: rng.random(idx.size).astype(np.float32) for k in FEATURES}
    z.update(idx=idx, shape=np.array([H, W]), cid=rng.integers(0, 4, idx.size).astype(np.int8),
             tgt=rng.random(idx.size).astype(np.float32), bt108=(200 + 100 * rng.random(idx.size)).astype(np.float32))
    return z


def test_pool_is_reproducible_and_balanced():
    cfg = load_config(os.path.join(ROOT, "config.yaml"), {"training.per_scene": 300})
    caches = {s: _fake_cache(np.random.default_rng(i)) for i, s in enumerate(["A", "B", "C"])}
    X1, y1 = models.build_pool(["A", "B", "C"], caches.get, cfg)
    X2, y2 = models.build_pool(["A", "B", "C"], caches.get, cfg)
    assert X1.shape == (900, 4) and np.array_equal(X1, X2) and np.array_equal(y1, y2)


def test_mlr_recovers_coefficients():
    rng = np.random.default_rng(1)
    X = rng.random((20000, 4)).astype(np.float32)
    beta = np.array([0.04, 0.7, 0.03, -0.08, 0.06])
    y = beta[0] + X @ beta[1:] + 0.001 * rng.standard_normal(20000)
    m = models.MLRBaseline().fit(X, y, chunk=3000)
    assert np.allclose(m.beta, beta, atol=2e-3)
    assert np.allclose(m.predict(X), y, atol=0.01)


def test_rf_and_gbm_small():
    from seviri22.config import load_config as lc
    rng = np.random.default_rng(2)
    X = rng.random((3000, 4)).astype(np.float32)
    y = (X[:, 0] ** 2 * (1 + X[:, 2])).astype(np.float32)
    cfg = lc(os.path.join(ROOT, "config.yaml"))
    cfg["training"]["rf"].update(n_estimators=4, tree_batch=2, max_depth=6)
    cfg["training"]["gbm"].update(max_iter=20)
    with tempfile.TemporaryDirectory() as td:
        rf = models.train_rf(X, y, cfg, td, n_jobs=1)
    assert rf.n_estimators == 4 and len(rf.estimators_) == 4
    assert np.corrcoef(models.predict_chunked(rf, X, 1000), y)[0, 1] > 0.9
    assert np.corrcoef(models.train_gbm(X, y, cfg).predict(X), y)[0, 1] > 0.9


def test_metrics():
    rng = np.random.default_rng(3)
    x = rng.random(5000); y = 0.95 * x + 0.01 + 0.02 * rng.standard_normal(5000)
    st = metrics.stats(metrics.sums(x, y))
    assert abs(st["rmse"] - np.sqrt(np.mean((y - x) ** 2))) < 1e-12
    assert abs(st["r2"] - (1 - ((y - x) ** 2).sum() / ((x - x.mean()) ** 2).sum())) < 1e-12
    b, a = np.polyfit(x, y, 1)
    assert abs(st["slope"] - b) < 1e-9 and abs(st["intercept"] - a) < 1e-9
    assert abs(st["bias"] - np.mean(y - x)) < 1e-12
    # pooled sums equal the sums of the whole
    assert np.allclose(metrics.sums(x[:2000], y[:2000]) + metrics.sums(x[2000:], y[2000:]), metrics.sums(x, y))
    cm = metrics.class_masks(np.array([2, 2, 2, 1, 0]), np.array([280.0, 260.0, 220.0, 300.0, 290.0]))
    assert cm["Warm Cloud"].tolist() == [1, 0, 0, 0, 0] and cm["Cold Cloud"].tolist() == [0, 1, 0, 0, 0]
    assert cm["Ice Cloud"].tolist() == [0, 0, 1, 0, 0]


def test_scene_scores_runs():
    from seviri22.scene import to2d
    z = _fake_cache(np.random.default_rng(4))
    out = metrics.scene_scores(z, z["tgt"] + 0.01, to2d)
    assert set(out) >= {"All", "All Cloud"}
    assert abs(metrics.stats(out["All"][0])["rmse"] - 0.01) < 1e-6


def test_mask_orientation():
    rng = np.random.default_rng(5)
    truth = rng.integers(0, 3, (200, 200)).astype(np.float32)
    bt = np.where(truth == 2, 230.0, 290.0) + rng.standard_normal((200, 200))
    stored = cloudmask.FLIPS["flip rows"](truth)
    k, why = cloudmask.choose_orientation(cloudmask.orientation_scores(stored, bt))
    assert k == "flip rows" and why == "ok"


def test_rgb():
    a = np.full((4, 5), 0.2)
    out = rgb.cloud_phase_rgb(a, a, a)
    assert out.shape == (4, 5, 3) and np.allclose(out[..., 0], 0.5) and np.allclose(out[..., 2], 0.25)


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); n += 1
    print(f"{n} tests passed")
