"""Training pool, the three models (Random Forest, gradient boosting, multiple linear regression) and
chunked prediction."""
import os

import joblib
import numpy as np

from .config import FEATURES


def select_pool(cid, draw_u, ocean_keep, rng=None, per_scene=None):
    """Land, cloud, and clear ocean where the uniform draw is below ocean_keep (pixels the cloud mask left
    unprocessed carry no class and are excluded); optionally an equal random subset per scene."""
    ii = np.flatnonzero((cid == 1) | (cid == 2) | ((cid == 0) & (draw_u < ocean_keep)))
    if per_scene is not None and ii.size > per_scene:
        ii = np.sort(rng.choice(ii, per_scene, replace=False))
    return ii


def build_pool(scenes, load, cfg, prep=None):
    """Training pool from the pixel caches. `load(scene)` returns a cache dict; `prep(cache)`, if given, returns
    it with converted predictors and target (see illum.py). The draw order (one uniform field per scene, then
    the subset) fixes the pool for a given seed and scene order."""
    tr = cfg["training"]
    rng = np.random.default_rng(tr["seed"])
    per = tr.get("per_scene")
    X = np.empty((per * len(scenes), len(FEATURES)), np.float32) if per else None
    y = np.empty(per * len(scenes), np.float32) if per else None
    parts, n = [], 0
    for s in scenes:
        z = load(s)
        if prep is not None:
            z = prep(z)
        H, W = (int(v) for v in z["shape"])
        u = rng.random((H, W)).ravel()[z["idx"]]
        sel = select_pool(z["cid"], u, tr["ocean_keep_fraction"], rng, per)
        if per:
            for j, k in enumerate(FEATURES):
                X[n:n + sel.size, j] = z[k][sel]
            y[n:n + sel.size] = z["tgt"][sel]
        else:
            parts.append((np.column_stack([z[k][sel] for k in FEATURES]), z["tgt"][sel]))
        n += sel.size
        print(f"  pool {s}: {sel.size:,} px")
    if per:
        return X[:n], y[:n]
    return (np.concatenate([a for a, _ in parts]).astype(np.float32),
            np.concatenate([b for _, b in parts]).astype(np.float32))


class MLRBaseline:
    """Multiple linear regression by streamed normal equations (ordinary least squares)."""

    def __init__(self, k=len(FEATURES)):
        self.k = k
        self.XtX = np.zeros((k + 1, k + 1), np.float64)
        self.Xty = np.zeros(k + 1, np.float64)
        self.beta = None

    def add(self, X, y):
        Xb = np.empty((X.shape[0], self.k + 1), np.float64)
        Xb[:, 0] = 1.0
        Xb[:, 1:] = X
        self.XtX += Xb.T @ Xb
        self.Xty += Xb.T @ y.astype(np.float64)

    def solve(self):
        self.beta = np.linalg.solve(self.XtX + 1e-9 * np.eye(self.k + 1), self.Xty).astype(np.float32)
        return self.beta

    def fit(self, X, y, chunk=5_000_000):
        for i in range(0, X.shape[0], chunk):
            self.add(X[i:i + chunk], y[i:i + chunk])
        self.solve()
        return self

    def predict(self, X):
        if self.beta is None:
            self.solve()
        return (self.beta[0] + X.astype(np.float32) @ self.beta[1:]).astype(np.float32)


def train_rf(X, y, cfg, work_dir, n_jobs=-1):
    """Random Forest fitted in batches of trees (seed random_state*1000 + batch) and joined; this keeps
    memory bounded and reproduces the published forest for the same pool."""
    from sklearn.ensemble import RandomForestRegressor
    p = dict(cfg["training"]["rf"])
    n_trees, batch, seed = p.pop("n_estimators"), p.pop("tree_batch"), p.pop("random_state")
    parts = os.path.join(work_dir, "rf_parts")
    os.makedirs(parts, exist_ok=True)
    rf = None
    for b in range((n_trees + batch - 1) // batch):
        k = min(batch, n_trees - b * batch)
        path = os.path.join(parts, f"part_{b:03d}.joblib")
        if os.path.exists(path):
            part = joblib.load(path)
        else:
            part = RandomForestRegressor(n_estimators=k, random_state=seed * 1000 + b, n_jobs=n_jobs, **p).fit(X, y)
            joblib.dump(part, path, compress=3)
        print(f"  RF batch {b + 1}: {k} trees")
        if rf is None:
            rf = part
        else:
            rf.estimators_ += part.estimators_
    rf.n_estimators = len(rf.estimators_)
    return rf


def train_gbm(X, y, cfg):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(**cfg["training"]["gbm"]).fit(X, y)


def predict_chunked(model, X, chunk=1_000_000):
    out = np.empty(X.shape[0], np.float32)
    for i in range(0, X.shape[0], chunk):
        out[i:i + chunk] = model.predict(X[i:i + chunk])
    return out


def features(z):
    return np.column_stack([z[k] for k in FEATURES]).astype(np.float32)
