#!/usr/bin/env python3
"""Train the Random Forest, the gradient-boosted ensemble and the linear baseline on the training scenes.

  python scripts/train.py                  # all three
  python scripts/train.py --models rf mlr

The pool: from each training scene, land and cloud pixels and a random 5% of clear-sky ocean; then an equal
random subset per scene (config: training.per_scene), seed 42, scenes in sorted order. With the published
settings the pool has 66,277,064 pixels. The RF needs a high-memory machine; trees are fitted in batches to
bound the peak, and --jobs sets the number of parallel tree builders. With the article's convention (config
training.convention = norm_all) the 0.6 and 1.6 um inputs and the target are divided by cos(SZA) before training,
so the models predict the normalised reflectance; multiply a prediction by cos(SZA) (validate.py and apply.py do)."""
import argparse
import json
import os
import sys
import time

import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seviri22.config import add_common_args, config_from_args, ensure_dirs, scene_list, FEATURES  # noqa: E402
from seviri22.scene import load_cache  # noqa: E402
from seviri22 import models as M  # noqa: E402
from seviri22 import illum as I  # noqa: E402

KEYS = ("idx", "shape", "cid", "tgt") + tuple(FEATURES)


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--models", nargs="*", default=["rf", "gbm", "mlr"], choices=["rf", "gbm", "mlr"])
    ap.add_argument("--jobs", type=int, default=-1, help="parallel tree builders for the RF")
    args = ap.parse_args()
    cfg = config_from_args(args)
    ensure_dirs(cfg)
    md = cfg["paths"]["models_dir"]
    train = scene_list(cfg, "training")
    missing = [s for s in train if not os.path.exists(os.path.join(cfg["paths"]["cache_dir"], f"{s}.npz"))]
    if missing:
        raise SystemExit(f"{len(missing)} training caches missing (run prepare.py): {missing[:5]}")
    mode = cfg["training"].get("convention", "published")
    keys = KEYS if mode == "published" else KEYS + ("sza",)
    prep = None if mode == "published" else (lambda z: I.convert(z, mode)[0])
    X, y = M.build_pool(train, lambda s: load_cache(cfg, s, keys), cfg, prep=prep)
    print(f"training pool: {X.shape[0]:,} pixels from {len(train)} scenes, convention {mode}")
    json.dump({"scenes": train, "n_pool": int(X.shape[0]), "features": FEATURES, "convention": mode,
               "calib_mode": cfg["seviri"]["calib_mode"], "rsr_platform": cfg["seviri"]["rsr_platform"],
               "training": cfg["training"]}, open(os.path.join(md, "pool_config.json"), "w"), indent=1)
    if "mlr" in args.models:
        mlr = M.MLRBaseline().fit(X, y)
        joblib.dump(mlr, os.path.join(md, "mlr.joblib"))
        print(f"MLR coefficients (intercept, {', '.join(FEATURES)}): {mlr.beta.round(4).tolist()}")
    if "gbm" in args.models:
        t0 = time.time()
        joblib.dump(M.train_gbm(X, y, cfg), os.path.join(md, "gbm.joblib"))
        print(f"GBM trained in {(time.time() - t0) / 60:.0f} min")
    if "rf" in args.models:
        t0 = time.time()
        rf = M.train_rf(X, y, cfg, md, n_jobs=args.jobs)
        joblib.dump(rf, os.path.join(md, "rf.joblib"))
        print(f"RF trained ({rf.n_estimators} trees) in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
