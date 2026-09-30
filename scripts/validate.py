#!/usr/bin/env python3
"""Score the trained models on the validation scenes.

  python scripts/validate.py

Writes to <out_dir>:
  per_scene.csv      per scene, class and model: N, R2, RMSE, MAE, relRMSE, bias, slope, intercept (Tables S1-S3)
  pooled.csv         the same pooled over all validation scenes and per year (Tables 2 and 4)
  scene_level.csv    per class: scene-median R2 and slope per model, scenes where RF beats MLR, exact paired
                     Wilcoxon p (Section 3.3)"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seviri22.config import add_common_args, config_from_args, ensure_dirs, scene_list, CLASSES  # noqa: E402
from seviri22.scene import load_cache, to2d  # noqa: E402
from seviri22 import models as M  # noqa: E402
from seviri22 import metrics as S  # noqa: E402


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--scenes", nargs="*", default=None)
    args = ap.parse_args()
    cfg = config_from_args(args)
    ensure_dirs(cfg)
    md, od = cfg["paths"]["models_dir"], cfg["paths"]["out_dir"]
    models = {n: joblib.load(os.path.join(md, f"{f}.joblib"))
              for n, f in (("RF", "rf"), ("GBM", "gbm"), ("MLR", "mlr")) if os.path.exists(os.path.join(md, f"{f}.joblib"))}
    if "RF" not in models:
        raise SystemExit(f"no rf.joblib in {md} (run train.py)")
    v = cfg["validation"]
    scenes = args.scenes or scene_list(cfg, "validation")
    rows, pooled = [], {}
    for i, s in enumerate(scenes, 1):
        z = load_cache(cfg, s)
        X = M.features(z)
        for name, model in models.items():
            pred = M.predict_chunked(model, X)
            for c, (a, r) in S.scene_scores(z, pred, to2d, v["rel_window"], v["rel_floor"]).items():
                st = S.stats(a)
                rows.append(dict(model=name, scene=s[6:], year=s[6:10], cls=c, relrmse=S.rel_value(r), **st))
                for grp in ("all", s[6:10]):
                    k = (name, grp, c)
                    acc = pooled.setdefault(k, [np.zeros(8), np.zeros(3)])
                    acc[0] += a
                    acc[1] += r
        print(f"[{i}/{len(scenes)}] {s}")
    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(od, "per_scene.csv"), index=False)
    pr = [dict(model=n, group=g, cls=c, relrmse=S.rel_value(r), **S.stats(a)) for (n, g, c), (a, r) in pooled.items()]
    pd.DataFrame(pr).sort_values(["group", "model", "cls"]).to_csv(os.path.join(od, "pooled.csv"), index=False)

    from scipy.stats import wilcoxon
    sl = []
    for c in CLASSES:
        d = per[per.cls == c].pivot(index="scene", columns="model", values="r2")
        row = {"cls": c, "n_scenes": len(d)}
        for n in models:
            row[f"median_r2_{n}"] = d[n].median()
            row[f"median_slope_{n}"] = per[(per.cls == c) & (per.model == n)].slope.median()
        if "MLR" in d:
            row["rf_higher_than_mlr"] = int((d["RF"] > d["MLR"]).sum())
            row["wilcoxon_p"] = wilcoxon(d["RF"], d["MLR"], method="exact").pvalue if len(d) <= 50 else \
                wilcoxon(d["RF"], d["MLR"]).pvalue
        sl.append(row)
    pd.DataFrame(sl).to_csv(os.path.join(od, "scene_level.csv"), index=False)
    allp = pd.DataFrame(pr)
    for n in models:
        r = allp[(allp.model == n) & (allp.group == "all") & (allp.cls == "All Cloud")].iloc[0]
        print(f"{n}: all-cloud pooled R2 {r.r2:.4f}, RMSE {r.rmse:.4f}, slope {r.slope:.3f}")
    print(f"results in {od}")


if __name__ == "__main__":
    main()
