"""Configuration: one YAML file, paths resolved relative to it, command-line overrides on top."""
import copy
import os

import yaml

FEATURES = ["vis16", "refl39", "temp", "vis06"]           # model input order (Table 1)
CLASSES = ["All", "All Cloud", "Warm Cloud", "Cold Cloud", "Ice Cloud", "Land", "Ocean"]


def load_config(path="config.yaml", overrides=None):
    """Read the YAML configuration. `overrides` is a dict of dotted keys, e.g.
    {"seviri.rsr_platform": "Meteosat-10"}."""
    path = os.path.abspath(path)
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    base = os.path.dirname(path)
    for k, v in cfg.get("paths", {}).items():
        if isinstance(v, str) and not os.path.isabs(v):
            cfg["paths"][k] = os.path.join(base, v)
    for key, val in (overrides or {}).items():
        if val is None:
            continue
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return cfg


def add_common_args(parser):
    """Arguments shared by all scripts."""
    parser.add_argument("--config", default="config.yaml", help="configuration file")
    parser.add_argument("--rsr-platform", default=None,
                        help="platform whose SEVIRI spectral response is used for the 3.9 um reflectance "
                             "(default from the config, Meteosat-11)")
    parser.add_argument("--calib-mode", default=None,
                        help="SEVIRI solar-channel calibration passed to satpy (default MEIRINK-2023)")
    return parser


def config_from_args(args):
    return load_config(args.config, {"seviri.rsr_platform": args.rsr_platform,
                                     "seviri.calib_mode": args.calib_mode})


def scene_list(cfg, use=None):
    import pandas as pd
    df = pd.read_csv(cfg["paths"]["scene_list"])
    if use:
        df = df[df["use"] == use]
    return sorted(df["scene"])


def ensure_dirs(cfg):
    for k in ("scenes_dir", "masks_dir", "cache_dir", "models_dir", "out_dir"):
        os.makedirs(cfg["paths"][k], exist_ok=True)


def copy_cfg(cfg):
    return copy.deepcopy(cfg)
