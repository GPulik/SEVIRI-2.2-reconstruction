# SEVIRI 2.25 µm reconstruction

Code for *Reconstructing the Missing 2.25 µm Reflectance Channel for MSG SEVIRI* (G. Pulik and D. Rosenfeld).

The 2.25 µm reflectance is reconstructed from four SEVIRI predictors (1.6 and 0.6 µm reflectance, 3.9 µm solar
reflectance, 10.8 µm brightness temperature), trained against the MTG-I1 FCI 2.25 µm channel mapped onto the SEVIRI
grid with the SEVIRI point-spread function. The 0.6 and 1.6 µm reflectances and the target are divided by cos(SZA);
predictions are multiplied by cos(SZA) again.

## Contents

| Path | What it does |
|---|---|
| `config.yaml` | paths, calibration (MEIRINK-2023), PSF and model settings |
| `data/scenes.csv` | the 76 scenes and their use (37 training, 39 validation) |
| `seviri22/scene.py` | reads SEVIRI and FCI files, builds the predictors and the PSF-matched target, pixel caches |
| `seviri22/psf.py` | PSF from the pre-launch MTF; NaN-aware forward operator |
| `seviri22/cloudmask.py` | EUMETSAT cloud mask (CLM) for the evaluation classes |
| `seviri22/illum.py` | cos(SZA) normalization |
| `seviri22/models.py` | training pool, Random Forest, gradient boosting, linear baseline |
| `seviri22/metrics.py` | validation statistics and classes |
| `scripts/prepare.py` | opens the scene files and writes the pixel caches |
| `scripts/train.py` | trains the three models |
| `scripts/validate.py` | per-scene and pooled statistics on the validation scenes |
| `tests/test_core.py` | tests that need no satellite data |

## Use

The SEVIRI Level 1.5 native files, the FCI Level 1C chunks and the MSG cloud mask of each scene in `data/scenes.csv`
are available from the EUMETSAT Data Store; place them in `work/scenes/<scene>/` (the cloud mask in its `clm_grib/` subfolder). The SEVIRI MTF workbook (EUMETSAT)
goes to `data/MSG-4 SEVIRI_MTF.xls`.

```bash
pip install -r requirements.txt
python tests/test_core.py
python scripts/prepare.py        # pixel caches
python scripts/train.py          # RF, GBM, MLR
python scripts/validate.py       # results in work/results
```

## License

MIT (see `LICENSE`).
