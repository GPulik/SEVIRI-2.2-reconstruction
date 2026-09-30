# SEVIRI 2.25 µm reconstruction

Code for *Cross-Sensor Machine-Learning Reconstruction of the Missing 2.25 µm Reflectance Channel for MSG SEVIRI*
(G. Pulik and D. Rosenfeld).

The Meteosat Second Generation imager SEVIRI has no 2.25 µm channel. This code reconstructs it from four SEVIRI
predictors (1.6 and 0.6 µm reflectance, 3.9 µm solar reflectance, 10.8 µm brightness temperature), trained against
the 2.25 µm channel of the MTG-I1 Flexible Combined Imager (FCI) mapped onto the SEVIRI grid with the SEVIRI
point-spread function. The synthetic channel completes the Cloud Phase RGB for SEVIRI.

## Contents

| Path | What it does |
|---|---|
| `config.yaml` | all settings: paths, calibration, spectral response, PSF, model parameters |
| `data/scenes.csv` | the 76 co-located scenes and their use (37 training, 39 validation) |
| `seviri22/psf.py` | PSF from the pre-launch MTF; NaN-aware forward operator (Equations 1-4) |
| `seviri22/scene.py` | SEVIRI predictors, FCI target on the SEVIRI grid, pixel caches |
| `seviri22/cloudmask.py` | EUMETSAT cloud mask (CLM), oriented against the scene's 10.8 µm BT |
| `seviri22/datastore.py` | EUMETSAT Data Store download (eumdac) |
| `seviri22/models.py` | training pool, Random Forest, gradient boosting, linear baseline |
| `seviri22/metrics.py` | validation statistics and classes (Equations S1-S5) |
| `scripts/download.py` | download SEVIRI, FCI and CLM for the scene list |
| `scripts/prepare.py` | build cloud masks and pixel caches |
| `scripts/train.py` | train the three models |
| `scripts/validate.py` | per-scene and pooled statistics (Tables 2, 4, S1-S3) |
| `scripts/operator_test.py` | test of the mapping operator on the 1.6 µm band (Table S4, Figure S4) |
| `scripts/apply.py` | synthetic 2.25 µm and Cloud Phase RGB for any SEVIRI file |
| `tests/test_core.py` | tests that need no satellite data |

## Setup

```bash
pip install -r requirements.txt
python tests/test_core.py
```

Two inputs are not in the repository:

* **EUMETSAT Data Store credentials.** Set `EUMETSAT_KEY` and `EUMETSAT_SECRET` to the consumer key and secret from
  https://api.eumetsat.int/api-key/.
* **The SEVIRI MTF workbook** (EUMETSAT, *SEVIRI Modulation Transfer Function (MTF) Characterisations for MSG*,
  https://www.eumetsat.int/msg-calibration). Save it as `data/MSG-4 SEVIRI_MTF.xls`, or set `paths.mtf_xls`.

## Reproducing the paper

```bash
python scripts/download.py                 # ~1.2 GB per scene
python scripts/prepare.py                  # cloud masks and pixel caches
python scripts/train.py                    # RF, GBM, MLR (high-memory machine)
python scripts/validate.py                 # results in work/results
python scripts/operator_test.py            # needs the FCI files of the validation scenes
```

`prepare.py --delete-fci` removes each scene's FCI files once its cache is written. Keep them for the validation
scenes if you want to run the operator test.

## Applying the model

```bash
python scripts/apply.py <SEVIRI .nat file> --model work/models/rf.joblib
python scripts/apply.py <Meteosat-9 .nat file> --rsr-platform Meteosat-9
```

The output is the synthetic 2.25 µm reflectance (NetCDF) and a Cloud Phase RGB (PNG).

## Settings that matter

* **`--rsr-platform`** (`seviri.rsr_platform`): the SEVIRI unit whose spectral response is used by pyspectral for the
  3.9 µm solar reflectance. The paper used `Meteosat-11`. Pass the platform that acquired the data when applying the
  model elsewhere.
* **`--calib-mode`** (`seviri.calib_mode`): solar-channel calibration passed to satpy. The models were trained on
  `MEIRINK-2023`, which is continuous across the March 2026 change of the operational calibration; inputs must use
  the same calibration. Thermal channels always use the nominal calibration.
* **`psf.scale`**: width of the PSF relative to the pre-launch kernel (shape kept); 1.0 in the paper.

## Limits

The models were trained and validated on Meteosat-10 at 0° (2025 and January-September 2026), daytime only
(solar zenith angle below 82°). Other MSG units, positions and years are not validated. The synthetic channel is
not an absolutely calibrated substitute for a measured one (pooled all-cloud slope 0.965).

## Citation

See `CITATION.cff`.

## License

MIT, see `LICENSE`. SEVIRI, FCI and CLM data are EUMETSAT data under EUMETSAT's own licence.
