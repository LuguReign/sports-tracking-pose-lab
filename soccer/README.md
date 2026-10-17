# Soccer Movement Lab

**Predicting soccer player movement and examining measurement reliability.** Independent research portfolio prepared for Nicholas Reign Lugu, with interests in statistical modeling and sports tracking data.

This pilot uses real, anonymized Metrica Sports positional tracking. It compares five methods at 1- and 2-second horizons, asks whether nearby players and the ball add predictive information, and measures sensitivity to artificially missing observations. It does **not** use skeletal pose data.

## Explore

Open `docs/index.html` directly in a browser. It is self-contained and works offline, with six animated held-out examples, model selection, uncertainty circles, numerical results, and a blackout sensitivity table. Read `RESEARCH_MEMO.md` for interpretation and limitations.

## Reproduce

Python 3.10+ is required. `requirements-lock.txt` records the exact directly used package versions; use it instead of `requirements.txt` to match this run. The full run was checked with the versions in `output/environment.json`.

```bash
python -m pip install -r requirements.txt
python run.py
python -m unittest discover -s tests -v
python build_report.py
```

`data/metrica-source.zip` includes the unmodified source data, original README, and a source-commit record. The pipeline extracts it to `data/metrica` when first run. No live API key or download is required. To use your own checkout of the same data, run `python run.py /absolute/path/to/metrica-sports/sample-data`.

## Design

- Match 1: train all fitted models.
- Match 2: calibrate an empirical 90th-percentile endpoint-error radius separately by model and horizon.
- Match 3: final evaluation; no hyperparameter tuning or model selection on this match.
- Models: stationary; constant velocity; ridge regression on backward movement features; boosted trees on history; boosted trees adding teammate, opponent, and ball features.
- One origin per second per eligible player. Metric: Euclidean error of the predicted endpoint, in metres. Repeated player observations and overlapping windows remain dependent.
- Game 3 uses XML-defined player channels and changing data formats. Stable IDs are respected through substitutions.

All hyperparameters are fixed in code. Calibration radii are transfer diagnostics, not guaranteed coverage claims. Intervals in the plot resample 60-second bins within one test match; they do not measure across-match uncertainty. The blackout experiment compares a constant-velocity baseline on shared eligible examples and does not evaluate ML dropout robustness.

## Files

- `run.py`: train/evaluate; audit and record provenance.
- `src/data.py`: source readers, feature extraction, sampling.
- `src/blackout.py`: paired observation-blackout experiment.
- `build_report.py`: numerical summary, chart, research memo, animated HTML explorer.
- `tests/`: velocity/target correctness, future-feature leakage prevention, period boundaries, missing observations, and paired masking checks.
- `output/`: full test errors, aggregate metrics, block-bootstrap intervals, calibration diagnostics, descriptive opponent-distance strata, hashes, and example animation data.

## Assumptions and open research

The sample is only three matches, with one held-out evaluation match. No population-level generalization is warranted. Data quality can affect estimated velocity; tracking is the reference measurement, not physical ground truth. Goalkeepers and stoppages are included. Player identities cannot be linked across matches. Proximity is a descriptive interaction measure, not a causal measure of defensive pressure.

Next steps: additional independent matches; in-play segmentation; direction and role normalization; multistep trajectory models; context ablations; dependence-aware uncertainty; a compact neural sequence benchmark; verified body-orientation or pose data. These are planned extensions, not completed claims.

## Source and terms

Metrica Sports: https://github.com/metrica-sports/sample-data
Pinned source commit: `e706dd506b360d69d9d123d5b8026e7294b13996`.

The source README requests responsible use and public acknowledgement. It does not supply a blanket MIT license for the data. Source files retain their own terms. The MIT license in this project covers original analytical code only. No affiliation or endorsement by Metrica Sports, Dr. Schuckers, or a university is implied.

## AI assistance

OpenAI Codex assisted implementation, drafting, and checks. Results were computed from real source files by the supplied code. No independent human review or Claude Code usage is claimed. The applicant should review the methods and be able to explain the work before presenting it in an application.

Optional UI logic check with Node.js: `node tests/check_explorer.cjs` executes every example, model, animation frame, and horizon filter without network access. This checks logic and finite drawing coordinates, not browser rendering.
