# Basketball Shooting Mechanics and Movement Consistency

An exploratory research study by **Nicholas Reign Lugu** using real MLSE Sport Performance Lab free-throw pose data. Main question: how does movement vary between athletes and between shots by the same athlete?

## Results and scope

- Source: **583 trials, five non-professional participants, two sessions**.
- Main analysis: **438 eligible trials** from the five-person 2025 session; 20 excluded for unrepaired core-marker gaps.
- Older session: 125 trials from the same P0001 at a different frame rate and with a different pose model; reported separately, never treated as a sixth person.
- Random-intercept models partition movement variation, with trial-progress adjustment. Only five groups: ICCs are descriptive, not measurement-system accuracy or broad population estimates.
- Retrospective made/missed classifiers are evaluated with later-shot and whole-athlete holdouts against simple rate baselines. Small improvements do not establish coaching usefulness.

Read [RESEARCH_MEMO.md](RESEARCH_MEMO.md) and open `docs/index.html` directly in a browser for skeleton animations, trajectories, variance components, outcome metrics, and session summaries.

## Reproduce offline

```bash
python -m pip install -r requirements-lock.txt
python run.py
python -m unittest discover -s tests -v
python build_report.py
node tests/check_explorer.cjs  # optional UI logic check
```

Bundled NPZ inputs preserve full-precision source coordinates for 13 selected body markers and the ball, timestamps, and trial metadata across all 583 trials. Unused finger/face/foot markers are omitted. These are real source-derived arrays, not synthetic data. The pipeline reads these compact bundles without a live download. They are float64 in original units; analysis converts feet to metres and milliseconds to seconds.

To reconstruct the bundles from original JSON files, clone https://github.com/Sport-Performance-Lab/SPL-Open-Data at the commit in `data/provenance.json`, then run `python prepare_data.py /path/to/SPL-Open-Data`. Each original trial JSON hash is recorded in `data/index.json`. Different JSON spacing may change byte hashes even if values match; this study uses the pinned original files.

## Methods

Short internal gaps (up to three frames) are interpolated. Quadratic Savitzky–Golay smoothing uses nominal 0.15 seconds, with 0.10/0.25-second sensitivity checks. Movements align to peak upward bilateral wrist velocity in an elevated-arm window. This is a phase proxy, **not an annotated ball-release time**. Curves span -0.6 to +0.4 seconds around it.

The main study uses REML random participant intercepts with linear trial-progress adjustment, PCA of aligned movement curves, regularized logistic regression and constrained random forests. Whole trials and whole people are held out; frames are never randomly split across training/test sets. Ball landing coordinates, entry angle, ball trajectory and participant IDs are excluded from learned predictors. Retrospective feature extraction includes the selected movement window and does not support live pre-release forecasting.

The analysis does not infer optimal technique, causal effects, injury risk, or professional selection. Only five athletes are available. Chronological shots may remain serially dependent, and a session comparison is confounded with changes in measurement system and frame rate.

## Attribution and terms

Source: **Maple Leaf Sports & Entertainment (MLSE), Sport Performance Lab, SPL Open Data**. https://github.com/Sport-Performance-Lab/SPL-Open-Data

Derived basketball inputs and study materials are distributed under **CC BY-NC-SA 4.0 and the original source usage terms**, retained in `data/SOURCE_LICENSE.txt`. See https://creativecommons.org/licenses/by-nc-sa/4.0/. Changes: selected-marker extraction, unit conversion in analysis, gap repair, smoothing, alignment, and derived features/figures. Source documentation is preserved. No provider or institutional endorsement is implied.

## AI assistance

OpenAI Codex assisted implementation, drafting and checks. Numerical results come from executed code. Independent human peer review and Claude Code use are not claimed. Review and understand the work before describing personal contributions in an application.
