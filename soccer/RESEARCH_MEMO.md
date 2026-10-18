# Soccer Movement Forecasting and Measurement Reliability

## Research question
Does spatial information about teammates, opponents, and the ball improve 1- and 2-second player-position forecasts beyond recent player movement? How does a temporary observation blackout affect extrapolation?

## Data and separation
Metrica Sports sample tracking data, repository commit `e706dd506b360d69d9d123d5b8026e7294b13996`. All three matches are real anonymized positional tracking data, recorded at 25 Hz. Coordinates are converted using 105 by 68 metres. This is not skeletal pose data. Game 3 uses metadata-driven channel mappings that change with substitutions; we retain stable player IDs rather than assume columns always represent the same player.

Match 1 is used exclusively for training. Match 2 supplies a separate empirical 90th-percentile error radius for each model and horizon. Match 3 is the held-out evaluation set. Hyperparameters were fixed before viewing Match 3 results. No hyperparameter search or model selection on the test match was performed. Labels are future displacement vectors, never input features. Sample origins are one second apart; prediction windows overlap, especially at two seconds. They are dependent observations, not independent experimental units.

## Models
Stationary and constant-velocity baselines; standardized ridge regression using movement history; boosted-tree regressors using the same history; and boosted trees with interaction features. Two separate scalar regressors predict horizontal and vertical displacement. History includes position, backward velocities over 0.2, 0.4 and 1 seconds, and a backward acceleration proxy. Context adds up to three nearest opponents and two teammates (relative locations and velocities) plus relative ball position. Missing context values are zero-filled with explicit validity flags. No attack-direction or goalkeeper-position normalization is applied. These are tree models on temporal features, not a recurrent neural network or trained pose estimator.

## Main result
For one-second forecasts, constant velocity has mean endpoint error 0.567 m; boosted trees with movement history 0.411 m; and boosted trees with interactions 0.405 m. Adding interactions changes mean error by 1.44% relative to history-only boosted trees (positive means improvement). This is an observed held-out-match result, not an assumed benefit of a more complex model.

## Uncertainty and reliability
Every forecast receives a circle with radius given by the calibration match's 90th-percentile residual. For the one-second interaction model the radius is 0.683 m and held-out empirical coverage is 84.1%. This is a transfer-calibration diagnostic. Temporal dependence and differences between matches mean that nominal 90% coverage is not a guaranteed conformal coverage statement. A circular region also ignores direction-dependent uncertainty.

Mean-error intervals use 1,000 bootstrap resamples of 60-second bins within Match 3; all players in a bin are kept together. These intervals describe within-match resampling variability, not variation across a population of matches. They do not establish a statistically significant advantage between models.

The blackout experiment withholds the focal player's final 0–1 seconds before the prediction origin. A constant-velocity forecast extrapolates from the last available observations to the original origin plus one second. All gap conditions use a shared eligible origin/player set, allowing a paired sensitivity comparison. This is an artificial observation outage, not a model of actual camera missingness, and it evaluates this baseline only.

## Sample construction and limitations
Complete focal positions are required at four history points and the prediction endpoint; history coordinates beyond a 10% pitch margin are excluded. No future speed threshold is used to select favorable test cases. Raw missingness also includes inactive and substituted players, so it is not a sensor-failure rate. Context can be missing. The analysis includes goalkeepers, stoppages, and out-of-play periods because an in-play classifier has not been implemented. Endpoint error is not path error or a measure of tactical value. Ground truth is the vendor's tracking measurement, not error-free physical movement. Opponent-distance groups are descriptive and do not isolate defensive pressure causally. Blackout conditions use a shared set of valid history endpoints.

Only three sample matches are available and one is tested. Athlete identities are anonymized; IDs across matches are not assumed to identify the same people. Performance cannot be generalized to leagues, demographic groups, unseen camera systems, or professional scouting decisions. No injury risk, optimal technique, causal tactical effect, or skeletal pose accuracy is inferred.

## Reproduction and provenance
Run `python run.py PATH_TO_METRICA_REPO`, `python -m unittest discover -s tests -v`, then `python build_report.py`. The archive contains source input files for offline reconstruction as described in README. `output/manifest.json` records input hashes and the pinned commit. `metrics.csv`, `test_predictions.csv.gz`, `opponent_distance_strata.csv`, and the data audit support the narrative. Attribution: Metrica Sports https://github.com/metrica-sports/sample-data and its README usage notice. Original code is MIT licensed; third-party data retain their source terms. No blanket open-data license is invented.

## AI assistance
OpenAI Codex assisted with implementation, drafting, and checks.
