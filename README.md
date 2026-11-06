# Heavy Rainfall and Urban Mobility Recovery in New York City

## 1. Project Overview

This package reproduces the GE5219 final report's analyses of taxi and ride-hailing activity, post-rain recovery and recurring local shortfalls. The default run starts from the supplied, frozen hourly observations, fitted baseline predictions and normal-reference residuals. It recalculates recovery metrics, peer comparisons, recurrence classifications, six figures and Table 1.

Three samples remain separate: the August–October 2023 Pilot, the January 2021–August 2026 ride-hailing source panel (normal-centered comparisons through December 2025), and the 2022–2023 matched taxi/ride-hailing comparison.

## 2. Data Sources

- **NYC Taxi and Limousine Commission (TLC):** Yellow and Green taxi trip records and High Volume For-Hire Vehicle records for Uber and Lyft. Activity is completed pickups by Taxi Zone and hour.
- **NOAA MRMS:** MultiSensor QPE, one-hour Pass 2 rainfall. Grid-cell intersections provide area-weighted Taxi Zone rainfall.
- **NOAA station observations:** four-station weather records used in the Pilot and in weather-quality and normal-period eligibility checks. These checks are retained in the frozen input metadata.
- **NYC TLC Taxi Zone boundaries:** the original polygon coordinates and zone names. The study excludes Newark Airport from the NYC maps.

The supplied data are research intermediates derived from these sources. They contain observations, predictions, eligibility metadata and fixed comparison rosters; final recovery scores and report answers are calculated during execution. `data/input_manifest.csv` records the origin and contents of each input.

## 3. Folder Structure

| Path | Purpose |
|---|---|
| `run_pipeline.py` | Unified command-line entry point |
| `src/raw_processing.py` | Optional TLC cleaning, MRMS decoding and area aggregation, event segmentation and baseline fitting |
| `src/recovery_metrics.py` | Gap, threshold-crossing, reversal and recovery measures |
| `src/peer_selection.py` | Original deterministic peer selection |
| `src/service_metrics.py` | Common six-hour and 24-hour service measures |
| `src/analysis.py` | Main, service, recurrence, normal-day and local-reference analyses |
| `src/figures.py` | Six report figures and Table 1 |
| `data/` | Self-contained frozen inputs |
| `raw_inputs.example.json` | Optional raw-preprocessing job syntax |
| `requirements.txt` | Python dependencies |

## 4. Requirements

Use Python 3.11 or later and install `requirements.txt`. The analysis was executed with Python 3.14.6. NumPy and pandas handle the data; Matplotlib and Shapely generate the figures. DuckDB, Pillow, pyproj and scikit-learn support optional raw preprocessing. Arial is used when installed; otherwise Matplotlib uses its available sans-serif font. Install Arial to reproduce the report typography.

## 5. Input Data

All default inputs are included in `data/`. Keep the filenames unchanged when using another input directory. CSV files are UTF-8, comma-separated with a header; `.csv.gz` files use gzip compression. Empty numeric cells mean unavailable observations, not zero. Boolean fields contain `True` or `False`.

| Input file(s) | Contents |
|---|---|
| `event_registry.csv` | Local rain episodes, timing, rainfall, weather checks and baseline eligibility |
| `hourly_predictions.csv.gz` | Main ride-hailing `Y`, `P`, `b`, hourly rainfall and availability around each event |
| `weather_groups.csv` | Complete weather-process identifiers and chronological boundaries |
| `recovery_cohort.csv` | Fixed 208-zone analytical cohort |
| `service_cases.csv`, `service_hourly_predictions.csv.gz` | Fixed service cases and both services' hourly observations through hour 24 |
| `pilot_hourly_baselines.csv` | Corrected Pilot observations and reference medians for the displayed zone-hour |
| `normal_hourly_predictions.csv.gz`, `normal_window_metadata.csv.gz` | Paired storm/normal trajectories, leave-date-out centers and eligible normal windows |
| `normal_fixed_peers.csv`, `normal_target_membership.csv` | Fixed five-zone peer comparisons and original metric-specific candidate membership |
| `local_reference_centers.csv`, `local_reference_fixed_peers.csv` | Original/corrected reference centers and fixed pairs for the local CW0249 comparison |
| `map_geometry_2263.json`, `zone_labels.csv` | Original polygons, names, boroughs and label anchors |
| `study_periods.json` | Source/sample dates, timezone and the fixed Pilot display event/hour |
| `mrms_polygon_weights.npz`, `mrms_grid_geometry.json` | Optional raw-preprocessing area weights and original MRMS crop/grid definition |

Important field conventions:

| Field | Meaning, unit and format |
|---|---|
| `zone_id`, `target_zone`, `peer_zone` | Integer TLC location identifier |
| `event_id`, `target_event`, `peer_event` | String local-episode identifier |
| `canonical_group` | String identifier joining local episodes in one weather process |
| `hour_utc`, `h1_utc`, `start_utc`, `end_utc` | ISO 8601 UTC timestamp; an episode's end is exclusive and `h1_utc` starts recovery hour 1 |
| `hour_NYC`, `h1_NYC`, Pilot `hour` | Local New York timestamp; Pilot `hour` is timezone-naive `YYYY-MM-DD HH:MM:SS` interpreted in `America/New_York` |
| `offset_from_end`, `post_hour` | Integer offset: offset 0 is recovery hour 1; `post_hour = offset + 1` only in the post-rain segment |
| `segment` | Main input: `pre`, `rain`, `post`; service input: `pre_rain`, `original_rain`, `post` |
| `service` | Pilot: `Taxi` or `Ride-hailing`; service comparison: `Taxi` or `RideHailing` |
| `Y`, `actual` | Observed completed pickups, trips per zone-hour |
| `P`, `baseline` | Expected completed pickups, trips per zone-hour; values below 10 are unsupported |
| `b`, `normal_error_center` | Normal-reference mean residual ratio, dimensionless |
| `b_minus_d` | Reference center excluding the paired normal date, dimensionless |
| `raw_gap`, `old_b`, `corrected_b` | Raw gap and original/corrected reference centers, dimensionless |
| `rain_mean_mm`, `peak_mm`, `total_mm` | Area-mean hourly rainfall, peak hourly accumulation and episode sum, millimetres |
| `duration_h` | Local rain-episode duration, hours |
| `raw_usable`, `centered_usable`, `source_available` | Boolean availability/eligibility flags |
| `raw_status`, `centered_status` | Service-input eligibility status; `VALID` identifies usable records |
| `core_weather_verified` | Boolean confirmed-liquid-weather flag for the broad descriptive sample |
| `primary_model_weather_eligible` | Boolean eligibility for the main verified, closed-weather-group comparisons |
| `group_boundary_closed`, `training_support` | Boolean complete weather-process boundary and adequate baseline training support flags |
| `period` | `exploration` or `later`; event/peer outputs also use `DESCRIPTIVE_ONLY`, and period summaries use `all` |
| `trajectory` | `storm` or `normal` in paired normal-window data |
| `normal_window_id`, `normal_anchor_date` | String identifier and `YYYY-MM-DD` anchor date of a paired normal window |
| `calendar_weather_source_eligible` | Boolean normal-window eligibility after calendar, weather and source checks |
| `reference_n`, `reference_dates_count`, `reference_weeks_count` | Counts of valid reference observations, dates or weeks, respectively |
| `S80_point_clue`, `Sprogress_point_clue` | Boolean original candidate membership, kept fixed for normal-day comparisons |
| `label_x_2263`, `label_y_2263` | Display anchors in EPSG:2263, US survey feet |

`map_geometry_2263.json` is a GeoJSON FeatureCollection with Polygon/MultiPolygon coordinates in **EPSG:2263**, not longitude/latitude.
The Pilot map requires `reference_n >= 4` and `baseline >= 10`; other zone-hours remain unavailable.

Command-line parameters:

| Argument | Syntax and meaning |
|---|---|
| `--input-root PATH` | Input directory; defaults to `data/` beside the script, independently of the current working directory |
| `--output-dir PATH` | Destination directory, created if absent; defaults to `outputs/` in the current working directory |
| `--stage analysis\|preprocess` | Run the supplied-input analyses (default `analysis`) or explicit raw-preprocessing jobs |
| `--raw-config FILE.json` | Required for `preprocess`; JSON file listing local raw input jobs |
| `--skip-figures` | Flag producing numerical analyses without the report figures or Table 1 |

Use a fresh output directory for each run. Existing files with the same output names are replaced.

## 6. How to Run

From this package directory:

```bash
python -m pip install -r requirements.txt
python run_pipeline.py --input-root data --output-dir outputs
```

For numerical results alone:

```bash
python run_pipeline.py --output-dir numeric_outputs --skip-figures
```

To run from another working directory, provide the path to `run_pipeline.py`; its default input directory still resolves beside the script.

Optional raw preprocessing uses locally supplied files:

```bash
python run_pipeline.py --stage preprocess --raw-config raw_inputs.example.json --input-root . --output-dir prepared
```

Place the example raw files in `raw_data/` and edit the job list to match files actually present. Paths inside the JSON are relative to `--input-root`; no download occurs. The example uses the package directory as its root so it can also read the included static inputs in `data/`.

| Raw configuration key | Syntax and meaning |
|---|---|
| `zone_labels` | CSV path containing integer `zone_id` values to retain |
| `trip_jobs` | List of objects with `source` (`yellow`, `green`, `fhvhv`), `month` (`YYYY-MM`) and `file` (TLC Parquet path) |
| `rainfall.files` | List of MRMS one-hour Pass 2 `.grib2.gz` paths |
| `rainfall.weights` | NPZ path containing `zone_ids` and `weights` (zone × flattened crop-cell area fractions) |
| `rainfall.geometry` | JSON grid metadata path; `crop_window` is `[column, row, width, height]` in original-grid pixels |
| `rainfall.study_start`, `rainfall.study_end_exclusive` | ISO UTC timestamps defining the optional local-event segmentation grid, start included and end excluded |
| `baseline_jobs` | Optional list of `{ "file": "raw_data/baseline_request.npz", "output": "baseline_predictions.csv" }` jobs; an empty list skips fitting |

Each baseline NPZ contains same-length vectors `hour_utc` (UTC timestamp strings), `actual` (pickups/hour), `objective_normal` and `excluded` (Boolean masks), plus `requested_indices` (zero-based integer positions to predict). The fixed baseline requires at least 1,000 eligible training hours. `objective_normal` and `excluded` must encode the original weather/calendar eligibility and model-specific excluded windows. The TLC input needs `PULocationID` and pickup/dropoff timestamps (`tpep_` prefix for Yellow, `lpep_` for Green, no prefix for FHVHV); FHVHV also requires `hvfhs_license_num`. TLC timestamps are New York civil times; durations must be positive and no longer than six hours. FHVHV retains `HV0003`/`HV0005`. Ambiguous or nonexistent civil hours are excluded and counted.

Preprocessing writes monthly `{source}_{month}_hourly.csv.gz` files (`hour_utc`, `zone_id`, `source_service`, `pickups`), `zone_hourly_rainfall.csv.gz` (rain in mm and valid-area fraction), optional `local_rainfall_events.csv`, requested baseline prediction CSVs and a source-count summary. These exports are separate products and do not automatically replace the frozen analysis inputs. The fixed weather eligibility, event catalogue, normal-reference selection and model exclusions must remain aligned when preparing replacement inputs.

## 7. Analysis Workflow

TLC preprocessing filters completed trips, retains NYC pickup zones, aggregates hourly pickups and resolves local timestamps to UTC. MRMS one-hour accumulation ending at a timestamp is assigned to the preceding activity hour. Area-weighted rainfall requires virtually complete valid polygon coverage. Main-study local episodes use a 0.1 mm wet-hour threshold, a fixed 9.4 mm peak-hour heavy-rain threshold and three complete dry hours of separation; weak episodes are retained for weather-process grouping.

The main baseline uses the original 173-feature Ridge model and fixed normal/exclusion masks. The supplied analysis inputs retain its fitted predictions and normal-reference centers. The Pilot instead uses same-weekday/hour historical reference medians; its station-based event definition and sample remain separate.

The default analysis calculates raw gap `1 - Y/P` and centered gap `1 - Y/P - b`. Initial deficit `D0` is the mean of the two hourly bins immediately before local rain ending (offsets −2 and −1). It measures first threshold crossings and subsequent reversals, `H80` (hours with a gap above 20% of positive `D0`) and `Qlate` (initial deficit minus late-window mean gap). H80 is a duration, not the first crossing time. Late means use hours 2–3 for H3, hours 4–6 for H6 and hours 22–24 for H24.

Matching uses the original exposure, timing, initial-deficit and subsequent-rain criteria, with at most five other zones. Relative shortfalls average peers within an episode, episodes within a weather group, then groups. `S80` is target minus peer H80; `Sprogress` is peer minus target Qlate. A recurring pattern requires at least three valid groups, a positive fraction of at least two thirds and a positive mean. Earlier candidates are fixed before assessing later groups. The normal-day comparison and local reference correction retain their own fixed rosters and scopes.

## 8. Outputs

| Directory/file | Contents and units |
|---|---|
| `results/analysis_summary.json` | Computed sample counts, rebound/reversal fraction and recurring-zone names |
| `results/recovery_event_metrics.csv` | Raw/centered D0 and hourly gaps (ratios), H80 and crossing times (hours), Qlate (ratio), reversal flags and rainfall class |
| `results/hourly_curves.csv.gz` | Recalculated raw and centered trajectories |
| `results/temporal_split_groups.csv` | Chronological exploration/later assignment |
| `results/peer_selection.csv`, `event_peer_support.csv` | Selected peer pairs, comparison criteria and support counts |
| `results/event_peer_recovery.csv`, `zone_weather_group_recovery.csv` | Relative metrics by local episode and weather group; S80 in hours, Sprogress as a ratio |
| `results/recurrence_period_summary.csv`, `replication_status_long.csv` | Valid/positive group counts, means, positive fractions and recurrence classifications |
| `results/service_event_process.csv`, `service_weather_group_summary.csv`, `service_comparison_summary.csv` | Separate H80 and Qlate samples at six/24 hours; S80 in hours and Sprogress in percentage points |
| `results/normal_window_metrics.csv.gz`, `normal_excess_event_metrics.csv`, `normal_excess_group_scores.csv`, `five_zone_period_summary.csv` | Paired normal-day comparisons and five-zone period summaries; group S80 in hours and Sprogress in percentage points |
| `results/local_reference_event_metrics.csv`, `local_reference_group_scores.csv` | Local corrected-center comparison; group S80 in hours and Sprogress in percentage points |
| `results/pilot_snapshot.csv` | Pilot percentage changes, `100 × (actual/baseline - 1)`, and map eligibility |
| `figures/` | Figures 1–6 as 300-dpi PNG, PDF and SVG |
| `tables/` | Table 1 as CSV and Markdown, plus calculated supporting figure values |

In recurrence outputs, positive scores indicate greater relative shortfall. `valid_groups` counts groups contributing an estimable metric, while `selected_groups` includes selected groups even when that metric is unavailable. `later_mean_display` converts Sprogress to percentage points. Missing results remain empty; no-data categories are retained on maps.

`mode` is `raw` or `centered`; `H`/`horizon` gives the observation window in hours. Metric columns use this convention: `raw_T50_6_hour` is the first observed six-hour-window value at or below half of D0, and `raw_T50_6_relapse` identifies a subsequent recrossing above it. T80 uses 20% of D0; status fields distinguish complete observations from missing follow-up.

Service `sample` identifies `ORIGINAL20_H6` (fixed candidate cases at six hours), `FIXED20_H24_AVAILABLE` (available 24-hour measurements), or `H24_METRIC_COMMON_H6` / `H24_METRIC_COMMON_H24` (the same metric-specific 24-hour-supported cases evaluated at each horizon). H80 and Qlate have separate complete samples. In these tables, `metric = S80` is measured in hours and `metric = Sprogress_pp` in percentage points.

## 9. Reproducibility

The supplied frozen-input analysis has been executed from hourly inputs through all six figures and Table 1. Sample counts, reversal proportions, matched-service results, recurrence classifications and Mount Hope trajectories were compared with the final report and its source tables. Final result tables are generated by the calculations, not used as answer inputs.

The copied package also ran from a different working directory with identical numerical outputs. Optional preprocessing was checked on the real August 2023 Green taxi file and one real MRMS hour; a real fixed-training baseline was refitted and agreed with its saved predictions. These checks cover those inputs, not every raw month or baseline model.

A complete raw-from-scratch rerun was not performed. Twelve monthly High Volume FHV raw files are absent from the retained collection: January–June, August and November–December 2024; January, June and August 2025. The complete MRMS GRIB archive is also absent. Retained monthly activity aggregates and MRMS crops support the existing frozen inputs; their recorded missing hours remain missing. Full raw regeneration requires those source records, the original rainfall intersection weights, station eligibility information, complete weather catalogue and fixed baseline/reference masks. The default supplied-input run requires none of these external raw files.

The local reference correction is limited to its original CW0249 comparisons; it does not revise the global peer roster. The Pilot, main ride-hailing analysis and service comparison must retain their separate definitions when extending or replacing inputs.
