# Basic-statistics parameters and calculations

This guide covers the non-turn and non-run/tumble statistics exported from
`MiMoSA.Stats`:

- `05_Basic_Stats/01_Fitted_Mean_Speed_Distribution.csv`
- `06_Additional_Analysis/01_Particle_Characteristics.csv`
- `06_Additional_Analysis/12_Area_Speed_Particles.csv`
- `06_Additional_Analysis/13_Area_Speed_Correlation.csv`
- `06_Additional_Analysis/14_Individual_MSD.csv`
- `06_Additional_Analysis/15_Ensemble_MSD.csv`
- `06_Additional_Analysis/16_MSD_Fit.csv`

See [STATS_EXPORTED_DATA_FILES_SUMMARY.md](STATS_EXPORTED_DATA_FILES_SUMMARY.md)
for shared conventions and the complete export inventory.

## Core distance, duration, and speed calculation

For two retained detections at frames `f0` and `f1`, with Tracker centroids
`(x0, y0)` and `(x1, y1)`:

```text
frame_delta     = f1 - f0
pixel_distance  = sqrt((x1 - x0)^2 + (y1 - y0)^2)
elapsed_seconds = frame_delta / capture_speed_in_fps
scaled_distance = pixel_distance * pixel_scale_factor
speed           = distance / elapsed_time
```

The requested `speed_unit` determines whether distance remains in pixels or is
scaled and whether elapsed time remains in frames or is converted to seconds.

Worked example: a centroid moves by 3 pixels in x and 4 pixels in y over two
frames. With `pixel_scale_factor=0.2 um/pixel` and 10 FPS, the distance is
`sqrt(3² + 4²) * 0.2 = 1.0 um`, the duration is `2 / 10 = 0.2 s`, and the speed
is `1.0 / 0.2 = 5.0 um/s`.

## Fitted mean speed

### Calculation inputs

`calculate_speed_and_plot_mean()` uses these exact argument names:

| Argument | Default | Effect |
|---|---:|---|
| `distribution_type` | `"norm"` | Distribution passed to `distfit`. |
| `fit_range` | `None` | Inclusive explicit lower and upper speed bounds. |
| `ci_range` | `(5, 95)` | Percentiles used as bounds when `fit_range=None`. |
| `bin_size` | `30` | Histogram bin count; it does not change fitted values. |
| `speed_unit` | `"scale_units/s"` | Speed calculation and display unit. |
| `plot_results` | `True` | Whether to draw per-particle plots; it does not change fitted values. |
| `discard_initial_frames` | `10` | Initial source-frame window width removed per particle. |
| `discard_final_frames` | `10` | Final source-frame window width removed per particle. |

This workflow has no `max_frame_gap` argument. Every adjacent retained pair is
used, with its real `frame_delta` included in the speed denominator.

For each particle, the method calculates interval speeds, selects values inside
the requested or percentile-derived inclusive bounds, fits the requested
distribution to that subset, and exports the fitted model's expected value.
This is not necessarily the raw arithmetic mean of all interval speeds.

Example: speeds `[1, 2, 3, 4, 10] um/s` with `fit_range=(1, 4)` retain
`[1, 2, 3, 4]`. A normal fit has a model mean of `2.5 um/s`; the excluded
`10 um/s` does not affect the result. At least two retained values are required.

### `01_Fitted_Mean_Speed_Distribution.csv`

One row represents one particle's fitted speed mean. `save_mean_speeds()` adds
`.csv` when needed and rejects other file extensions.

| Column | Calculation or provenance |
|---|---|
| `strain` | Label supplied to `Stats`; copied to every row. |
| `source_dataframe` | Resolved Tracker snapshot used for the calculation. |
| `particle` | Linked trajectory ID. |
| `mean_speed` | Expected value returned by the fitted distribution model after range filtering. Blank if fewer than two filtered speeds remain or fitting fails. |
| `distribution_type` | Requested `distfit` distribution name. |
| `fit_range_source` | `requested` for explicit `fit_range`; otherwise `ci_percentiles`. |
| `requested_fit_range_start`, `requested_fit_range_end` | Explicit requested bounds; blank when percentile bounds were used. They are settings, not particle-specific effective percentiles. |
| `ci_range_lower_percentile`, `ci_range_upper_percentile` | Requested percentile pair, retained even when an explicit fit range takes precedence. |
| `capture_speed_in_fps` | Frame-rate snapshot used for per-second conversion. |
| `pixel_scale_factor` | Configured physical distance per pixel. |
| `scale_units` | Spatial label associated with the pixel scale, such as `um`. |
| `speed_unit` | Resolved unit of `mean_speed`. |
| `discard_initial_frames`, `discard_final_frames` | Requested per-particle boundary-window widths. |

`include_details=True` writes all columns above. The backward-compatible
`include_details=False` mode writes only `strain` and `mean_speed`.

## Particle characteristics

### Calculation inputs

`calculate_particle_characteristics()` accepts `particle_ids=None`,
`length_unit="scale_units"`, `speed_unit="scale_units/s"`,
`max_frame_gap=None`, and both discard arguments defaulting to 10. The
`max_frame_gap` setting changes accepted motion intervals but is not a column in
this dataframe, so retain the producing call or notebook as provenance.

### `01_Particle_Characteristics.csv`

One row summarizes one retained particle track.

| Column | Calculation or provenance |
|---|---|
| `strain`, `source_dataframe`, `particle` | Dataset label, resolved Tracker source, and trajectory ID. |
| `first_frame`, `last_frame` | First and last retained source frames after boundary trimming. |
| `detection_count` | Number of retained detection rows. |
| `frame_span` | `last_frame - first_frame`. |
| `frame_span_duration_seconds` | `frame_span / capture_speed_in_fps`; includes time occupied by internal gaps. |
| `observed_duration_seconds` | Sum of accepted `frame_delta / FPS` motion intervals; intervals rejected by `max_frame_gap` are excluded. |
| `mean_major_axis_length`, `median_major_axis_length`, `std_major_axis_length` | Mean, median, and sample SD of finite `major_axis_length * length_factor`. |
| `mean_minor_axis_length`, `median_minor_axis_length`, `std_minor_axis_length` | Corresponding statistics for `minor_axis_length * length_factor`. |
| `mean_aspect_ratio`, `median_aspect_ratio`, `std_aspect_ratio` | Statistics of `major_axis_length / minor_axis_length` for rows with two finite axes and a strictly positive minor axis. Scaling cancels, so the ratio is dimensionless. |
| `mean_area`, `median_area`, `std_area` | Statistics of finite `area * length_factor²`. |
| `speed_interval_count` | Number of accepted detection-to-detection speed intervals. |
| `fitted_mean_speed` | Particle-matched `mean_speed` from the latest compatible fitted-speed cache. Compatibility requires the same source, speed unit, and discard windows; otherwise blank. |
| `mean_speed` | Time-weighted whole-track speed: `sum(step_distance) / sum(elapsed_time)`. |
| `mean_interval_speed` | Arithmetic mean of accepted interval speeds; every interval receives equal weight. |
| `median_speed`, `std_speed` | Median and sample SD of accepted interval speeds. |
| `total_path_length` | Sum of accepted raw centroid step distances converted with `length_unit`. |
| `length_unit`, `area_unit`, `speed_unit` | Resolved units for axes/path, area, and speed. |
| `discard_initial_frames`, `discard_final_frames` | Requested boundary-window widths. |

Worked speed example: two accepted intervals have distances `[1, 2] um` and
durations `[0.1, 0.4] s`. Their speeds are `[10, 5] um/s`.
`mean_interval_speed=(10+5)/2=7.5 um/s`, but
`mean_speed=(1+2)/(0.1+0.4)=6.0 um/s`. The difference is an aggregation choice,
not a calculation error.

Worked morphology example: retained major-axis lengths `[2, 3, 4] um` have
mean `3 um`, median `3 um`, and sample SD `1 um` because `ddof=1`.

## Feature-speed correlation

### Calculation inputs

`calculate_speed_feature_correlation()` accepts:

| Argument | Default | Effect |
|---|---:|---|
| `feature_column` | `"area"` | Detection-level feature to average per particle. |
| `feature_unit` | `"auto"` | For area/axis columns, applies physical scaling; `raw` or `stored` retains stored values. Other features stay in stored units. |
| `speed_unit` | `"scale_units/s"` | Unit for interval and track speeds. |
| `particle_ids` | `None` | All particles, or a selected ID/list/tuple. |
| `max_frame_gap` | `None` | Largest frame difference accepted for motion intervals. |
| `discard_initial_frames`, `discard_final_frames` | `10`, `10` | Boundary-frame windows removed before both feature and speed means. |

At least two finite particle pairs are required, and both variables must contain
at least two distinct values.

### `12_Area_Speed_Particles.csv`

Despite the standard filename, `feature_column` can describe another feature.
Each row is one particle used as potential correlation input.

| Column | Calculation or provenance |
|---|---|
| `strain`, `source_dataframe`, `particle` | Dataset label, resolved source, and trajectory ID. |
| `detection_count` | Retained feature-row count, including rows whose feature later proves nonfinite. |
| `speed_interval_count` | Count of finite accepted interval speeds. |
| `mean_feature` | Arithmetic mean of finite retained feature values after the chosen unit conversion. Area uses `pixel_scale_factor²`; axis length uses `pixel_scale_factor`. |
| `mean_speed` | `sum(step_distance) / sum(elapsed_time)` over accepted intervals. |
| `mean_interval_speed` | Arithmetic mean of accepted interval speeds. |
| `feature_column`, `feature_unit`, `speed_unit` | Selected feature and resolved units. |
| `discard_initial_frames`, `discard_final_frames` | Boundary-window widths. |

`max_frame_gap` affects the speed values and counts but is not stored in this
CSV.

### `13_Area_Speed_Correlation.csv`

One row summarizes the finite `(mean_feature, mean_speed)` particle pairs.

| Column | Calculation or provenance |
|---|---|
| `strain`, `source_dataframe` | Dataset label and resolved source. |
| `feature_column`, `feature_unit`, `speed_unit` | Analysis variable and units inherited from the particle table. |
| `discard_initial_frames`, `discard_final_frames` | Boundary trimming used for the input particle means. |
| `particle_count` | Number of finite paired particle rows included. |
| `pearson_r` | Pearson product-moment correlation coefficient. |
| `p_value` | Two-sided p-value for the null hypothesis `pearson_r = 0`, from `scipy.stats.pearsonr`. |
| `slope`, `intercept` | Ordinary least-squares line `mean_speed = slope * mean_feature + intercept`. |
| `r_squared` | Square of the regression correlation coefficient. |
| `slope_standard_error` | Standard error of the fitted slope from `scipy.stats.linregress`. |

Worked example: particle pairs `(area, speed) = (1,6), (2,4), (3,2)` lie on
`speed = -2 * area + 8`. Thus `slope=-2`, `intercept=8`, `pearson_r=-1`, and
`r_squared=1`.

## Translational mean squared displacement

### Calculation inputs

`calculate_mean_squared_displacement()` accepts:

| Argument | Default | Effect |
|---|---:|---|
| `particle_ids` | `None` | All particles or a selected subset. |
| `max_lag_time_frames` | `100` | Largest requested source-frame lag passed to Trackpy. |
| `distance_unit` | `"scale_units"` | Pixel or configured physical position scale. |
| `time_unit` | `"seconds"` | `seconds` uses capture FPS; `frames` uses FPS 1. |
| `alpha_fit_lag_range` | `None` | Optional inclusive lag range used only for the log-log fit. |
| `discard_initial_frames`, `discard_final_frames` | `10`, `10` | Boundary windows removed before MSD calculation. |

For particle `j` and lag `tau`, Trackpy calculates

```text
MSD_j(tau) = mean(||r_j(t + tau) - r_j(t)||²)
```

using available coordinate pairs at that exact source-frame lag. Gapped tracks
are reindexed and pairs lacking either endpoint are omitted.

Worked example: one-dimensional positions `[0, 1, 3]` at consecutive frames
give lag-one squared displacements `[1², 2²]`, so
`individual_msd(1 frame)=(1+4)/2=2.5 distance_units²`. The lag-two result is
`(3-0)²=9 distance_units²`.

### `14_Individual_MSD.csv`

One row is one particle-lag result.

| Column | Calculation or provenance |
|---|---|
| `strain`, `source_dataframe`, `particle` | Dataset label, resolved source, and trajectory ID. |
| `lag_time` | Source-frame lag divided by FPS for seconds, or unchanged for frames. |
| `individual_msd` | Particle mean squared two-dimensional displacement at that lag after spatial scaling. |
| `capture_speed_in_fps`, `pixel_scale_factor`, `scale_units` | Acquisition calibration snapshot. |
| `time_unit`, `msd_unit` | Resolved lag and squared-distance units. |

### `15_Ensemble_MSD.csv`

One row is one lag. Trackpy's ensemble mean is measurement-weighted rather than
an equal-particle arithmetic mean:

```text
ensemble_msd(tau) = sum(N_j(tau) * MSD_j(tau)) / sum(N_j(tau))
```

where `N_j(tau)` is Trackpy's effective number of independent measurements for
particle `j` at that lag.

| Column | Calculation or provenance |
|---|---|
| `strain`, `source_dataframe` | Dataset label and resolved source. |
| `lag_time` | Lag in the requested time unit. |
| `ensemble_msd` | Trackpy's effective-measurement-weighted ensemble result. |
| `capture_speed_in_fps`, `pixel_scale_factor`, `scale_units` | Acquisition calibration snapshot. |
| `time_unit`, `msd_unit` | Resolved units. |

### `16_MSD_Fit.csv`

The positive finite ensemble points inside `alpha_fit_lag_range`, if supplied,
are fitted by ordinary least squares in natural-log space:

```text
ln(ensemble_msd) = log_intercept + alpha * ln(lag_time)
ensemble_msd     = exp(log_intercept) * lag_time^alpha
```

| Column | Calculation or provenance |
|---|---|
| `strain`, `source_dataframe` | Dataset label and resolved source. |
| `particle_count` | Unique particles with at least one individual-MSD row; currently identical to `contributing_particle_count`. |
| `selected_particle_count` | Unique trajectories selected before checking whether they contribute MSD support. |
| `contributing_particle_count` | Unique selected particles with at least one individual-MSD result. |
| `max_lag_time_frames` | Requested maximum source-frame lag. |
| `discard_initial_frames`, `discard_final_frames` | Boundary windows used before MSD calculation. |
| `requested_alpha_fit_lag_start`, `requested_alpha_fit_lag_end` | Requested fit bounds; blank when no range restriction was supplied. |
| `fit_lag_start`, `fit_lag_end` | Smallest and largest positive finite ensemble lags actually fitted. |
| `fit_point_count` | Number of ensemble lag points included in the regression. |
| `alpha` | Slope of `ln(MSD)` versus `ln(lag)`, the anomalous-scaling exponent. |
| `log_intercept` | Natural-log intercept; exponentiating it gives the power-law coefficient. |
| `r_squared` | Squared correlation coefficient of the log-log regression. |
| `p_value` | Two-sided p-value for a zero log-log slope. |
| `alpha_standard_error` | Standard error of `alpha`. |
| `fit_status` | `fitted` with at least two usable points; otherwise `insufficient_data`. |
| `fit_message` | Blank after success; otherwise explains that at least two usable points are required. |
| `capture_speed_in_fps`, `pixel_scale_factor`, `scale_units` | Acquisition calibration snapshot. |
| `time_unit`, `msd_unit` | Resolved units. |

Worked fit example: if the ensemble results follow
`MSD(tau)=2*tau²`, then `ln(MSD)=ln(2)+2*ln(tau)`. The fitted values are
`alpha=2` and `log_intercept=ln(2)` in an exact noise-free example.
