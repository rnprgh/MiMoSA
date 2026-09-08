# Stats exported-data guide and calculation index

This file is the concise index for dataframes produced by `MiMoSA.Stats` and
saved by the analysis notebook. Detailed formulas, column-by-column definitions,
and worked examples are split by analysis so that each guide remains readable:

| Analysis | Detailed guide | Exported files |
|---|---|---|
| Basic statistics | [STATS_BASIC_STATS_PARAMETERS.md](STATS_BASIC_STATS_PARAMETERS.md) | Fitted speeds, particle characteristics, feature-speed correlation, and translational MSD |
| Geometric turns | [STATS_TURN_ANALYSIS_PARAMETERS.md](STATS_TURN_ANALYSIS_PARAMETERS.md) | Turn summary and long-form turn angles |
| Angle-based run/tumble | [STATS_ANGLE_BASED_ANALYSIS_PARAMETERS.md](STATS_ANGLE_BASED_ANALYSIS_PARAMETERS.md) | Turner-style particle, event, point, population, and Table 1 tables |
| Velocity-based run/tumble | [STATS_VELOCITY_BASED_ANALYSIS_PARAMETERS.md](STATS_VELOCITY_BASED_ANALYSIS_PARAMETERS.md) | Najafi-style particle, event, point, population, and Table 1 tables |

The names in these guides were checked against the current column constants and
calculation code in `stats.py`. They describe newly calculated dataframes. Older
CSV files can retain legacy velocity names until the producing analysis is rerun.

## Common conventions

- Every export-ready dataframe starts with `strain`. It is the label passed to
  `Stats(..., strain=...)`; it is not inferred from the trajectory data.
- `source_dataframe` is the resolved Tracker snapshot used by the calculation.
  It can be `full`, `filtered`, or `updated`; `source_dataframe="active"` is
  resolved when `Stats` is initialized.
- A particle is one linked trajectory ID. Particle IDs need not be consecutive.
- With configured physical units, a pixel distance is multiplied by
  `pixel_scale_factor`; an area is multiplied by its square. A frame difference
  is divided by `capture_speed_in_fps` when a duration in seconds is required.
- `discard_initial_frames=N` removes detections in the inclusive source-frame
  window from `first_frame` through `first_frame + N - 1` for each particle.
  `discard_final_frames=N` applies the corresponding window at the other end.
  These values are frame-window widths, not guaranteed numbers of removed rows.
- `max_frame_gap` is the largest adjacent source-frame difference retained
  inside one segment or motion calculation. Motion across an excluded gap is
  not counted as observed distance or time.
- Unless a guide states otherwise, means use finite values, medians use finite
  values, sample standard deviations use `ddof=1`, and SEM is
  `sample SD / sqrt(n)`. A blank CSV cell represents `NaN`: the result was
  unavailable, inapplicable, or had insufficient support.
- Angle- and velocity-based Cartesian calculations use
  `x = centroid_y` and `y = -centroid_x`. The stored Tracker centroid columns
  are not modified.
- Supported smoothers are `moving_average`, `triangular_smoothing`, and
  `sg_filter_rdp`. Only the selected smoother's requested settings are
  populated. `effective_smoothing_window` records the span actually supported
  by a segment after any length adaptation.

## Current export inventory

The notebook supplies the numbered filenames in `06_Additional_Analysis` when
it saves dataframes returned by `Stats`. Only `save_mean_speeds()` writes its
CSV directly. The current notebook export surface contains 19 CSV files:

| Folder/file | Producing method | Row unit |
|---|---|---|
| `05_Basic_Stats/01_Fitted_Mean_Speed_Distribution.csv` | `calculate_speed_and_plot_mean()` then `save_mean_speeds()` | Particle |
| `06_Additional_Analysis/01_Particle_Characteristics.csv` | `calculate_particle_characteristics()` | Particle |
| `02_Turn_Summary.csv` | `calculate_turn_statistics()` | Particle |
| `03_Turn_Angles.csv` | `calculate_turn_statistics()` | Measurable processed-path angle |
| `04_Angle_Tumble_Summary.csv` | `calculate_angle_tumble_statistics()` | Particle |
| `05_Angle_Tumble_Events.csv` | `calculate_angle_tumble_statistics()` | Contiguous classified run or tumble |
| `06_Angle_Tumble_Points.csv` | `calculate_angle_tumble_statistics()` | Finite velocity-direction angle point |
| `07_Angle_Tumble_Population.csv` | `calculate_angle_tumble_statistics()` | Population |
| `07_Angle_Tumble_Table_1.csv` | `calculate_angle_tumble_statistics()` | Turner-style parameter; 33 rows |
| `08_Velocity_Tumble_Summary.csv` | `calculate_velocity_tumble_statistics()` | Particle |
| `09_Velocity_Tumble_Events.csv` | `calculate_velocity_tumble_statistics()` | Accepted tumble |
| `10_Velocity_Tumble_Points.csv` | `calculate_velocity_tumble_statistics()` | Retained detection/kinematic point |
| `11_Velocity_Tumble_Population.csv` | `calculate_velocity_tumble_statistics()` | Population |
| `11_Velocity_Tumble_Table_1.csv` | `calculate_velocity_tumble_statistics()` | Najafi-style parameter; 32 rows |
| `12_Area_Speed_Particles.csv` | `calculate_speed_feature_correlation()` | Particle |
| `13_Area_Speed_Correlation.csv` | `calculate_speed_feature_correlation()` | Correlation/regression result |
| `14_Individual_MSD.csv` | `calculate_mean_squared_displacement()` | Particle-lag pair |
| `15_Ensemble_MSD.csv` | `calculate_mean_squared_displacement()` | Lag |
| `16_MSD_Fit.csv` | `calculate_mean_squared_displacement()` | Log-log fit result |

There is no notebook-named standalone step-metrics CSV or filtered geometric
turn-events CSV. `calculate_step_metrics()` returns a useful dataframe, and
`get_turn_events_dataframe()` filters the most recent turn-angle dataframe, but
the standard export cell does not save them separately.

## Canonical velocity parameter names

Velocity-based speed names were made explicit about their aggregation level.
Use the following names in new calculations, documentation, and downstream
analysis:

| Scope | Canonical name | Meaning |
|---|---|---|
| Particle | `point_mean_run_speed`, `point_mean_tumble_speed` | Equal weight per finite labelled point |
| Particle | `time_weighted_vR`, `time_weighted_vT` | Elapsed-time-weighted same-state intervals within that particle |
| Population | `particle_mean_point_run_speed`, `particle_mean_point_tumble_speed` | Equal weight per finite particle-level point mean |
| Population | `pooled_time_weighted_vR`, `pooled_time_weighted_vT` | State-support-weighted pool across particles |
| Population | `particle_mean_time_weighted_vR`, `particle_mean_time_weighted_vT` | Equal weight per finite particle-level time-weighted value |

The current `ComparativeStats.load_dataframes()` normalizes older saved
velocity CSVs in memory. Important legacy-to-canonical mappings are:

| Legacy velocity name | Canonical name |
|---|---|
| particle `mean_run_speed`, `std_run_speed` | `point_mean_run_speed`, `point_std_run_speed` |
| particle `mean_tumble_speed`, `std_tumble_speed` | `point_mean_tumble_speed`, `point_std_tumble_speed` |
| particle `v_R`, `v_T` | `time_weighted_vR`, `time_weighted_vT` |
| particle `v_R_sample_count`, `v_T_sample_count` | `vR_interval_count`, `vT_interval_count` |
| population `mean_run_speed`, `mean_tumble_speed` | `particle_mean_point_run_speed`, `particle_mean_point_tumble_speed` |
| population/Table 1 `v_R`, `vR` | `pooled_time_weighted_vR` |
| population/Table 1 `v_T`, `vT` | `pooled_time_weighted_vT` |
| population `mean_particle_v_R`, `mean_particle_v_T` | `particle_mean_time_weighted_vR`, `particle_mean_time_weighted_vT` |

These mappings apply only to velocity-based outputs. The angle-based classifier
still intentionally uses `mean_run_speed` and `mean_tumble_speed` for its
equal-event per-particle metrics.

## Schema and regeneration notes

- `01_Particle_Characteristics.csv` now includes `fitted_mean_speed` when a
  compatible fitted-speed calculation is cached; otherwise it is blank.
- Detailed fitted-speed exports include requested fit settings, acquisition
  calibration, units, and frame-discard provenance. With
  `include_details=False`, `save_mean_speeds()` writes only `strain` and
  `mean_speed` for backward compatibility.
- Angle-based summaries include `mean_run_point_speed`, `std_run_point_speed`,
  and `run_speed_point_count` in addition to equal-event speed statistics.
- Velocity summaries include complete/censored episode counts and `std_t_T`,
  `std_p`, and `std_R`; velocity Table 1 rows include `n` and `n_definition`.
- Velocity Table 1 includes `Dr (rad²/s)` and six angular-MSD fit diagnostics.
- Existing CSVs are not rewritten automatically. Rerun the corresponding
  calculation and notebook export cell to materialize the current schema.
