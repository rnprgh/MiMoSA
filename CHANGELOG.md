# Changelog

## 0.2.0 — 2026-09-08

### Changed

- Renamed velocity-analysis columns so point-weighted, per-particle
  time-weighted, pooled support-weighted, and equal-particle estimands are
  explicit in each field name.
- Expanded the velocity Table 1 export from 16 to 32 rows and added `n` and
  `n_definition` columns with metric-specific observation units.
- Replaced the sample-specific statistics export document with a concise
  19-file index and four focused calculation guides containing current
  formulas, defaults, schemas, provenance, and worked examples.
- Updated both example notebooks for the canonical schema and reproducible
  bootstrap settings. Saved outputs that depended on the previous velocity
  schema were cleared and must be regenerated from the complete source run.

### Added

- Whole-particle percentile bootstrap confidence intervals for pooled
  time-weighted run and tumble speeds.
- Equal-particle and classified-duration-weighted tumble-time fraction rows in
  velocity comparisons.
- Sample SD and SEM reporting for velocity metrics, including `std_t_T`,
  `std_p`, and `std_R`, with explicit low-sample explanations.
- `show_n` for overall fitted-speed plots and `show_label` for comparative
  fitted-speed plots.
- A histogram-only major-axis-length result for 1–19 valid observations, with
  mixture classification explicitly marked unsupported.

### Legacy velocity-schema compatibility

`ComparativeStats.load_dataframes()` normalizes documented legacy velocity
columns and Table 1 parameter labels in memory. Existing CSV files are not
rewritten. Files containing both a legacy field and its canonical replacement
are rejected as ambiguous.

Important migrations include:

| Legacy name | Canonical name |
|---|---|
| `mean_run_speed` | `point_mean_run_speed` |
| `mean_tumble_speed` | `point_mean_tumble_speed` |
| `v_R` / `v_T` | `time_weighted_vR` / `time_weighted_vT` |
| Population `v_R` / `v_T` | `pooled_time_weighted_vR` / `pooled_time_weighted_vT` |
| `mean_particle_v_R` / `mean_particle_v_T` | `particle_mean_time_weighted_vR` / `particle_mean_time_weighted_vT` |

No existing public calculation keyword was removed. In particular,
`tumble_threshold_angle` remains the angle-classifier argument;
`tumble_threshold_angle_degrees` is the exported provenance column.

See [`docs/STATS_EXPORTED_DATA_FILES_SUMMARY.md`](docs/STATS_EXPORTED_DATA_FILES_SUMMARY.md)
for the complete mapping and links to detailed guides.
