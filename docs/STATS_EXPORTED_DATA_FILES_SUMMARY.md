# Stats exported-data file guide

This guide describes the data tables in:

- `02_Outputs/05_Basic_Stats`
- `02_Outputs/06_Additional_Analysis`

It is based on the headers of the existing CSV files and the calculations in
`src/MiMoSA/stats.py`. The file names in `06_Additional_Analysis` are
assigned by the project notebook when it saves the DataFrames returned by
`Stats`; only `save_mean_speeds()` writes a CSV directly from `stats.py`.
The updated notebook defines 19 CSV exports, including the two new vertical
Table 1-style population tables. Existing output folders will contain the
older 17-file set until the updated export cell is rerun. No standalone
step-metrics or filtered-turn-events CSV is present there.

## Conventions used throughout

- Every dataframe returned for export starts with a `strain` column. Its value
  comes from the `strain` argument supplied once when `Stats` is initialized;
  the default is `define cell strain`.
- A **particle** is one linked trajectory ID. IDs need not be consecutive.
- `source_dataframe` is the resolved Tracker table used for the calculation. It
  is `updated` in all of these exports.
- If the requested distance is `scale_units`, a pixel distance is multiplied by
  the Capture pixel-scale factor. The resolved label in these files is `um`
  (micrometres); areas are therefore `um²` and speeds are `um/s`.
- This project's loaded calibration is 0.1625 `um` per pixel and
  28.27888911452496 frames per second. These values supply `pixel_scale` and
  `FPS` in the formulas below. Current fitted-speed and MSD exports record the
  underlying `capture_speed_in_fps`, `pixel_scale_factor`, and `scale_units`
  values so physical-unit conversions can be audited after export.
- For two retained detections, the spatial step is
  `sqrt((delta centroid_x)^2 + (delta centroid_y)^2)`. Elapsed time is
  `delta frame / capture FPS`, and step speed is distance divided by elapsed
  time.
- `discard_initial_frames=N` removes detections from `first_frame` through
  `first_frame + N - 1`; `discard_final_frames=N` removes detections from
  `last_frame - N + 1` through `last_frame`. These are source-frame windows,
  not guaranteed counts of removed rows.
- A `max_frame_gap` splits or rejects motion across a larger frame difference.
  Missing inter-segment time and distance are not treated as observed motion.
- Unless stated otherwise, `mean_*` is an arithmetic mean of finite values,
  `median_*` is the finite-value median, and `std_*` is the sample standard
  deviation (`ddof=1`). A blank CSV cell normally means the value was
  unavailable, not applicable, or lacked enough observations.
- The three trajectory smoothers are `moving_average`,
  `triangular_smoothing`, and `sg_filter_rdp`. Only parameters used by the
  selected smoother are populated; unused smoothing columns are blank.
  `effective_smoothing_window` is the window actually usable for that segment
  after adapting to its length.
- Angle- and velocity-tumble Cartesian coordinates use
  `x = centroid_y` and `y = -centroid_x`. The original Tracker centroid columns
  themselves are not changed.

## 05_Basic_Stats

### 01_Fitted_Mean_Speed_Distribution.csv

**Row meaning:** one row per particle (120 rows). Despite the filename, this is
the set of per-particle fitted means used to form a distribution; it is not a
histogram-bin table.

The notebook calculation trims each track, calculates interval speeds, keeps
the particle-specific 5th-through-95th percentile interval speeds, fits a normal
distribution with `distfit`, and exports the fitted model mean. For this run,

`speed_i = pixel_distance_i * pixel_scale / (frame_delta_i / FPS)`.

This fitted-speed workflow has no `max_frame_gap` argument: every adjacent
retained detection pair contributes a speed, while its actual `frame_delta`
still determines elapsed time.

| Column | Meaning and calculation |
|---|---|
| `strain` | Dataset strain label supplied when `Stats` was initialized. |
| `source_dataframe` | Resolved Tracker source snapshot; `updated` here. |
| `particle` | Linked trajectory ID. |
| `mean_speed` | Mean of the fitted normal model for that particle's retained, percentile-trimmed interval speeds. It is not the raw mean of all interval speeds. |
| `distribution_type` | Requested distribution passed to `distfit`; `norm` here. |
| `fit_range_source` | `requested` when an explicit fit range supplied the bounds; otherwise `ci_percentiles`. |
| `requested_fit_range_start`, `requested_fit_range_end` | Explicit requested speed bounds in `speed_unit`; both are blank when `fit_range=None`. These are requested settings, not particle-specific percentile-derived bounds. |
| `ci_range_lower_percentile`, `ci_range_upper_percentile` | Requested percentile bounds used when no explicit fit range is supplied; 5 and 95 here. |
| `capture_speed_in_fps` | Capture frame rate snapshotted by `Stats`; used for per-second speed conversion. |
| `pixel_scale_factor` | Configured `scale_units` per pixel; used for physical-distance speed conversion. |
| `scale_units` | Spatial unit associated with `pixel_scale_factor`; `um` here. |
| `speed_unit` | Resolved speed unit; `um/s` here. |
| `discard_initial_frames` | Width of the removed starting source-frame window; 10 here. |
| `discard_final_frames` | Width of the removed ending source-frame window; 10 here. |

`include_details=True` produces the strain field and all provenance, requested
fit-setting, calibration, unit, and trimming columns. With
`include_details=False`, `save_mean_speeds()` continues to write only `strain`
and `mean_speed` for backward compatibility.
The histogram `bin_size` and `plot_results` settings do not affect the fitted
values.

## 06_Additional_Analysis

### 01_Particle_Characteristics.csv

**Row meaning:** one direct morphology-and-motion summary per particle (120
rows). Morphology uses retained detections; motion uses accepted intervals
between retained detections. The notebook call that produced this table used
`max_frame_gap=30` and 10-frame boundary discards.

| Column(s) | Meaning and calculation |
|---|---|
| `source_dataframe`, `particle` | Source snapshot and linked trajectory ID. |
| `first_frame`, `last_frame` | First and last retained source frames after boundary-window trimming. |
| `detection_count` | Number of retained detections. |
| `frame_span` | `last_frame - first_frame`, in frames. |
| `frame_span_duration_seconds` | `frame_span / FPS`; spans the endpoints and therefore includes time occupied by any internal gaps. |
| `observed_duration_seconds` | Sum of `frame_delta / FPS` over accepted motion intervals. Intervals exceeding the calculation's `max_frame_gap` are excluded. |
| `mean_major_axis_length`, `median_major_axis_length`, `std_major_axis_length` | Mean, median, and sample SD of finite `major_axis_length * pixel_scale`. |
| `mean_minor_axis_length`, `median_minor_axis_length`, `std_minor_axis_length` | Mean, median, and sample SD of finite `minor_axis_length * pixel_scale`. |
| `mean_aspect_ratio`, `median_aspect_ratio`, `std_aspect_ratio` | Mean, median, and sample SD of `major_axis_length / minor_axis_length` for rows with two finite axes and a positive minor axis. Dimensionless. |
| `mean_area`, `median_area`, `std_area` | Mean, median, and sample SD of finite `area * pixel_scale²`. |
| `speed_interval_count` | Number of accepted detection-to-detection speed intervals. |
| `mean_speed` | Time-weighted track speed: `sum(step_distance) / sum(elapsed_time)`. |
| `mean_interval_speed` | Unweighted arithmetic mean of the accepted interval speeds. |
| `median_speed`, `std_speed` | Median and sample SD of accepted interval speeds. |
| `total_path_length` | Sum of accepted raw centroid step distances, converted by the pixel scale. |
| `length_unit`, `area_unit`, `speed_unit` | Unit labels for lengths, areas, and speeds (`um`, `um²`, and `um/s` here). |
| `discard_initial_frames`, `discard_final_frames` | Requested boundary-frame window widths; both are 10 here. |

The existing CSV does **not** contain `fitted_mean_speed`. The current
`stats.py` now adds that field when a compatible fitted-speed result is cached,
so regenerating this file with the current code will change the schema unless
that new column is intentionally removed before export. When present,
`fitted_mean_speed` is the particle-matched distribution-fit mean described in
`01_Fitted_Mean_Speed_Distribution.csv`; it is blank if the cached source, unit,
or discard windows do not match. The CSV does not record the `max_frame_gap`
argument, so that value must be recovered from the producing notebook rather
than the CSV itself.

### 02_Turn_Summary.csv

**Row meaning:** one geometric-turn summary per particle (120 rows). This is
turn-angle detection, not run/tumble classification. In this export the
threshold is 30 degrees, smoothing is a 3-point triangular filter,
`max_frame_gap=1`, and both discard windows are 10 frames.

| Column(s) | Meaning and calculation |
|---|---|
| `source_dataframe`, `particle` | Source snapshot and trajectory ID. |
| `first_frame`, `last_frame`, `detection_count` | Endpoint frames and detection count after boundary trimming. |
| `segment_count` | Number of continuous pieces after splitting wherever adjacent detections differ by more than `max_frame_gap`. |
| `processed_point_count` | Total points used in the smoothed turn paths. For `sg_filter_rdp`, this would count RDP-retained points; triangular and moving-average modes use their finite smoothed points. |
| `angle_count` | Number of finite interior-vertex angles measurable on the processed paths. |
| `number_of_turns` | Count of angle rows with `abs(turn_angle_degrees) >= minimum_turn_angle_degrees`. Consecutive qualifying points are counted separately, not merged. |
| `mean_absolute_turn_angle_degrees` | Arithmetic mean of absolute angles for detected turns only; blank when there are no turns. |
| `total_path_length` | Sum of raw centroid step lengths inside retained segments, converted to `distance_unit`; gap-crossing steps are excluded. |
| `observed_duration_seconds` | Sum over segments of `(segment_last_frame - segment_first_frame) / FPS`. |
| `turns_per_distance` | `number_of_turns / total_path_length`. Unit is turns per `distance_unit`. |
| `turns_per_second` | `number_of_turns / observed_duration_seconds`. |
| `distance_unit` | Resolved path-length unit; `um` here. |
| `minimum_turn_angle_degrees` | Absolute angle threshold; 30 here. |
| `smoothing_method` | Selected trajectory smoother; `triangular_smoothing` here. |
| `moving_average_window`, `triangular_smoothing_window` | Requested centered moving-average or triangular span. Only the selected method is populated; triangular span is 3 here. |
| `sg_filter_window_length`, `sg_filter_polyorder`, `rdp_epsilon_pixels` | Savitzky-Golay span/order and RDP perpendicular-distance tolerance. Populated only for `sg_filter_rdp`. |
| `max_frame_gap` | Largest adjacent frame difference kept in one segment; 1 here. |
| `discard_initial_frames`, `discard_final_frames` | Boundary-frame window widths; 10 and 10 here. |

### 03_Turn_Angles.csv

**Row meaning:** one finite angle at an interior vertex of a processed turn path
(18,374 rows). It is the long-form evidence underlying
`02_Turn_Summary.csv`.

| Column(s) | Meaning and calculation |
|---|---|
| `source_dataframe`, `particle` | Source snapshot and trajectory ID. |
| `segment_id` | One-based continuous-segment number within the particle. |
| `processed_vertex_index` | Zero-based array index of the interior vertex within that segment's processed path; the first measurable interior vertex has index 1 because index 0 is an endpoint. |
| `frame` | Original source frame aligned with the processed vertex. |
| `elapsed_time_seconds` | `(frame - particle_first_retained_frame) / FPS`. This is a frame-axis timestamp; unlike observed duration, it can span inter-segment gaps. |
| `centroid_x_pixels`, `centroid_y_pixels` | Original stored Tracker centroid at the aligned source frame, in pixels. |
| `turn_angle_degrees` | Signed processed-path angle `degrees(atan2(cross(v1,v2), dot(v1,v2)))`, using `[centroid_y, -centroid_x]`. Positive is counterclockwise and negative is clockwise. |
| `absolute_turn_angle_degrees` | `abs(turn_angle_degrees)`. |
| `is_turn` | True when the absolute angle is at least the configured minimum. |
| `cumulative_distance` | Cumulative distance along the processed path through this vertex, converted to `distance_unit`. Segment path totals are joined without adding a gap-crossing distance. |
| `distance_unit` | Unit of `cumulative_distance`; `um` here. |
| `smoothing_method` | Smoother used for this path. |
| `effective_smoothing_window` | Actual smoothing span used for this segment after shortening for segment length, if necessary. |

### 04_Angle_Tumble_Summary.csv

**Row meaning:** one per-particle summary from the Turner-style, angle-based
run/tumble classifier (120 rows). This export used SG smoothing with window 5
and polynomial 3, RDP epsilon 5 pixels, a 30-degree threshold,
`max_frame_gap=10`, and 10-frame boundary discards. The sparse RDP path is
interpolated back to source-frame coordinates before temporal derivatives.

| Column(s) | Meaning and calculation |
|---|---|
| `source_dataframe`, `particle` | Source snapshot and trajectory ID. |
| `original_first_frame`, `original_last_frame`, `original_detection_count` | Endpoints and row count before boundary trimming. |
| `discarded_detection_count` | `original_detection_count - analyzed_detection_count`. |
| `analyzed_first_frame`, `analyzed_last_frame`, `analyzed_detection_count` | Endpoints and row count after boundary trimming. |
| `segment_count` | Number of analyzed pieces after splitting at gaps larger than `max_frame_gap`. |
| `analyzable_angle_count` | Number of finite angle-point rows. |
| `classified_angle_count` | Angle points assigned to either a run or a tumble. |
| `unclassified_angle_count` | `analyzable_angle_count - classified_angle_count`. |
| `number_of_runs`, `number_of_tumbles` | Counts of contiguous classified run and tumble events. |
| `runs_per_tumble` | `number_of_runs / number_of_tumbles`; blank if there are no tumbles. |
| `analyzed_tracking_time_seconds` | Sum over retained segments of `(last_frame - first_frame) / FPS`. Short, unclassified segments still contribute; excluded gaps do not. |
| `total_run_interval_seconds`, `total_tumble_interval_seconds` | Sums of event `interval_seconds`, where each interval is `(event_end_frame - event_start_frame) / FPS`. |
| `tumble_time_fraction` | `total_tumble_interval_seconds / analyzed_tracking_time_seconds`. |
| `tumble_frequency_per_second` | `number_of_tumbles / analyzed_tracking_time_seconds`. |
| `mean_run_speed`, `std_run_mean_speed_between_events` | Arithmetic mean and sample SD across the run events' `mean_speed` values; events receive equal weight. |
| `mean_run_point_speed`, `std_run_point_speed`, `run_speed_point_count` | Arithmetic mean, sample SD (`ddof=1`), and count after pooling all finite `instantaneous_speed` values whose point-level state is `run`. |
| `mean_tumble_speed`, `std_tumble_speed` | Equivalent statistics across tumble-event means. |
| `mean_run_interval_seconds`, `std_run_interval_seconds` | Mean and sample SD across run-event intervals. |
| `mean_tumble_interval_seconds`, `std_tumble_interval_seconds` | Mean and sample SD across tumble-event intervals. |
| `mean_time_between_tumble_starts_seconds`, `std_time_between_tumble_starts_seconds` | Mean and sample SD of differences between consecutive tumble start frames divided by FPS, calculated only within the same segment. |
| `mean_run_angular_speed_degrees_per_point`, `std_run_angular_speed_degrees_per_point` | Mean and sample SD across run events' mean unsigned point angles. Despite “speed” in the name, the unit is degrees per trajectory point, not degrees per second. |
| `mean_tumble_angular_speed_degrees_per_point`, `std_tumble_angular_speed_degrees_per_point` | Equivalent event-level statistics for tumbles. |
| `mean_direction_change_within_runs_degrees`, `std_direction_change_within_runs_degrees` | Across-run mean and sample SD of the angle between the sum of the first three and the sum of the last three velocity vectors in each run. |
| `mean_run_to_run_direction_change_degrees`, `std_run_to_run_direction_change_degrees` | Across-tumble mean and sample SD of direction change between a qualifying preceding and following run. |
| `distance_unit`, `speed_unit` | `um` and `um/s` here. |
| `smoothing_method` | Selected smoother; `sg_filter_rdp` here. |
| `moving_average_window` | Requested complete centered moving-average span; blank because that smoother was not selected. |
| `triangular_smoothing_window` | Requested centered triangular span; blank because that smoother was not selected. |
| `sg_filter_window_length` | Requested odd Savitzky-Golay span; 5 points here. |
| `sg_filter_polyorder` | Polynomial order fitted inside the SG window; 3 here. |
| `rdp_epsilon_pixels` | RDP perpendicular-distance tolerance applied after SG smoothing; 5 pixels here. |
| `discard_initial_frames`, `discard_final_frames` | Boundary-frame window widths; both 10. |
| `tumble_threshold_angle_degrees` | High-angle cutoff; 30 degrees here. |
| `max_frame_gap` | Largest adjacent frame difference retained in one segment; 10 here. |
| `analysis_status` | `complete`, `no_detections_after_discard`, `insufficient_points_for_five_point_velocity`, or `no_runs_or_tumbles_classified`, depending on available evidence. All rows in this export are `complete`. |

The classifier calculates velocity with a five-point fourth-order central
derivative. A run needs at least three consecutive angles at or below the
threshold. A tumble needs at least two consecutive angles above it; an isolated
high angle is accepted only if its summed-neighbor-vector confirmation angle is
also above the threshold.

### 05_Angle_Tumble_Events.csv

**Row meaning:** one contiguous classified run or tumble event (379 rows). This
is the event-level evidence summarized by files 04 and 07.

| Column(s) | Meaning and calculation |
|---|---|
| `source_dataframe`, `particle`, `segment_id` | Source, trajectory, and one-based gap-bounded segment. |
| `event_id` | Sequential event identifier within the particle. |
| `event_type` | `run` or `tumble`. |
| `start_frame`, `end_frame` | First and last classified angle-point frames in the event. |
| `support_end_frame` | Following detection required to calculate the event's final point angle. |
| `point_count` | Number of classified angle points in the event. |
| `interval_seconds` | `(end_frame - start_frame) / FPS`; follows the paper's event convention. |
| `sampling_support_seconds` | `(support_end_frame - start_frame) / FPS`; includes the following support detection. |
| `path_length` | Sum of smoothed displacement among event points, converted to `distance_unit`; it does not add the final support edge. |
| `mean_speed`, `std_speed` | Mean and sample SD of finite point speeds within the event. |
| `mean_angular_speed_degrees_per_point`, `std_angular_speed_degrees_per_point` | Mean and sample SD of unsigned point angles within the event. |
| `direction_change_within_run_degrees` | For a run of at least three points, angle between summed first-three and last-three velocity vectors; otherwise blank. |
| `run_to_run_direction_change_degrees` | On a tumble directly bracketed by qualifying runs, angle between the preceding run's final-three and following run's first-three summed velocities. |
| `preceding_run_event_id`, `following_run_event_id` | IDs of those bracketing runs; blank when the required adjacency/support is absent. |
| `distance_unit`, `speed_unit` | Units of path length and point speed; `um` and `um/s` here. |
| `smoothing_method` | Trajectory smoother used before differentiation; `sg_filter_rdp` here. |
| `effective_smoothing_window` | SG window actually supported by this event's segment after adapting to segment length. |
| `tumble_threshold_angle_degrees` | Strict high-angle cutoff used by the state classifier; 30 degrees here. |

### 06_Angle_Tumble_Points.csv

**Row meaning:** one finite velocity-direction angle point (18,458 rows). This
is the most detailed audit table for the angle-based classifier.

For consecutive frames, the five-point velocity derivative is

`velocity[i] = (p[i-2] - 8 p[i-1] + 8 p[i+1] - p[i+2]) * FPS / 12`.

For uneven retained frames, `stats.py` solves equivalent finite-difference
weights from the real frame offsets.

| Column(s) | Meaning and calculation |
|---|---|
| `source_dataframe`, `particle`, `segment_id` | Source, trajectory, and gap-bounded segment. |
| `frame`, `next_frame`, `frame_delta_to_next` | Current angle-point frame, following velocity-support frame, and their source-frame difference. |
| `elapsed_time_seconds` | Cumulative observed time: segment offset plus `(frame - segment_first_frame) / FPS`. Excluded inter-segment gaps are omitted. |
| `raw_position_x_pixels`, `raw_position_y_pixels` | Cartesian analysis position `[centroid_y, -centroid_x]` before smoothing. |
| `smoothed_position_x_pixels`, `smoothed_position_y_pixels` | Position after the selected smoother; SG+RDP is reconstructed on source-frame coordinates in this run. |
| `velocity_x`, `velocity_y` | Five-point derivative of the smoothed positions, converted to `distance_unit/s`. |
| `instantaneous_speed` | `sqrt(velocity_x² + velocity_y²)`. |
| `turn_angle_degrees` | Unsigned angle between the velocity at this point and the following point. |
| `exceeds_tumble_threshold` | True when `turn_angle_degrees` is strictly greater than the threshold (with a small numerical tolerance). |
| `singleton_confirmation_angle_degrees` | For an isolated high-angle candidate, the angle between `v[i-1] + v[i]` and `v[i+1] + v[i+2]`; blank otherwise. |
| `state` | `run`, `tumble`, or `unclassified` under the sequence and singleton rules described above. |
| `event_id` | ID of the containing classified event; blank for unclassified points. |
| `distance_unit`, `speed_unit` | Units of derived path/speed values; `um` and `um/s` here. |
| `smoothing_method` | Trajectory smoother used before velocity calculation; `sg_filter_rdp` here. |
| `effective_smoothing_window` | SG window actually supported by this point's segment. |
| `tumble_threshold_angle_degrees` | Strict high-angle cutoff used for `exceeds_tumble_threshold`; 30 degrees here. |

### 07_Angle_Tumble_Population.csv

**Row meaning:** one equal-particle-weight population summary (1 row). It
contains 120 particles, of which 120 have analyzable angles and 49 have at
least one angle-based tumble.

| Column(s) | Meaning and calculation |
|---|---|
| `source_dataframe` | Resolved source snapshot. |
| `particle_count` | Number of particle-summary rows. |
| `particles_with_analyzable_angles` | Count with `analyzable_angle_count > 0`. |
| `particles_with_tumbles` | Count with `number_of_tumbles > 0`. |
| `total_number_of_runs`, `total_number_of_tumbles` | Sums of per-particle event counts. |
| `total_analyzed_tracking_time_seconds` | Sum of per-particle analyzed tracking times. |
| `mean_tumble_frequency_per_second`, `std_tumble_frequency_per_second` | Arithmetic mean and sample SD of the finite per-particle tumble frequencies. |
| `mean_number_of_runs_per_particle`, `std_number_of_runs_per_particle` | Mean and sample SD of per-particle run counts. |
| `mean_number_of_tumbles_per_particle`, `std_number_of_tumbles_per_particle` | Mean and sample SD of per-particle tumble counts. |
| `mean_run_speed`, `std_run_speed_between_particles` | Mean and sample SD of each particle's `mean_run_speed`; particles receive equal weight. |
| `mean_tumble_speed`, `std_tumble_speed_between_particles` | Equal-particle statistics of `mean_tumble_speed`. |
| `mean_run_interval_seconds`, `std_run_interval_seconds_between_particles` | Equal-particle statistics of mean run interval. |
| `mean_tumble_interval_seconds`, `std_tumble_interval_seconds_between_particles` | Equal-particle statistics of mean tumble interval. |
| `mean_time_between_tumble_starts_seconds`, `std_time_between_tumble_starts_seconds_between_particles` | Equal-particle statistics of each particle's mean start-to-start interval. |
| `mean_run_angular_speed_degrees_per_point`, `std_run_angular_speed_degrees_per_point_between_particles` | Equal-particle statistics of mean run point angle. |
| `mean_tumble_angular_speed_degrees_per_point`, `std_tumble_angular_speed_degrees_per_point_between_particles` | Equal-particle statistics of mean tumble point angle. |
| `mean_direction_change_within_runs_degrees`, `std_direction_change_within_runs_degrees_between_particles` | Equal-particle statistics of within-run direction change. |
| `mean_run_to_run_direction_change_degrees`, `std_run_to_run_direction_change_degrees_between_particles` | Equal-particle statistics of run-to-run change across tumbles. |
| `distance_unit`, `speed_unit` | Population path and speed units; `um` and `um/s` here. |
| `smoothing_method` | Smoother shared by the summarized particles; `sg_filter_rdp` here. |
| `moving_average_window`, `triangular_smoothing_window` | Requested alternative-smoother spans; blank because neither method was selected. |
| `sg_filter_window_length`, `sg_filter_polyorder` | Requested SG span and polynomial order; 5 and 3 here. |
| `rdp_epsilon_pixels` | RDP perpendicular-distance tolerance after SG smoothing; 5 pixels. |
| `discard_initial_frames`, `discard_final_frames` | Boundary-frame window widths; both 10. |
| `tumble_threshold_angle_degrees` | High-angle classifier cutoff; 30 degrees. |
| `max_frame_gap` | Largest adjacent frame difference retained within a segment; 10 here. |

Population `mean_*` fields are not pooled over all events or points. A particle
with many events and a particle with few events each contribute one finite
particle-level value.

### 07_Angle_Tumble_Table_1.csv

**Row meaning:** a vertical Turner Table 1-style population table. `strain` is
the dataset label, `Parameter` contains the requested display label, and
`Value` contains its result. Each motion metric is followed by separate `±std`
and `±sem` rows.

- Motion means and sample SDs give every finite per-particle mean equal weight,
  matching the Turner population convention. SEM is `SD / sqrt(n)`, where `n`
  is the finite particle count for that specific metric; SD and SEM are blank
  when fewer than two particles contribute.
- Run and tumble speeds use the equal-event per-particle values in
  `mean_run_speed` and `mean_tumble_speed`, rather than pooling trajectory
  points across cells.
- Non-tumbling particles with positive analyzed time contribute a zero tumble
  frequency but do not contribute to tumble-only means.
- `Number of events (runs, tumbles)` stores the two population totals as one
  display value.
- Body diameter and length are MiMoSA morphology proxies inherited from
  RABiTPy: each particle's finite,
  post-discard `minor_axis_length` or `major_axis_length` measurements are
  averaged first, and those particle means are then weighted equally. These
  region-property ellipse axes are not direct caliper measurements. Missing
  morphology columns produce blank body-size values and a warning without
  blocking tumble classification.

### 08_Velocity_Tumble_Summary.csv

**Row meaning:** one per-particle summary from the Najafi-style velocity/angular
run-tumble classifier (120 rows). This export used a 9-point triangular filter,
`max_frame_gap=1`, boundary discards of 10 frames, speed thresholds 0.7 and
0.2, angular coefficient 0.8, a persistence interval of 1/6 second, and
four-point run-direction fits.

A speed minimum qualifies when `speed_minimum_depth / minimum_speed >= 0.7`.
Its low-speed period satisfies
`speed - minimum_speed <= 0.2 * speed_minimum_depth`. An angular-velocity peak
qualifies when cumulative absolute heading change is greater than
`sqrt(0.8 * elapsed_seconds)`. A tumble requires overlapping qualifying speed
and angular periods, but its final duration is the angular period.

The same calculation now estimates the run-phase rotational diffusion
coefficient independently of `p`. Within each particle and gap-bounded segment,
finite headings are split into contiguous run-only blocks and unwrapped. For
every supported source-frame lag `tau`, squared heading differences are pooled
over all eligible time origins and blocks:

`RMSD(tau) = mean((alpha(t + tau) - alpha(t))²)`.

The fit interval extends from the first positive lag through
`sum(total_run_interval_seconds) / sum(number_of_runs)`, so its upper bound is
the pooled mean of all reconstructed run intervals, including boundary-censored
intervals. MiMoSA gives each pair-pooled lag mean equal weight in a
zero-intercept fit to `RMSD(tau) = 2 * Dr * tau`. This makes the otherwise
unspecified regression convention reproducible and prevents tumbles, undefined
headings, particles, or segment boundaries from being crossed.

| Column(s) | Meaning and calculation |
|---|---|
| `source_dataframe`, `particle` | Source snapshot and trajectory ID. |
| `original_first_frame`, `original_last_frame`, `original_detection_count` | Endpoints and detection count before boundary trimming. |
| `discarded_detection_count` | Original minus analyzed detection count. |
| `analyzed_first_frame`, `analyzed_last_frame`, `analyzed_detection_count` | Endpoints and detection count after boundary trimming. |
| `segment_count` | Number of pieces after gap splitting. |
| `analyzable_segment_count` | Segments having at least five finite angular-velocity samples. |
| `kinematic_point_count` | Number of finite point-level angular-velocity values. |
| `unclassified_detection_count` | Point rows whose `state` is `unclassified`, normally from unsupported boundaries or short segments. |
| `number_of_runs` | Count of reconstructed run intervals before, between, and after tumbles in analyzable segments; boundary intervals count here. |
| `number_of_tumbles` | Count of accepted, merged velocity/angular tumble events. |
| `runs_per_tumble` | `number_of_runs / number_of_tumbles`; blank with no tumbles. |
| `analyzed_tracking_time_seconds` | Sum of complete post-trim observed-segment spans. |
| `classified_tracking_time_seconds` | Sum, over segments with at least five finite angular-velocity samples, of `(last frame with a finite smoothed position - first frame with a finite smoothed position) / FPS`. Triangular and SG-interpolated smoothing generally support the full segment; a centered moving average can remove boundary support. |
| `total_run_interval_seconds` | Sum of all reconstructed run intervals, including boundary-censored intervals. |
| `total_tumble_interval_seconds` | Sum of detected angular-period event durations. |
| `total_unclassified_interval_seconds` | `analyzed_tracking_time_seconds - classified_tracking_time_seconds`. |
| `tumble_time_fraction` | `total_tumble_interval_seconds / classified_tracking_time_seconds`. |
| `tumble_frequency_per_second` | `number_of_tumbles / classified_tracking_time_seconds`. |
| `mean_run_speed`, `std_run_speed` | Arithmetic mean and sample SD across finite point speeds labelled `run`. These are point-weighted, not time-weighted. |
| `mean_tumble_speed`, `std_tumble_speed` | Equivalent point-level statistics for `tumble` points. |
| `mean_run_interval_seconds`, `std_run_interval_seconds` | Mean and sample SD across all reconstructed run intervals, including censored boundaries. |
| `mean_tumble_interval_seconds`, `std_tumble_interval_seconds` | Mean and sample SD across all detected tumble-event durations. |
| `mean_time_between_tumble_starts_seconds`, `std_time_between_tumble_starts_seconds` | Mean and sample SD of consecutive tumble start-to-start times within segments. |
| `mean_run_angular_velocity_magnitude_radians_per_second`, `std_run_angular_velocity_magnitude_radians_per_second` | Mean and sample SD of finite angular-velocity magnitudes at run points. |
| `mean_tumble_angular_velocity_magnitude_radians_per_second`, `std_tumble_angular_velocity_magnitude_radians_per_second` | Equivalent point-level statistics at tumble points. |
| `v_R`, `v_T` | Time-weighted run and tumble speeds: `sum(speed_i * duration_i) / sum(duration_i)` over steps whose current and preceding points have the same state in the same segment. |
| `v_R_sample_count`, `v_T_sample_count` | Numbers of steps contributing to `v_R` and `v_T`. |
| `v_R_support_seconds`, `v_T_support_seconds` | Total contributing step durations. |
| `t_R`, `t_T` | Mean durations of complete, uncensored run and tumble episodes, respectively. |
| `std_t_R`, `sem_t_R` | Sample SD across this particle's complete run episodes and `std_t_R / sqrt(t_R_interval_count)`. Both are blank with fewer than two complete runs. |
| `t_R_interval_count`, `t_T_interval_count` | Numbers of complete episodes contributing to `t_R` and `t_T`. |
| `t_R_censored_interval_count`, `t_T_censored_interval_count` | Boundary-touching episode counts excluded from the complete-episode means. |
| `p` | Mean `cos(delta heading)` after `persistence_interval_seconds` within the same contiguous block of finite run headings, where the preceding point is also a run. The future heading is circularly interpolated in time. A blank/low-speed heading breaks the block, and the calculation never crosses a tumble or gap boundary. |
| `p_direction_change_count` | Number of supported within-run direction pairs contributing to `p`. |
| `R` | Mean cosine of the signed direction change between least-squares fitted run directions before and after a tumble. |
| `R_run_transition_count` | Number of supported run-to-run transitions contributing to `R`. |
| `distance_unit`, `speed_unit`, `angular_velocity_unit` | `um`, `um/s`, and `rad/s` here. |
| `smoothing_method` | `triangular_smoothing` here. |
| `moving_average_window` | Requested complete centered moving-average span; blank because that smoother was not selected. |
| `triangular_smoothing_window` | Requested odd triangular span; 9 points here. |
| `sg_filter_window_length`, `sg_filter_polyorder` | Requested SG span/order; blank because SG smoothing was not selected. |
| `rdp_epsilon_pixels` | RDP tolerance used only with SG smoothing; blank here. |
| `discard_initial_frames`, `discard_final_frames` | Boundary-frame window widths; both 10. |
| `speed_drop_ratio_threshold` | Required speed-dip depth ratio; 0.7 here. |
| `speed_period_depth_fraction` | Fraction of dip depth defining the low-speed period; 0.2 here. |
| `angular_change_coefficient` | Coefficient inside the angular threshold square root; 0.8 here. |
| `minimum_heading_speed` | Optional speed floor below which heading is undefined; blank means only the numerical zero-speed tolerance was used. |
| `speed_extrema_prominence`, `angular_velocity_extrema_prominence` | Optional peak-prominence filters in `speed_unit` and `rad/s`; blank here, so no explicit prominence cutoff was requested. |
| `extrema_min_distance` | Minimum extrema separation in retained kinematic samples; 1 here. |
| `persistence_interval_seconds` | Heading separation used for `p`; 1/6 second here. |
| `run_direction_fit_points` | Smoothed points fitted on each side of a tumble to calculate `R`; 4 here. |
| `max_frame_gap` | Largest frame difference retained within a segment; 1 here. |
| `analysis_status` | `analyzed_with_tumbles`, `analyzed_no_tumbles`, `insufficient_kinematic_points_for_extrema`, or `no_detections_after_discard`. This export contains the first two statuses. |

The ordinary `mean_run_speed`/`mean_tumble_speed` fields and the Table-1-style
`v_R`/`v_T` fields are intentionally different: the former average point
samples equally, while the latter weight by step duration and require the
previous and current point to share a state.

### 09_Velocity_Tumble_Events.csv

**Row meaning:** one accepted velocity/angular tumble (71 rows). All
`event_type` values are `tumble`. Overlapping or directly adjacent accepted
angular periods are merged.

| Column(s) | Meaning and calculation |
|---|---|
| `source_dataframe`, `particle` | Source snapshot and trajectory ID. |
| `segment_id` | One-based gap-bounded segment number within the particle. |
| `event_id` | Sequential accepted-tumble identifier within the particle. |
| `event_type` | Constant `tumble` in this method's event table. |
| `start_frame`, `end_frame` | Start and end of the final angular-defined tumble period. |
| `support_start_frame` | Previous valid-heading frame supporting the first event angular change. |
| `point_count` | Inclusive number of point rows from start through end. |
| `interval_seconds` | `(end_frame - start_frame) / FPS`. |
| `sampling_support_seconds` | `(end_frame - support_start_frame) / FPS`. |
| `path_length` | Sum of smoothed within-event displacement, converted to `distance_unit`; the support edge is excluded. |
| `mean_speed`, `minimum_speed`, `maximum_speed` | Mean, minimum, and maximum of finite speeds within the merged angular event period. |
| `mean_angular_velocity_magnitude_radians_per_second`, `maximum_angular_velocity_magnitude_radians_per_second` | Mean and maximum finite angular-velocity magnitude within the event. |
| `speed_minimum_frame`, `speed_minimum_value` | Frame and value of the representative qualifying local speed minimum. |
| `speed_minimum_depth` | `max(left_speed_peak - minimum_speed, right_speed_peak - minimum_speed)`, clipped at zero. |
| `speed_depth_ratio` | `speed_minimum_depth / speed_minimum_value`; treated as infinity when the minimum is numerically zero and depth is positive. |
| `speed_left_maximum_frame`, `speed_right_maximum_frame` | Nearest bracketing local speed maxima. |
| `speed_period_start_frame`, `speed_period_end_frame` | Connected low-speed region around the minimum satisfying `speed - minimum <= speed_period_depth_fraction * depth`, restricted to the bracketing maxima. |
| `angular_velocity_maximum_frame`, `angular_velocity_maximum_value` | Frame and value of the representative qualifying angular-velocity peak. |
| `angular_velocity_maximum_height` | `max(peak - left_angular_minimum, peak - right_angular_minimum)`, clipped at zero. |
| `angular_left_minimum_frame`, `angular_right_minimum_frame` | Local angular-velocity minima bracketing the representative peak. |
| `angular_total_directional_change_radians` | Sum of absolute wrapped heading changes between the bracketing minima. |
| `angular_directional_change_threshold_radians` | `sqrt(angular_change_coefficient * elapsed_time_between_angular_minima)`. |
| `angular_period_start_frame`, `angular_period_end_frame` | Bounds of the accepted angular candidate; for a merged event these are the merged angular bounds. |
| `matching_overlap_start_frame`, `matching_overlap_end_frame` | Intersection of the representative qualifying speed and angular periods. |
| `merged_candidate_count` | Number of matched angular candidates combined into the final event. |
| `evidence_scope` | `representative_matched_candidate`: diagnostic speed/angular evidence comes from the strongest representative when several candidates merge. |
| `preceding_run_fit_start_frame`, `preceding_run_fit_end_frame` | Frame range of the configured points fitted immediately before the tumble. |
| `following_run_fit_start_frame`, `following_run_fit_end_frame` | Equivalent fitted range after the tumble. |
| `preceding_run_direction_radians`, `following_run_direction_radians` | `atan2(slope_y, slope_x)` from least-squares fits of smoothed position versus time on each side. |
| `run_to_run_turn_angle_radians` | Wrapped signed difference between the following and preceding fitted directions. |
| `run_to_run_turn_angle_degrees` | Degree conversion of that signed angle. |
| `run_to_run_directional_cosine` | `cos(run_to_run_turn_angle_radians)`; one transition-level contribution to `R`. |
| `distance_unit`, `speed_unit`, `angular_velocity_unit` | Units for path, speed, and angular velocity; `um`, `um/s`, and `rad/s` here. |
| `smoothing_method` | Trajectory smoother used before backward differentiation; `triangular_smoothing` here. |
| `effective_smoothing_window` | Triangular span actually supported by the event's segment, up to the requested 9 points. |
| `speed_drop_ratio_threshold` | Minimum accepted `speed_depth_ratio`; 0.7 here. |
| `speed_period_depth_fraction` | Dip-depth fraction defining the low-speed period; 0.2 here. |
| `angular_change_coefficient` | Coefficient inside the square-root directional-change threshold; 0.8 here. |
| `minimum_heading_speed` | Optional speed floor for valid heading; blank means only the numerical zero-speed tolerance was used. |

The run-fit and run-to-run fields are blank when the configured four-point fits
cannot be supported on both sides of an event.

### 10_Velocity_Tumble_Points.csv

**Row meaning:** one retained detection after trimming and gap segmentation
(19,074 rows), including unsupported boundary and short-segment rows.

| Column(s) | Meaning and calculation |
|---|---|
| `source_dataframe`, `particle`, `segment_id` | Source, trajectory, and gap-bounded segment. |
| `frame`, `previous_frame` | Current detection and preceding detection supporting the backward difference; previous is blank at a segment start. |
| `frame_delta_from_previous` | `frame - previous_frame`. |
| `elapsed_time_from_previous_seconds` | `frame_delta_from_previous / FPS`. |
| `elapsed_time_seconds` | Cumulative observed time: segment offset plus elapsed time inside the segment; excluded gaps are omitted. |
| `raw_position_x_pixels`, `raw_position_y_pixels` | Cartesian analysis position `[centroid_y, -centroid_x]`. |
| `smoothed_position_x_pixels`, `smoothed_position_y_pixels` | Position after smoothing. Short segments adapt the requested triangular span from 9 to 7, 5, or 3; blank effective windows mean smoothing was not possible. |
| `velocity_x`, `velocity_y` | `(current_smoothed_position - previous_smoothed_position) / elapsed_time_from_previous_seconds`, scaled to physical distance and assigned to the ending/current frame. |
| `speed` | Euclidean velocity magnitude. |
| `heading_radians` | `atan2(velocity_y, velocity_x)` when speed is above the numerical/configured floor; blank otherwise. |
| `previous_valid_heading_frame` | Most recent earlier frame with a valid heading. This explicitly records support when zero/low-speed headings are bridged. |
| `angular_elapsed_time_seconds` | Time since that previous valid heading. |
| `angular_change_radians` | Absolute wrapped heading change from the previous valid heading. |
| `angular_velocity_magnitude_radians_per_second` | `angular_change_radians / angular_elapsed_time_seconds`. |
| `persistence_target_elapsed_time_seconds` | Current elapsed time plus the configured persistence interval when a target exists inside the same contiguous block of finite run headings and the preceding point is also a run. A blank/low-speed heading breaks the block. |
| `persistence_target_frame` | Time-interpolated source-frame coordinate of that target. It may be fractional. |
| `persistence_direction_change_radians` | Signed wrapped change from current heading to the circularly interpolated target heading. |
| `persistence_cosine` | Cosine of the persistence direction change; one contribution to `p`. |
| `is_speed_minimum` | True at a detected local speed minimum, whether or not it qualifies. |
| `speed_minimum_passes` | True when the local minimum has two-sided support and meets the speed-depth ratio criterion. |
| `is_angular_velocity_maximum` | True at a detected local angular-velocity peak, whether or not it qualifies. |
| `angular_velocity_maximum_passes` | True when the angular candidate's cumulative change exceeds its duration-dependent threshold. |
| `state` | In an analyzable segment, every detection with a finite smoothed position is initialized as `run`, including boundary rows whose speed or angular velocity can be blank. Accepted angular-period rows are then overwritten as `tumble`; remaining rows are `unclassified`. |
| `event_id` | Accepted tumble ID for tumble points; blank for run and unclassified points. |
| `distance_unit`, `speed_unit`, `angular_velocity_unit` | Units for positions/path, speed, and angular velocity; `um`, `um/s`, and `rad/s` here. Raw and smoothed coordinate columns remain pixels as named. |
| `smoothing_method` | Trajectory smoother used before kinematic differences; `triangular_smoothing` here. |
| `effective_smoothing_window` | Triangular span actually supported by this segment: 9, 7, 5, or 3, or blank when smoothing was unavailable. |

### 11_Velocity_Tumble_Population.csv

**Row meaning:** one population summary combining 120 particle summaries. It
reports 71 tumbles across 652.819 seconds of classified tracking time, giving a
pooled frequency of about 0.10876 per second.

| Column(s) | Meaning and calculation |
|---|---|
| `source_dataframe` | Resolved source snapshot. |
| `particle_count` | Number of particle-summary rows. |
| `particles_with_kinematic_points` | Count with at least one finite angular-velocity point. |
| `particles_with_tumbles` | Count with at least one accepted tumble. |
| `total_number_of_runs`, `total_number_of_tumbles` | Sums of per-particle counts. |
| `total_analyzed_tracking_time_seconds`, `total_classified_tracking_time_seconds`, `total_unclassified_interval_seconds` | Sums of the corresponding per-particle times. |
| `pooled_tumble_frequency_per_second` | `total_number_of_tumbles / total_classified_tracking_time_seconds`. |
| `mean_particle_tumble_frequency_per_second`, `std_particle_tumble_frequency_per_second` | Equal-particle mean and sample SD of finite per-particle frequencies. |
| `mean_particle_tumble_time_fraction`, `std_particle_tumble_time_fraction` | Equal-particle statistics of per-particle tumble fractions. |
| `mean_number_of_runs_per_particle`, `std_number_of_runs_per_particle` | Equal-particle run-count statistics. |
| `mean_number_of_tumbles_per_particle`, `std_number_of_tumbles_per_particle` | Equal-particle tumble-count statistics. |
| `mean_run_speed`, `std_run_speed_between_particles` | Mean and sample SD of the per-particle `mean_run_speed` values. |
| `mean_tumble_speed`, `std_tumble_speed_between_particles` | Equivalent statistics for per-particle tumble speeds. |
| `mean_run_interval_seconds`, `std_run_interval_seconds_between_particles` | Equal-particle statistics of mean run intervals. |
| `mean_tumble_interval_seconds`, `std_tumble_interval_seconds_between_particles` | Equal-particle statistics of mean tumble intervals. |
| `mean_time_between_tumble_starts_seconds`, `std_time_between_tumble_starts_seconds_between_particles` | Equal-particle statistics of start-to-start intervals. |
| `mean_run_angular_velocity_magnitude_radians_per_second`, `std_run_angular_velocity_between_particles` | Equal-particle statistics of mean run angular velocity. |
| `mean_tumble_angular_velocity_magnitude_radians_per_second`, `std_tumble_angular_velocity_between_particles` | Equal-particle statistics of mean tumble angular velocity. |
| `v_R`, `v_T` | Pooled Table-1 speeds: weighted means of particle `v_R`/`v_T` using their support seconds. |
| `v_R_sample_count`, `v_T_sample_count` | Sums of contributing step counts. |
| `v_R_support_seconds`, `v_T_support_seconds` | Sums of contributing support durations. |
| `t_R`, `t_T` | Pooled complete-episode means, weighting particle values by `t_R_interval_count` or `t_T_interval_count`. |
| `std_t_R`, `sem_t_R` | Exact sample SD across all complete run-episode durations and `std_t_R / sqrt(t_R_interval_count)`. The pooled variance combines each particle's count, mean, and within-particle sample SD; it is not the SD across particle means. |
| `t_R_interval_count`, `t_T_interval_count` | Total complete run/tumble episodes. |
| `t_R_censored_interval_count`, `t_T_censored_interval_count` | Total boundary-censored run/tumble episodes excluded from `t_R`/`t_T`. |
| `p` | Pooled persistence weighted by `p_direction_change_count`. |
| `p_direction_change_count` | Total supported within-run heading pairs. |
| `R` | Pooled consecutive-run persistence weighted by `R_run_transition_count`. |
| `R_run_transition_count` | Total supported transitions. |
| `mean_particle_v_R`, `std_particle_v_R`, `mean_particle_v_T`, `std_particle_v_T` | Unweighted mean and sample SD across finite particle-level `v_R`/`v_T` values. |
| `mean_particle_t_R`, `std_particle_t_R`, `mean_particle_t_T`, `std_particle_t_T` | Unweighted particle-level complete-duration statistics. |
| `mean_particle_p`, `std_particle_p`, `mean_particle_R`, `std_particle_R` | Unweighted particle-level persistence statistics. |
| `distance_unit`, `speed_unit`, `angular_velocity_unit` | Population path, speed, and angular-velocity units; `um`, `um/s`, and `rad/s` here. |
| `smoothing_method` | Smoother shared by the summarized particles; `triangular_smoothing` here. |
| `moving_average_window` | Alternative moving-average span; blank because that smoother was not selected. |
| `triangular_smoothing_window` | Requested triangular span; 9 points here. |
| `sg_filter_window_length`, `sg_filter_polyorder`, `rdp_epsilon_pixels` | Alternative SG/RDP span, polynomial order, and tolerance; blank because SG/RDP was not selected. |
| `discard_initial_frames`, `discard_final_frames` | Boundary-frame window widths; both 10. |
| `speed_drop_ratio_threshold` | Minimum qualifying speed-dip depth ratio; 0.7. |
| `speed_period_depth_fraction` | Fraction of dip depth defining a candidate low-speed period; 0.2. |
| `angular_change_coefficient` | Coefficient in `sqrt(coefficient * elapsed_seconds)`; 0.8. |
| `minimum_heading_speed` | Optional valid-heading speed floor; blank here. |
| `speed_extrema_prominence`, `angular_velocity_extrema_prominence` | Optional prominence cutoffs for speed minima and angular peaks; both blank here. |
| `extrema_min_distance` | Minimum separation of candidate extrema in retained samples; 1. |
| `persistence_interval_seconds` | Within-run heading interval used for `p`; 1/6 second. |
| `run_direction_fit_points` | Points fitted on each side of a tumble for `R`; 4. |
| `max_frame_gap` | Largest adjacent frame difference retained within a segment; 1. |

The pooled Table-1 fields and `mean_particle_*` fields answer different
questions: pooled values weight by actual measurement support, whereas
`mean_particle_*` gives each particle one equal-weight value.

### 11_Velocity_Tumble_Table_1.csv

**Row meaning:** a vertical Najafi Table 1-style parameter table. Like every
other exported dataframe it has a leading `strain` column, and it retains the
paper-style `Strain` parameter row. Both values come from the `strain` argument
used to initialize `Stats`; the default is `define cell strain`. The remaining
rows contain pooled `v_R`, `v_T`, `t_R`, `t_T`, `p`, and `R` values from file
11, plus the independently fitted angular-MSD result `Dr (rad²/s)`. The `±std`
and `±sem` rows immediately after `Mean tR (s)` report uncertainty across
complete, uncensored run
episodes. Najafi et al. did not include uncertainty columns in their published
Table 1; these two rows are the MiMoSA extension to the Najafi-style table.

The `Dr fit max lag (s)` row records the all-run mean used as the requested fit
ceiling; it is intentionally distinct from `Mean tR (s)`, which uses only
complete, uncensored episodes. `Dr largest fitted lag (s)`, `Dr fit lag count`,
and `Dr fit direction-pair count` report actual fit support. `Dr fit uncentered
R^2` is the coefficient of determination appropriate to this
origin-constrained fit, and `Dr fit status` is `fitted` or an explicit reason
the coefficient is blank. Fit-range lag-dependent RMSD values are calculated
internally but are not retained; only the scalar needed for the Najafi Figure
1D comparison and its audit fields are placed in this vertical table.

### 12_Area_Speed_Particles.csv

**Row meaning:** one per-particle area-versus-speed pair (120 rows), used as the
input to the correlation in file 13. The notebook used `feature_column="area"`,
`feature_unit="scale_units"`, `speed_unit="scale_units/s"`,
`max_frame_gap=1`, and 10-frame boundary discards.

| Column(s) | Meaning and calculation |
|---|---|
| `source_dataframe`, `particle` | Source snapshot and trajectory ID. |
| `detection_count` | Number of retained detections. Nonfinite feature values are still counted here but are separately ignored by `mean_feature`. |
| `speed_interval_count` | Number of finite accepted interval speeds after the gap rule. |
| `mean_feature` | Arithmetic mean of finite stored area values multiplied by `pixel_scale²`; `um²` here. |
| `mean_speed` | `sum(step_distance) / sum(elapsed_time)` over accepted intervals. |
| `mean_interval_speed` | Unweighted arithmetic mean of accepted interval speeds. |
| `feature_column` | Tracked feature analyzed; `area` here. |
| `feature_unit` | Unit of `mean_feature`; `um²` here. |
| `speed_unit` | Unit of the two speed fields; `um/s` here. |
| `discard_initial_frames`, `discard_final_frames` | Boundary-frame window widths; both 10. |

`max_frame_gap=1` affects the interval counts and speeds but is not written as a
column in this CSV.

### 13_Area_Speed_Correlation.csv

**Row meaning:** one Pearson-correlation and ordinary least-squares regression
summary across the finite rows of file 12 (1 row).

| Column(s) | Meaning and calculation |
|---|---|
| `source_dataframe`, `feature_column`, `feature_unit`, `speed_unit` | Source, analyzed feature, and x/y units. |
| `discard_initial_frames`, `discard_final_frames` | Boundary-trimming settings used before per-particle means. |
| `particle_count` | Number of particles with finite `(mean_feature, mean_speed)` pairs; 120 here. |
| `pearson_r` | Pearson product-moment correlation between per-particle mean area and mean speed. It is about -0.1800 here. |
| `p_value` | Default two-sided p-value for the null hypothesis of zero Pearson correlation; about 0.0491 here. |
| `slope` | Least-squares slope in `mean_speed = slope * mean_feature + intercept`; units are `(um/s)/um²` here. |
| `intercept` | Predicted mean speed at feature value zero, in `um/s`. |
| `r_squared` | Square of the regression correlation coefficient (`r²`). |
| `slope_standard_error` | Standard error of the fitted slope from `scipy.stats.linregress`. |

### 14_Individual_MSD.csv

**Row meaning:** one particle at one lag time (10,450 rows), returned by
`trackpy.imsd` in long form.

For particle `j` and lag `tau`,

`MSD_j(tau) = mean(||r_j(t + tau) - r_j(t)||²)`

over all valid coordinate pairs separated by exactly that source-frame lag.
Positions use `centroid_y` and `centroid_x`; the squared Euclidean result is
unchanged by axis order. Trackpy reindexes gapped tracks and omits displacement
pairs that lack either endpoint.

| Column | Meaning and calculation |
|---|---|
| `source_dataframe` | Resolved source snapshot. |
| `particle` | Trajectory ID. |
| `lag_time` | Source-frame lag divided by FPS; seconds here. |
| `individual_msd` | Mean squared 2D displacement for this particle and lag after pixel-to-physical scaling; `um²` here. |
| `capture_speed_in_fps` | Capture frame rate snapshotted by `Stats`; used to convert source-frame lags to seconds. |
| `pixel_scale_factor` | Configured `scale_units` per pixel; its square converts pixel-squared displacement to physical MSD. |
| `scale_units` | Spatial unit associated with `pixel_scale_factor`; `um` here. |
| `time_unit` | Unit of `lag_time`; `s` here. |
| `msd_unit` | Unit of MSD; `um²` here. |

### 15_Ensemble_MSD.csv

**Row meaning:** one ensemble MSD at each lag (100 rows), returned by
`trackpy.emsd`.

| Column | Meaning and calculation |
|---|---|
| `source_dataframe` | Resolved source snapshot. |
| `lag_time` | Source-frame lag divided by FPS. |
| `ensemble_msd` | Trackpy's weighted ensemble mean at that lag: `sum(N_j * MSD_j) / sum(N_j)`, where `N_j` is Trackpy's effective number of independent measurements for particle `j` at that lag. This is not an equal-particle arithmetic mean. |
| `capture_speed_in_fps`, `pixel_scale_factor`, `scale_units` | Same snapshotted calibration recorded in the individual-MSD table. |
| `time_unit` | `s` here. |
| `msd_unit` | `um²` here. |

### 16_MSD_Fit.csv

**Row meaning:** one log-log linear-fit diagnostic row (1 row). MSD was
calculated to a maximum lag of 100 source frames after 10-frame boundary
discards. The requested fit range was 0.1-1.0 seconds; the available points
actually used span approximately 0.1061-0.9901 seconds.

The regression is

`ln(ensemble_MSD) = log_intercept + alpha * ln(lag_time)`,

or equivalently

`ensemble_MSD = exp(log_intercept) * lag_time^alpha`.

| Column | Meaning and calculation |
|---|---|
| `source_dataframe` | Resolved source snapshot. |
| `particle_count` | Unique particles appearing in the individual-MSD result; currently the same calculation as contributing count. |
| `selected_particle_count` | Unique trajectories selected before determining whether they contribute an MSD value. |
| `contributing_particle_count` | Unique particles with at least one individual-MSD output row. |
| `max_lag_time_frames` | Requested maximum lag in source frames; 100 here. |
| `discard_initial_frames`, `discard_final_frames` | Boundary-frame window widths; both 10. |
| `requested_alpha_fit_lag_start`, `requested_alpha_fit_lag_end` | Requested alpha-fit limits in `time_unit`; 0.1 and 1.0 seconds here. Both are blank when no fit-range restriction was requested. |
| `fit_lag_start`, `fit_lag_end` | Minimum and maximum positive, finite ensemble lag actually fitted after applying the requested lag range. |
| `fit_point_count` | Number of positive, finite ensemble-MSD points used in the regression; 26 here. |
| `alpha` | Slope of the natural-log MSD versus natural-log lag regression; about 1.9727 here. It is the fitted anomalous-scaling exponent. |
| `log_intercept` | Natural-log intercept of the regression. Exponentiating it gives the multiplicative coefficient in the power law. |
| `r_squared` | Squared correlation coefficient of the log-log regression. |
| `p_value` | Two-sided p-value for the null hypothesis that the log-log slope is zero. |
| `alpha_standard_error` | Standard error of the fitted slope. |
| `fit_status` | `fitted` when at least two usable points exist; otherwise `insufficient_data`. |
| `fit_message` | Blank after a successful fit; otherwise explains why the fit was unavailable. |
| `capture_speed_in_fps`, `pixel_scale_factor`, `scale_units` | Same snapshotted acquisition calibration recorded in the two MSD data tables. |
| `time_unit`, `msd_unit` | `s` and `um²` here. |

## Export inventory and version note

| Folder/file group | Producing `Stats` method | Rows in the existing export |
|---|---|---:|
| `05_Basic_Stats/01_Fitted_Mean_Speed_Distribution.csv` | `calculate_speed_and_plot_mean()` then `save_mean_speeds()` | 120 |
| `06_Additional_Analysis/01_Particle_Characteristics.csv` | `calculate_particle_characteristics()` | 120 |
| `02_Turn_Summary.csv`, `03_Turn_Angles.csv` | `calculate_turn_statistics()` | 120; 18,374 |
| `04`-`07_Angle_Tumble_*.csv` plus `07_Angle_Tumble_Table_1.csv` | `calculate_angle_tumble_statistics()` | 120; 379; 18,458; 1; 33 after rerun |
| `08`-`11_Velocity_Tumble_*.csv` plus `11_Velocity_Tumble_Table_1.csv` | `calculate_velocity_tumble_statistics()` | 120; 71; 19,074; 1; 9 existing, 16 with current code |
| `12_Area_Speed_Particles.csv`, `13_Area_Speed_Correlation.csv` | `calculate_speed_feature_correlation()` | 120; 1 |
| `14_Individual_MSD.csv`, `15_Ensemble_MSD.csv`, `16_MSD_Fit.csv` | `calculate_mean_squared_displacement()` | 10,450; 100; 1 |

The existing CSVs were inspected as the schema authority for this guide. The
installed `stats.py` and notebook were modified after those CSVs were written.
Current schema changes include `fitted_mean_speed` in particle characteristics,
the fitted-speed requested-setting/calibration fields, the MSD requested-range
and calibration fields, the angle run-point speed fields, complete-run `t_R`
uncertainty fields, and the two vertical Table 1-style dataframes. Every current
exported dataframe also includes the leading `strain` field. The current
velocity Table 1 additionally includes `Dr` and six angular-MSD fit diagnostics.
Rerunning the
notebook calculations and export cell is required to regenerate the CSVs with
these schemas.
