# Angle-based run/tumble parameters and calculations

This guide covers `calculate_angle_tumble_statistics()`, MiMoSA's
two-dimensional Turner-style classifier, and these normal notebook exports:

- `06_Additional_Analysis/04_Angle_Tumble_Summary.csv`
- `06_Additional_Analysis/05_Angle_Tumble_Events.csv`
- `06_Additional_Analysis/06_Angle_Tumble_Points.csv`
- `06_Additional_Analysis/07_Angle_Tumble_Population.csv`
- `06_Additional_Analysis/07_Angle_Tumble_Table_1.csv`

The method argument is named `tumble_threshold_angle`; its export provenance
column is named `tumble_threshold_angle_degrees`. This distinction is
intentional. See
[STATS_EXPORTED_DATA_FILES_SUMMARY.md](STATS_EXPORTED_DATA_FILES_SUMMARY.md)
for conventions shared by all Stats exports.

## Calculation inputs

| Argument | Default | Effect |
|---|---:|---|
| `particle_ids` | `None` | All particles, one ID, or a list/tuple of IDs. |
| `discard_initial_frames` | `10` | Initial per-particle source-frame window removed. |
| `discard_final_frames` | `10` | Final per-particle source-frame window removed. |
| `tumble_threshold_angle` | `35.0` | Direction-change threshold in degrees. Valid range is 0 through 180. |
| `smoothing_method` | `"moving_average"` | `moving_average`, `triangular_smoothing`, or `sg_filter_rdp`. |
| `moving_average_window` | `5` | Complete centered odd moving-average span. |
| `triangular_smoothing_window` | `17` | Odd triangular-filter span with edge-value padding. |
| `sg_filter_window_length` | `5` | Odd Savitzky-Golay span. |
| `sg_filter_polyorder` | `2` | SG polynomial order, smaller than the SG span. |
| `rdp_epsilon_pixels` | `5.0` | RDP perpendicular-distance tolerance after SG smoothing. |
| `distance_unit` | `"scale_units"` | Pixel or configured physical distance unit. |
| `max_frame_gap` | `1` | Largest adjacent frame difference retained in one segment. |

The moving-average default is the paper-aligned mode. In `sg_filter_rdp` mode,
the sparse RDP path is linearly reconstructed on the original source-frame
coordinates before temporal derivatives are calculated. A `max_frame_gap`
larger than 1 is supported using real frame times, but smoothing still operates
over retained detection order and therefore emits an irregular-sampling warning.

## Classifier calculation

### Five-point velocity

Each gap-bounded segment is converted to Cartesian image coordinates
`x=centroid_y`, `y=-centroid_x` and smoothed. With consecutive source frames,
the velocity at point `i` is the fourth-order central difference:

```text
velocity[i] = (p[i-2] - 8*p[i-1] + 8*p[i+1] - p[i+2]) * FPS / 12
```

For uneven retained frames, the method solves the corresponding five-point
finite-difference weights from the actual frame offsets. The first two and last
two smoothed positions cannot have a five-point velocity.

Worked example: positions increase by exactly 1 um per frame at 10 FPS. For an
interior point, the numerator is 12 um per frame, division by 12 gives
`1 um/frame`, and multiplication by 10 gives `10 um/s`.

### Point angles and states

`turn_angle_degrees` is the unsigned angle between the velocity at the current
point and the velocity at the following point. A point is a high-angle
candidate only when its angle is strictly greater than the threshold, with a
small numerical tolerance.

- At least three consecutive finite angles at or below the threshold form a
  run.
- At least two consecutive angles above the threshold form a tumble.
- One isolated above-threshold angle forms a tumble only when the angle between
  `v[i-1]+v[i]` and `v[i+1]+v[i+2]` also exceeds the threshold.
- Other finite angle points remain `unclassified`.

Worked example at a 35-degree threshold: angles `[10, 12, 15]` form a run;
`[42, 48]` form a tumble. A lone `42` is a tumble only if its
`singleton_confirmation_angle_degrees` is also greater than 35.

### Events and durations

Adjacent points with the same classified state are grouped into events. The
event interval follows the paper convention:

```text
interval_seconds         = (end_frame - start_frame) / FPS
sampling_support_seconds = (support_end_frame - start_frame) / FPS
```

`support_end_frame` is the following detection needed to calculate the last
event point's angle. Worked example: an event spans classified frames 20 through
23 and uses frame 24 as its final support at 10 FPS. Its interval is 0.3 s and
its sampling support is 0.4 s.

## `04_Angle_Tumble_Summary.csv`

One row summarizes one particle.

### Identity, trimming, and support

| Column | Calculation or provenance |
|---|---|
| `strain`, `source_dataframe`, `particle` | Dataset label, resolved Tracker source, and trajectory ID. |
| `original_first_frame`, `original_last_frame`, `original_detection_count` | Particle endpoints and row count before boundary trimming. |
| `discarded_detection_count` | `original_detection_count - analyzed_detection_count`. |
| `analyzed_first_frame`, `analyzed_last_frame`, `analyzed_detection_count` | Endpoints and count after boundary trimming. |
| `segment_count` | Retained pieces after splitting at gaps larger than `max_frame_gap`. |
| `analyzable_angle_count` | Number of finite angle rows in the point table. |
| `classified_angle_count` | Point rows assigned `run` or `tumble`. |
| `unclassified_angle_count` | `analyzable_angle_count - classified_angle_count`. |
| `analyzed_tracking_time_seconds` | Sum over retained segments of `(last_frame - first_frame) / FPS`. Short unclassified segments contribute; excluded gaps do not. |

### Counts, time fractions, and frequencies

| Column | Calculation |
|---|---|
| `number_of_runs`, `number_of_tumbles` | Counts of contiguous run and tumble events. |
| `runs_per_tumble` | `number_of_runs / number_of_tumbles`; blank with no tumbles. |
| `total_run_interval_seconds`, `total_tumble_interval_seconds` | Sums of the corresponding event `interval_seconds`. |
| `tumble_time_fraction` | `total_tumble_interval_seconds / analyzed_tracking_time_seconds`. |
| `tumble_frequency_per_second` | `number_of_tumbles / analyzed_tracking_time_seconds`. |

Worked example: two tumble events totaling 0.4 s during 5 s of analyzed
tracking give `tumble_time_fraction=0.08` and
`tumble_frequency_per_second=2/5=0.4 s^-1`.

### Speed and duration distributions

| Column | Calculation |
|---|---|
| `mean_run_speed` | Arithmetic mean of run-event `mean_speed`; every run event receives equal weight. |
| `std_run_mean_speed_between_events` | Sample SD across run-event means. The name makes the event-level support explicit. |
| `mean_run_point_speed`, `std_run_point_speed`, `run_speed_point_count` | Mean, sample SD, and count after pooling finite `instantaneous_speed` values from all run-labelled points in this particle. |
| `mean_tumble_speed`, `std_tumble_speed` | Mean and sample SD across tumble-event `mean_speed` values. |
| `mean_run_interval_seconds`, `std_run_interval_seconds` | Mean and sample SD of run-event intervals. |
| `mean_tumble_interval_seconds`, `std_tumble_interval_seconds` | Mean and sample SD of tumble-event intervals. |
| `mean_time_between_tumble_starts_seconds`, `std_time_between_tumble_starts_seconds` | Statistics of consecutive tumble start-frame differences divided by FPS, calculated only within the same segment. |

Event and point weighting can differ. Example: one run event has point speeds
`[2,4]` and another has `[10,10,10,10]`. Event means are 3 and 10, so
`mean_run_speed=6.5`; all six points pooled give
`mean_run_point_speed=46/6=7.667`.

### Angular summaries

| Column | Calculation |
|---|---|
| `mean_run_angular_speed_degrees_per_point`, `std_run_angular_speed_degrees_per_point` | Mean and sample SD across run events' mean unsigned point angles. Despite `speed` in the name, the unit is degrees per trajectory point, not degrees per second. |
| `mean_tumble_angular_speed_degrees_per_point`, `std_tumble_angular_speed_degrees_per_point` | Corresponding event-level tumble statistics. |
| `mean_direction_change_within_runs_degrees`, `std_direction_change_within_runs_degrees` | Mean and sample SD across runs of the angle between the sum of their first three and last three velocity vectors. Runs shorter than three points do not contribute. |
| `mean_run_to_run_direction_change_degrees`, `std_run_to_run_direction_change_degrees` | Mean and sample SD across tumbles directly bracketed by qualifying runs, comparing the preceding run's last three and following run's first three velocity vectors. |

### Units, configuration, and status

| Column | Meaning |
|---|---|
| `distance_unit`, `speed_unit` | Resolved path and velocity units. |
| `smoothing_method` | Selected smoother. |
| `moving_average_window`, `triangular_smoothing_window` | Requested window for the corresponding smoother; unused values are blank. |
| `sg_filter_window_length`, `sg_filter_polyorder`, `rdp_epsilon_pixels` | Requested SG/RDP configuration when selected; otherwise blank. |
| `discard_initial_frames`, `discard_final_frames` | Requested boundary-window widths. |
| `tumble_threshold_angle_degrees` | Exported value of the `tumble_threshold_angle` argument. |
| `max_frame_gap` | Largest retained within-segment frame difference; blank represents `None`. |
| `analysis_status` | `complete`, `no_detections_after_discard`, `insufficient_points_for_five_point_velocity`, or `no_runs_or_tumbles_classified`. |

## `05_Angle_Tumble_Events.csv`

One row represents one contiguous classified run or tumble.

| Column | Calculation or provenance |
|---|---|
| `strain`, `source_dataframe`, `particle`, `segment_id` | Dataset, source, trajectory, and one-based gap-bounded segment. |
| `event_id` | Sequential event identifier within the particle, continuing across segments. |
| `event_type` | `run` or `tumble`. |
| `start_frame`, `end_frame` | First and last classified angle-point frames. |
| `support_end_frame` | Following detection used to calculate the final event angle. |
| `point_count` | Number of classified angle points in the event. |
| `interval_seconds` | `(end_frame - start_frame) / FPS`. |
| `sampling_support_seconds` | `(support_end_frame - start_frame) / FPS`. |
| `path_length` | Sum of smoothed displacement among event points, converted to `distance_unit`; the final support edge is not added. |
| `mean_speed`, `std_speed` | Mean and sample SD of finite point speeds in the event. |
| `mean_angular_speed_degrees_per_point`, `std_angular_speed_degrees_per_point` | Mean and sample SD of unsigned point angles in the event. |
| `direction_change_within_run_degrees` | For a run with at least three points, angle between summed first-three and last-three velocity vectors; otherwise blank. |
| `run_to_run_direction_change_degrees` | For a tumble immediately bracketed by qualifying runs, angle between preceding last-three and following first-three velocity sums; otherwise blank. |
| `preceding_run_event_id`, `following_run_event_id` | IDs of those bracketing runs when all adjacency/support rules pass. |
| `distance_unit`, `speed_unit` | Resolved path and speed units. |
| `smoothing_method`, `effective_smoothing_window` | Selected smoother and the span supported by the event's segment. |
| `tumble_threshold_angle_degrees` | Exported classifier threshold. |

## `06_Angle_Tumble_Points.csv`

One row represents one finite velocity-direction angle point and is the most
detailed classifier audit table.

| Column | Calculation or provenance |
|---|---|
| `strain`, `source_dataframe`, `particle`, `segment_id` | Dataset, source, trajectory, and segment. |
| `frame`, `next_frame`, `frame_delta_to_next` | Current angle-point frame, following velocity-support frame, and their difference. |
| `elapsed_time_seconds` | Cumulative observed time: completed-segment duration offset plus `(frame - segment_first_frame)/FPS`. Excluded gaps are omitted. |
| `raw_position_x_pixels`, `raw_position_y_pixels` | Cartesian `[centroid_y, -centroid_x]` before smoothing. |
| `smoothed_position_x_pixels`, `smoothed_position_y_pixels` | Frame-aligned position after the selected smoother. |
| `velocity_x`, `velocity_y` | Five-point derivative converted to `distance_unit/s`. |
| `instantaneous_speed` | `sqrt(velocity_x² + velocity_y²)`. |
| `turn_angle_degrees` | Unsigned angle between current and following velocity vectors. |
| `exceeds_tumble_threshold` | True only when `turn_angle_degrees` is strictly above the threshold. |
| `singleton_confirmation_angle_degrees` | Summed-neighbor confirmation angle for an isolated high-angle point; blank when not evaluated. |
| `state` | `run`, `tumble`, or `unclassified`. |
| `event_id` | Containing event ID; blank for unclassified points. |
| `distance_unit`, `speed_unit` | Resolved units. |
| `smoothing_method`, `effective_smoothing_window` | Selected smoother and segment-supported span. |
| `tumble_threshold_angle_degrees` | Exported threshold. |

## `07_Angle_Tumble_Population.csv`

One row summarizes the particle summary with equal weight per finite particle
value. It does not pool all events or points across particles.

### Counts and overall time

| Column | Calculation |
|---|---|
| `strain`, `source_dataframe` | Dataset label and resolved source. |
| `particle_count` | Number of particle-summary rows. |
| `particles_with_analyzable_angles` | Count with `analyzable_angle_count > 0`. |
| `particles_with_tumbles` | Count with `number_of_tumbles > 0`. |
| `total_number_of_runs`, `total_number_of_tumbles` | Sums of particle event counts. |
| `total_analyzed_tracking_time_seconds` | Sum of particle analyzed tracking times. |

### Equal-particle statistics

For every pair below, the `mean_*` is the arithmetic mean of the named finite
particle-level value and the `std_*` is its sample SD across particles:

| Population columns | Particle-level source value |
|---|---|
| `mean_tumble_frequency_per_second`, `std_tumble_frequency_per_second` | `tumble_frequency_per_second` |
| `mean_number_of_runs_per_particle`, `std_number_of_runs_per_particle` | `number_of_runs` |
| `mean_number_of_tumbles_per_particle`, `std_number_of_tumbles_per_particle` | `number_of_tumbles` |
| `mean_run_speed`, `std_run_speed_between_particles` | `mean_run_speed` |
| `mean_tumble_speed`, `std_tumble_speed_between_particles` | `mean_tumble_speed` |
| `mean_run_interval_seconds`, `std_run_interval_seconds_between_particles` | `mean_run_interval_seconds` |
| `mean_tumble_interval_seconds`, `std_tumble_interval_seconds_between_particles` | `mean_tumble_interval_seconds` |
| `mean_time_between_tumble_starts_seconds`, `std_time_between_tumble_starts_seconds_between_particles` | `mean_time_between_tumble_starts_seconds` |
| `mean_run_angular_speed_degrees_per_point`, `std_run_angular_speed_degrees_per_point_between_particles` | `mean_run_angular_speed_degrees_per_point` |
| `mean_tumble_angular_speed_degrees_per_point`, `std_tumble_angular_speed_degrees_per_point_between_particles` | `mean_tumble_angular_speed_degrees_per_point` |
| `mean_direction_change_within_runs_degrees`, `std_direction_change_within_runs_degrees_between_particles` | `mean_direction_change_within_runs_degrees` |
| `mean_run_to_run_direction_change_degrees`, `std_run_to_run_direction_change_degrees_between_particles` | `mean_run_to_run_direction_change_degrees` |

Example: finite per-particle run speeds `[5, 15] um/s` produce a population
`mean_run_speed=10 um/s`, regardless of how many run events each particle had.

The remaining columns—`distance_unit`, `speed_unit`, `smoothing_method`,
`moving_average_window`, `triangular_smoothing_window`,
`sg_filter_window_length`, `sg_filter_polyorder`, `rdp_epsilon_pixels`,
`discard_initial_frames`, `discard_final_frames`,
`tumble_threshold_angle_degrees`, and `max_frame_gap`—copy the resolved units
and configuration shared by the particle rows.

## `07_Angle_Tumble_Table_1.csv`

This 33-row vertical table has leading `strain`, followed by `Parameter` and
`Value`. It does not have velocity Table 1's `n` and `n_definition` columns.

- `Number of cells` is the particle-summary row count.
- `Number of cells that tumbled` counts particles with at least one tumble.
- Each of the nine motion parameters below is the equal-particle mean of the
  corresponding finite particle metric and is followed immediately by `±std`
  and `±sem`. The repeated uncertainty labels must be interpreted together with
  their preceding parameter row.
- `±std` uses `ddof=1`; `±sem = SD/sqrt(n)` with the metric-specific finite
  particle count. Both are blank when fewer than two particles contribute.
- Non-tumbling particles with positive analyzed time contribute zero tumble
  frequency but do not contribute to tumble-only means.

The nine distribution labels and source columns are:

| `Parameter` label | Particle-summary source |
|---|---|
| `Run speed (<speed_unit>)` | `mean_run_speed` |
| `Tumble speed (<speed_unit>)` | `mean_tumble_speed` |
| `Angular speed while running (°/point)` | `mean_run_angular_speed_degrees_per_point` |
| `Angular speed while tumbling (°/point)` | `mean_tumble_angular_speed_degrees_per_point` |
| `Run interval (s)` | `mean_run_interval_seconds` |
| `Tumble interval (s)` | `mean_tumble_interval_seconds` |
| `Change in direction from run to run (°)` | `mean_run_to_run_direction_change_degrees` |
| `Change in direction during runs (°)` | `mean_direction_change_within_runs_degrees` |
| `Tumble frequency (s^-1)` | `tumble_frequency_per_second` |

The final rows are:

| `Parameter` label | Calculation |
|---|---|
| `Number of events (runs, tumbles)` | String containing the two population totals as `runs, tumbles`. |
| `Total tracking time (s)` | Sum of particle `analyzed_tracking_time_seconds`. |
| `Body diameter (<distance_unit>)` | Mean of per-particle mean finite `minor_axis_length` values after trimming and spatial scaling. |
| `Body length (<distance_unit>)` | Mean of per-particle mean finite `major_axis_length` values after trimming and spatial scaling. |

The body-size values are equal-particle region-property ellipse-axis proxies,
not direct caliper measurements. If morphology columns are missing, these
values are blank and a warning is issued without blocking classification.
