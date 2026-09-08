# Velocity-based run/tumble parameters and calculations

This guide covers `calculate_velocity_tumble_statistics()`, MiMoSA's
two-dimensional Najafi-style speed/angular-velocity classifier, and these
normal notebook exports:

- `06_Additional_Analysis/08_Velocity_Tumble_Summary.csv`
- `06_Additional_Analysis/09_Velocity_Tumble_Events.csv`
- `06_Additional_Analysis/10_Velocity_Tumble_Points.csv`
- `06_Additional_Analysis/11_Velocity_Tumble_Population.csv`
- `06_Additional_Analysis/11_Velocity_Tumble_Table_1.csv`

Canonical speed names explicitly identify point, time-weighted, pooled, and
equal-particle aggregation. See
[STATS_EXPORTED_DATA_FILES_SUMMARY.md](STATS_EXPORTED_DATA_FILES_SUMMARY.md)
for the legacy-name map and shared export conventions.

## Calculation inputs

| Argument | Default | Effect |
|---|---:|---|
| `particle_ids` | `None` | All particles, one ID, or a list/tuple of IDs. |
| `discard_initial_frames`, `discard_final_frames` | `10`, `10` | Per-particle source-frame boundary windows removed. |
| `smoothing_method` | `"triangular_smoothing"` | `moving_average`, `triangular_smoothing`, or `sg_filter_rdp`. |
| `moving_average_window` | `5` | Complete centered odd moving-average span. |
| `triangular_smoothing_window` | `17` | Odd triangular span with edge-value padding. |
| `sg_filter_window_length` | `5` | Odd Savitzky-Golay span. |
| `sg_filter_polyorder` | `2` | SG polynomial order, smaller than the SG span. |
| `rdp_epsilon_pixels` | `5.0` | RDP tolerance after SG smoothing; the result is reconstructed on source-frame coordinates. |
| `speed_drop_ratio_threshold` | `0.7` | Minimum `speed_minimum_depth / minimum_speed` for a speed-dip candidate. |
| `speed_period_depth_fraction` | `0.2` | Fraction of dip depth defining the low-speed candidate period. |
| `angular_change_coefficient` | `0.8` | Coefficient in the duration-dependent angular-change threshold. |
| `minimum_heading_speed` | `None` | Optional selected-unit/s speed floor; headings at or below it are undefined. |
| `speed_extrema_prominence` | `None` | Optional speed-dip prominence in the selected speed unit. |
| `angular_velocity_extrema_prominence` | `None` | Optional angular-speed peak prominence in rad/s. |
| `extrema_min_distance` | `1` | Minimum candidate-extrema separation in retained kinematic samples. |
| `persistence_interval_seconds` | `1/6` | Within-run time separation used for directional persistence `p`. |
| `run_direction_fit_points` | `4` | Smoothed points fitted on each side of a tumble to calculate `R`. Minimum 2. |
| `distance_unit` | `"scale_units"` | Pixel or configured physical distance unit. |
| `max_frame_gap` | `1` | Largest adjacent frame difference retained in one segment. |
| `bootstrap_resamples` | `10000` | Whole-particle bootstrap replicates for pooled `vR` and `vT` intervals. Minimum 2. |
| `bootstrap_confidence_level` | `0.95` | Central percentile confidence level, strictly between 0 and 1. |
| `bootstrap_random_seed` | `0` | Reproducible nonnegative seed; `None` requests nondeterministic entropy. |

The 17-point triangular default corresponds to the associated implementation at
60 FPS. Other smoothers and windows are explicit alternative analyses. With
`max_frame_gap>1`, elapsed times remain correct, but smoothing operates over
retained detection order and the method emits an irregular-sampling warning.

## Classifier calculation

### Backward-difference kinematics

After trimming, gap splitting, Cartesian conversion
`x=centroid_y, y=-centroid_x`, and smoothing, the method calculates:

```text
dt_i      = (frame_i - frame_(i-1)) / FPS
velocity_i = (position_i - position_(i-1)) / dt_i
speed_i    = ||velocity_i||
heading_i  = atan2(velocity_y_i, velocity_x_i)
```

The velocity is assigned to the ending point `i`. Worked example: positions
differ by 3 um between frames 10 and 12 at 10 FPS. The elapsed time is 0.2 s,
so the speed is 15 um/s and is stored on frame 12.

A heading is finite only when speed is greater than the maximum of numerical
tolerance and `minimum_heading_speed`, if supplied. For consecutive valid
headings, the principal signed change is

```text
delta_heading = atan2(sin(heading_i-heading_previous),
                      cos(heading_i-heading_previous))
angular_change = abs(delta_heading)
angular_velocity_magnitude = angular_change / elapsed_heading_time
```

The previous valid heading can lie before zero/low-speed samples, so a pause
does not itself create an artificial turn. Calculations never bridge a segment
boundary.

### Speed-dip candidates

For each local speed minimum with genuine left and right local maxima:

```text
depth = max(left_maximum - minimum_speed,
            right_maximum - minimum_speed,
            0)
depth_ratio = depth / minimum_speed
```

A numerical zero minimum has infinite ratio when depth is positive. The
candidate passes when `depth_ratio >= speed_drop_ratio_threshold`. Its connected
low-speed region around the minimum is bounded by the surrounding maxima and
satisfies:

```text
speed - minimum_speed <= speed_period_depth_fraction * depth
```

Worked example: flanking maxima are 12 and 10 um/s and the minimum is 4 um/s.
Depth is `max(8,6)=8`, ratio is `8/4=2`, and the candidate passes a 0.7
threshold. With fraction 0.2, its low-speed region contains connected values at
or below `4 + 0.2*8 = 5.6 um/s`.

### Angular candidates and accepted tumbles

Each local angular-velocity maximum is bracketed by local angular minima. The
method sums absolute heading changes inside that bracket and requires:

```text
total_directional_change >
    sqrt(angular_change_coefficient * surrounding_duration_seconds)
```

Example: with coefficient 0.8 and surrounding duration 0.2 s, the threshold is
`sqrt(0.16)=0.4 rad`. A cumulative change of 0.6 rad passes.

A tumble is accepted only when a passing speed period and passing angular
period overlap. The accepted duration and state region use the angular period,
not their overlap or the speed period. Overlapping or directly adjacent
accepted angular periods are merged. A segment needs at least five finite
angular-velocity samples to be analyzable; otherwise its positions remain
unclassified.

## Complete and censored episodes

The classifier's descriptive run intervals include the segment interval before
the first tumble, between tumbles, and after the last tumble. Boundary support
is treated separately for Table 1 episode means:

- A tumble is complete only when both its start and end lie strictly inside
  its segment.
- A run is complete only when it lies between two complete tumbles.
- Leading, trailing, no-tumble, or boundary-touching intervals are censored.
- `mean_run_interval_seconds` uses all reconstructed run intervals, whereas
  `t_R` uses only complete runs. `mean_tumble_interval_seconds` uses all tumble
  events, whereas `t_T` uses only complete tumbles.

Worked example: a 0-10 s segment has complete tumbles at 2-2.5 s and 5-5.4 s.
Runs 0-2 s and 5.4-10 s are censored; the 2.5-5 s run is complete. Thus
`t_R=2.5 s`, `t_R_interval_count=1`, and
`t_R_censored_interval_count=2`. Both tumbles contribute to `t_T`.

## Speed aggregation levels

The names distinguish three calculations:

```text
point_mean_run_speed
    = arithmetic mean of finite point speeds labelled run

time_weighted_vR for one particle
    = sum(speed_i * dt_i) / sum(dt_i)
      over positive intervals where current and previous points are both run

pooled_time_weighted_vR
    = sum(particle time_weighted_vR * particle vR_support_seconds)
      / sum(particle vR_support_seconds)
```

The same rules apply to tumble fields ending in `vT`. State-transition
intervals, such as `run -> tumble`, contribute to neither time-weighted state
speed.

Worked particle example: same-state run interval speeds `[2,8] um/s` with
durations `[0.75,0.25] s` give
`time_weighted_vR=(2*0.75+8*0.25)/1.0=3.5 um/s`; their unweighted mean is
5 um/s.

Worked population example: particles have time-weighted values 10 and 20 um/s
with run support 9 and 1 s. `pooled_time_weighted_vR=11 um/s`, while
`particle_mean_time_weighted_vR=(10+20)/2=15 um/s`.

## Directional persistence and rotational diffusion

### Within-run persistence `p`

Within each contiguous block where the current and preceding points are runs
and headings are finite, the target heading at
`source_time + persistence_interval_seconds` is circularly interpolated. Each
supported observation is `cos(principal direction change)`, and `p` is their
mean. A tumble, undefined heading, or segment boundary breaks support.

Example: supported direction changes of 0 and 60 degrees give cosines 1 and
0.5, so `p=0.75`.

### Consecutive-run persistence `R`

For a tumble with enough adjacent run support, separate least-squares x/time
and y/time slopes are fitted to the last and first
`run_direction_fit_points`. Their directions are compared using a signed
principal angle; `R` is the mean cosine across supported transitions. A
90-degree run-to-run change contributes approximately zero.

### Run-only angular MSD and `Dr`

Finite run headings are split by particle, segment, state continuity, and
heading availability, then unwrapped inside each block. Squared changes are
pooled by exact source-frame lag:

```text
RMSD(tau) = mean((alpha(t + tau) - alpha(t))²)
```

The requested fit ceiling is
`sum(total_run_interval_seconds) / sum(number_of_runs)`, which includes
censored runs and is intentionally different from complete-run `t_R`. Each
pair-pooled lag mean receives equal weight in the origin-constrained fit:

```text
RMSD(tau) = 2 * Dr * tau
slope = sum(tau * RMSD) / sum(tau²)
Dr = slope / 2
```

At least two positive supported lag bins are required. Example: lag means
`RMSD(0.1)=0.04` and `RMSD(0.2)=0.08 rad²` give slope 0.4 and
`Dr=0.2 rad²/s`.

## `08_Velocity_Tumble_Summary.csv`

One row summarizes one particle.

### Identity, support, counts, and time

| Column | Calculation or provenance |
|---|---|
| `strain`, `source_dataframe`, `particle` | Dataset label, resolved source, and trajectory ID. |
| `original_first_frame`, `original_last_frame`, `original_detection_count` | Particle endpoints and row count before trimming. |
| `discarded_detection_count` | Original minus analyzed detection count. |
| `analyzed_first_frame`, `analyzed_last_frame`, `analyzed_detection_count` | Endpoints and count after trimming. |
| `segment_count` | Pieces after gap splitting. |
| `analyzable_segment_count` | Segments with at least five finite angular-velocity samples. |
| `kinematic_point_count` | Count of finite `angular_velocity_magnitude_radians_per_second` values. |
| `unclassified_detection_count` | Point rows whose `state` is `unclassified`. |
| `number_of_runs` | All reconstructed run intervals, including boundary-censored intervals. |
| `number_of_tumbles` | Accepted merged tumble events. |
| `runs_per_tumble` | `number_of_runs / number_of_tumbles`; blank with no tumbles. |
| `analyzed_tracking_time_seconds` | Sum of complete post-trim segment spans. |
| `classified_tracking_time_seconds` | Sum of smoothed-position-supported spans for analyzable segments. Moving-average boundaries can reduce this relative to analyzed time. |
| `total_run_interval_seconds` | Sum of all reconstructed run intervals, including censored intervals. |
| `total_tumble_interval_seconds` | Sum of accepted event durations. |
| `total_unclassified_interval_seconds` | `max(analyzed_tracking_time_seconds - classified_tracking_time_seconds, 0)`. |
| `tumble_time_fraction` | `total_tumble_interval_seconds / classified_tracking_time_seconds`. |
| `tumble_frequency_per_second` | `number_of_tumbles / classified_tracking_time_seconds`. |

### Point-level and descriptive interval metrics

| Column | Calculation |
|---|---|
| `point_mean_run_speed`, `point_std_run_speed` | Mean and sample SD across finite run-labelled point speeds. |
| `point_mean_tumble_speed`, `point_std_tumble_speed` | Mean and sample SD across finite tumble-labelled point speeds. |
| `mean_run_interval_seconds`, `std_run_interval_seconds` | Mean and sample SD across all reconstructed run intervals, including censored boundaries. |
| `mean_tumble_interval_seconds`, `std_tumble_interval_seconds` | Mean and sample SD across all accepted tumble durations. |
| `mean_time_between_tumble_starts_seconds`, `std_time_between_tumble_starts_seconds` | Statistics of consecutive tumble start times within the same segment. |
| `mean_run_angular_velocity_magnitude_radians_per_second`, `std_run_angular_velocity_magnitude_radians_per_second` | Point-level mean and sample SD for finite run angular-velocity magnitudes. |
| `mean_tumble_angular_velocity_magnitude_radians_per_second`, `std_tumble_angular_velocity_magnitude_radians_per_second` | Corresponding tumble point statistics. |

### Table 1 particle metrics and support

| Column | Calculation |
|---|---|
| `time_weighted_vR`, `time_weighted_vT` | Per-particle elapsed-time-weighted same-state speed. |
| `vR_interval_count`, `vT_interval_count` | Same-state interval counts. |
| `vR_support_seconds`, `vT_support_seconds` | Sums of contributing interval durations. |
| `t_R`, `t_T` | Means of complete uncensored run and tumble episodes. |
| `std_t_R`, `std_t_T` | Sample SD across this particle's complete run or tumble episodes. |
| `sem_t_R` | `std_t_R / sqrt(t_R_interval_count)`. |
| `t_R_interval_count`, `t_T_interval_count` | Complete episode counts. |
| `t_R_censored_interval_count`, `t_T_censored_interval_count` | Boundary-censored episode counts excluded from `t_R` and `t_T`. |
| `p`, `std_p`, `p_direction_change_count` | Mean, sample SD, and count of supported within-run direction cosines. |
| `R`, `std_R`, `R_run_transition_count` | Mean, sample SD, and count of supported fitted run-to-run transition cosines. |

### Units, configuration, and status

| Column | Meaning |
|---|---|
| `distance_unit`, `speed_unit`, `angular_velocity_unit` | Resolved units; angular velocity is `rad/s`. |
| `smoothing_method` | Selected smoother. |
| `moving_average_window`, `triangular_smoothing_window` | Requested span for the corresponding smoother; unused values are blank. |
| `sg_filter_window_length`, `sg_filter_polyorder`, `rdp_epsilon_pixels` | Requested SG/RDP settings when selected; otherwise blank. |
| `discard_initial_frames`, `discard_final_frames` | Boundary-window widths. |
| `speed_drop_ratio_threshold`, `speed_period_depth_fraction`, `angular_change_coefficient` | Classifier thresholds copied from the call. |
| `minimum_heading_speed` | Requested heading floor; blank for numerical-tolerance-only behavior. |
| `speed_extrema_prominence`, `angular_velocity_extrema_prominence` | Optional prominence filters; blank when not requested. |
| `extrema_min_distance` | Requested minimum candidate-extrema separation. |
| `persistence_interval_seconds` | Target interval for `p`. |
| `run_direction_fit_points` | Points used in each adjacent-run fit for `R`. |
| `max_frame_gap` | Largest retained within-segment frame difference; blank represents `None`. |
| `analysis_status` | `analyzed_with_tumbles`, `analyzed_no_tumbles`, `insufficient_kinematic_points_for_extrema`, or `no_detections_after_discard`. |

## `09_Velocity_Tumble_Events.csv`

One row is one accepted, possibly merged tumble. `event_type` is always
`tumble`.

### Event identity and measured motion

| Column | Calculation or provenance |
|---|---|
| `strain`, `source_dataframe`, `particle`, `segment_id` | Dataset, source, trajectory, and one-based segment. |
| `event_id` | Sequential accepted-tumble ID within the particle. |
| `event_type` | Constant `tumble`. |
| `start_frame`, `end_frame` | Merged angular-defined event bounds. |
| `support_start_frame` | Previous valid-heading frame supporting the first event angular change; falls back to the prior row when needed. |
| `point_count` | Inclusive event point count. |
| `interval_seconds` | `(end_frame - start_frame) / FPS`. |
| `sampling_support_seconds` | `(end_frame - support_start_frame) / FPS`. |
| `path_length` | Sum of smoothed within-event step lengths; the support edge is excluded. |
| `mean_speed`, `minimum_speed`, `maximum_speed` | Statistics of finite ending-point speeds inside event bounds. |
| `mean_angular_velocity_magnitude_radians_per_second`, `maximum_angular_velocity_magnitude_radians_per_second` | Mean and maximum finite angular-velocity magnitude inside the event. |

### Representative speed and angular evidence

When accepted angular candidates merge, the diagnostic fields below describe
the representative matched candidate chosen by directional-change strength and
then speed-depth ratio. Event bounds still describe the merged angular region.

| Column | Calculation or provenance |
|---|---|
| `speed_minimum_frame`, `speed_minimum_value` | Representative speed minimum location and value. |
| `speed_minimum_depth` | Maximum of left-maximum minus minimum and right-maximum minus minimum. |
| `speed_depth_ratio` | `speed_minimum_depth / speed_minimum_value`, with zero-speed handling described above. |
| `speed_left_maximum_frame`, `speed_right_maximum_frame` | Bracketing speed maxima. |
| `speed_period_start_frame`, `speed_period_end_frame` | Connected low-speed candidate bounds. |
| `angular_velocity_maximum_frame`, `angular_velocity_maximum_value` | Representative angular peak location and value. |
| `angular_velocity_maximum_height` | Maximum of peak minus left angular minimum and peak minus right angular minimum. |
| `angular_left_minimum_frame`, `angular_right_minimum_frame` | Bracketing angular-velocity minima. |
| `angular_total_directional_change_radians` | Sum of absolute heading changes within the bracketing angular minima. |
| `angular_directional_change_threshold_radians` | `sqrt(angular_change_coefficient * surrounding_duration)`. |
| `angular_period_start_frame`, `angular_period_end_frame` | Final merged angular event bounds. |
| `matching_overlap_start_frame`, `matching_overlap_end_frame` | Overlap bounds for the representative speed/angular candidate match. |
| `merged_candidate_count` | Number of accepted candidate matches merged into the event. |
| `evidence_scope` | `representative_matched_candidate`, identifying the diagnostic scope. |

### Run-direction fit evidence and configuration

| Column | Calculation or provenance |
|---|---|
| `preceding_run_fit_start_frame`, `preceding_run_fit_end_frame` | Frames used in the least-squares run-direction fit before the tumble. |
| `following_run_fit_start_frame`, `following_run_fit_end_frame` | Frames used in the fit after the tumble. |
| `preceding_run_direction_radians`, `following_run_direction_radians` | Fitted directions from x/time and y/time slopes. |
| `run_to_run_turn_angle_radians`, `run_to_run_turn_angle_degrees` | Signed principal change from preceding to following fitted direction. |
| `run_to_run_directional_cosine` | Cosine of that change; the observation pooled into `R`. |
| `distance_unit`, `speed_unit`, `angular_velocity_unit` | Resolved units. |
| `smoothing_method`, `effective_smoothing_window` | Selected smoother and segment-supported span. |
| `speed_drop_ratio_threshold`, `speed_period_depth_fraction`, `angular_change_coefficient`, `minimum_heading_speed` | Classifier configuration copied to the event. |

## `10_Velocity_Tumble_Points.csv`

One row represents one retained detection in a processed segment, including
boundary rows with unavailable derivatives.

### Time, position, and kinematics

| Column | Calculation or provenance |
|---|---|
| `strain`, `source_dataframe`, `particle`, `segment_id` | Dataset, source, trajectory, and segment. |
| `frame`, `previous_frame`, `frame_delta_from_previous` | Current frame and backward support identity; previous fields are blank on the segment's first row. |
| `elapsed_time_from_previous_seconds` | Backward frame difference divided by FPS. |
| `elapsed_time_seconds` | Cumulative observed time with excluded inter-segment gaps omitted. |
| `raw_position_x_pixels`, `raw_position_y_pixels` | Cartesian `[centroid_y, -centroid_x]` before smoothing. |
| `smoothed_position_x_pixels`, `smoothed_position_y_pixels` | Position after the selected frame-aligned smoother. |
| `velocity_x`, `velocity_y` | Backward-difference velocity assigned to this ending frame. |
| `speed` | Euclidean magnitude of velocity. |
| `heading_radians` | `atan2(velocity_y, velocity_x)` when speed exceeds the heading floor. |
| `previous_valid_heading_frame` | Earlier frame used for this point's angular change. |
| `angular_elapsed_time_seconds` | Time since that previous valid heading. |
| `angular_change_radians` | Absolute principal heading change. |
| `angular_velocity_magnitude_radians_per_second` | `angular_change_radians / angular_elapsed_time_seconds`. |

### Persistence, candidates, and state

| Column | Calculation or provenance |
|---|---|
| `persistence_target_elapsed_time_seconds` | Source point time plus `persistence_interval_seconds` when a same-run target is supported. |
| `persistence_target_frame` | Circular-interpolation target expressed on the frame axis; it can be fractional. |
| `persistence_direction_change_radians` | Signed principal change to the interpolated target heading. |
| `persistence_cosine` | Cosine pooled into `p`. |
| `is_speed_minimum` | True for a local speed minimum before depth-ratio qualification. |
| `speed_minimum_passes` | True when that minimum has bracketing maxima and passes the ratio threshold. |
| `is_angular_velocity_maximum` | True for a local angular-velocity maximum before directional-change qualification. |
| `angular_velocity_maximum_passes` | True when the cumulative angular criterion passes. |
| `state` | `run` for supported positions in an analyzable segment unless overwritten by an accepted tumble; otherwise `tumble` or `unclassified`. A run boundary row can have blank speed/angular velocity. |
| `event_id` | Accepted tumble ID for tumble points; blank for run/unclassified points. |
| `distance_unit`, `speed_unit`, `angular_velocity_unit` | Resolved units. |
| `smoothing_method`, `effective_smoothing_window` | Selected smoother and segment-supported span. |

## `11_Velocity_Tumble_Population.csv`

One row combines all particle summaries. Its names state whether a metric is
pooled by measurement support or gives every particle equal weight.

### Counts, time, frequency, and equal-particle descriptive values

| Column | Calculation |
|---|---|
| `strain`, `source_dataframe` | Dataset label and resolved source. |
| `particle_count` | Number of particle-summary rows. |
| `particles_with_kinematic_points` | Count with `kinematic_point_count > 0`. |
| `particles_with_tumbles` | Count with `number_of_tumbles > 0`. |
| `total_number_of_runs`, `total_number_of_tumbles` | Sums of particle counts. |
| `total_analyzed_tracking_time_seconds`, `total_classified_tracking_time_seconds`, `total_unclassified_interval_seconds` | Sums of corresponding particle times. |
| `pooled_tumble_frequency_per_second` | `total_number_of_tumbles / total_classified_tracking_time_seconds`. |
| `mean_particle_tumble_frequency_per_second`, `std_particle_tumble_frequency_per_second` | Mean and sample SD of finite particle tumble frequencies. |
| `mean_particle_tumble_time_fraction`, `std_particle_tumble_time_fraction` | Equal-particle statistics of tumble fractions. |
| `mean_number_of_runs_per_particle`, `std_number_of_runs_per_particle` | Equal-particle run-count statistics. |
| `mean_number_of_tumbles_per_particle`, `std_number_of_tumbles_per_particle` | Equal-particle tumble-count statistics. |
| `particle_mean_point_run_speed`, `particle_std_point_run_speed` | Mean and sample SD across finite particle `point_mean_run_speed` values. |
| `particle_mean_point_tumble_speed`, `particle_std_point_tumble_speed` | Equivalent equal-particle point-speed statistics for tumbles. |
| `mean_run_interval_seconds`, `std_run_interval_seconds_between_particles` | Equal-particle statistics of particle mean all-run intervals. |
| `mean_tumble_interval_seconds`, `std_tumble_interval_seconds_between_particles` | Equal-particle statistics of particle mean all-tumble intervals. |
| `mean_time_between_tumble_starts_seconds`, `std_time_between_tumble_starts_seconds_between_particles` | Equal-particle start-to-start statistics. |
| `mean_run_angular_velocity_magnitude_radians_per_second`, `std_run_angular_velocity_between_particles` | Equal-particle statistics of particle run angular-velocity means. |
| `mean_tumble_angular_velocity_magnitude_radians_per_second`, `std_tumble_angular_velocity_between_particles` | Equivalent tumble statistics. |

### Pooled speed metrics and bootstrap intervals

| Column | Calculation |
|---|---|
| `pooled_time_weighted_vR`, `pooled_time_weighted_vT` | Particle time-weighted values weighted by their state-specific support seconds. |
| `pooled_time_weighted_vR_ci_lower`, `pooled_time_weighted_vR_ci_upper` | Central percentile particle-cluster bootstrap bounds for pooled run speed. |
| `pooled_time_weighted_vT_ci_lower`, `pooled_time_weighted_vT_ci_upper` | Corresponding tumble-speed bounds. |
| `vR_interval_count`, `vT_interval_count` | Sums of particle same-state interval counts. |
| `vR_support_seconds`, `vT_support_seconds` | Sums of particle state-specific support. |
| `vR_particle_count`, `vT_particle_count` | Counts with finite state value and positive support; bootstrap cluster counts. |

Each bootstrap replicate samples the contributing particles with replacement,
keeping their values and support weights together, then recomputes the pooled
weighted mean. Fewer than two contributing particles yields blank bounds.

### Exact pooled Table 1 metrics

| Column | Calculation |
|---|---|
| `t_R`, `std_t_R`, `sem_t_R`, `t_R_interval_count` | Exact pooled complete-run mean, sample SD, SEM, and count reconstructed from particle counts, means, and within-particle SDs. |
| `t_R_censored_interval_count` | Sum of particle censored run counts. |
| `t_T`, `std_t_T`, `t_T_interval_count` | Exact pooled complete-tumble mean, sample SD, and count. |
| `t_T_censored_interval_count` | Sum of particle censored tumble counts. |
| `p`, `std_p`, `p_direction_change_count` | Exact pooled mean, sample SD, and count of within-run direction cosines. |
| `R`, `std_R`, `R_run_transition_count` | Exact pooled mean, sample SD, and count of fitted transition cosines. |

For groups with means `m_g`, sample SDs `s_g`, and counts `n_g`, the exact
pooled variance is reconstructed as:

```text
pooled_mean = sum(n_g*m_g) / N
SS_within   = sum((n_g-1)*s_g²)
SS_between  = sum(n_g*(m_g-pooled_mean)²)
pooled_SD   = sqrt((SS_within+SS_between)/(N-1))
```

If a legacy particle file lacks a required within-particle SD for a group with
more than one observation, the exact pooled SD is unavailable even though the
pooled mean can still be calculated.

### Equal-particle Table 1 alternatives

| Columns | Calculation |
|---|---|
| `particle_mean_time_weighted_vR`, `particle_std_time_weighted_vR`, `particle_sem_time_weighted_vR` | Mean, sample SD, and SEM across finite particle `time_weighted_vR`. |
| `particle_mean_time_weighted_vT`, `particle_std_time_weighted_vT`, `particle_sem_time_weighted_vT` | Corresponding tumble statistics. |
| `mean_particle_t_R`, `std_particle_t_R` | Equal-particle statistics of particle `t_R`. |
| `mean_particle_t_T`, `std_particle_t_T` | Equal-particle statistics of particle `t_T`. |
| `mean_particle_p`, `std_particle_p` | Equal-particle statistics of particle `p`. |
| `mean_particle_R`, `std_particle_R` | Equal-particle statistics of particle `R`. |

### Units and reproducibility settings

The remaining columns copy the resolved units and call configuration:

- `distance_unit`, `speed_unit`, `angular_velocity_unit`;
- `smoothing_method`, `moving_average_window`,
  `triangular_smoothing_window`, `sg_filter_window_length`,
  `sg_filter_polyorder`, and `rdp_epsilon_pixels`;
- `discard_initial_frames`, `discard_final_frames`, `max_frame_gap`;
- `speed_drop_ratio_threshold`, `speed_period_depth_fraction`,
  `angular_change_coefficient`, `minimum_heading_speed`,
  `speed_extrema_prominence`, `angular_velocity_extrema_prominence`, and
  `extrema_min_distance`;
- `persistence_interval_seconds` and `run_direction_fit_points`; and
- `pooled_speed_bootstrap_resamples`,
  `pooled_speed_bootstrap_confidence_level`, and
  `pooled_speed_bootstrap_random_seed`.

The exported bootstrap names intentionally differ from the call arguments by
adding the `pooled_speed_` prefix because they describe population-output
provenance.

## `11_Velocity_Tumble_Table_1.csv`

This 32-row vertical table has columns `strain`, `Parameter`, `Value`, `n`, and
`n_definition`. The `strain` column and the paper-style `Strain` parameter row
both come from the `Stats` strain argument.

### Parameter rows and sample units

| `Parameter` row | `Value` calculation | Meaning of `n` |
|---|---|---|
| `Strain` | Dataset label | Particles in the population summary |
| `pooled_time_weighted_vR (<speed_unit>)` | Pooled run speed | Run-state speed intervals |
| `pooled_time_weighted_vR_ci_lower (<speed_unit>)`, `..._ci_upper (...)` | Bootstrap bounds | Contributing particle clusters |
| `particle_mean_time_weighted_vR (<speed_unit>)` | Equal-particle run mean | Contributing particles |
| `particle_std_time_weighted_vR (<speed_unit>)` | Between-particle sample SD | Contributing particles |
| `particle_sem_time_weighted_vR (<speed_unit>)` | Between-particle SEM | Contributing particles |
| `pooled_time_weighted_vT (<speed_unit>)` | Pooled tumble speed | Tumble-state speed intervals |
| `pooled_time_weighted_vT_ci_lower (<speed_unit>)`, `..._ci_upper (...)` | Bootstrap bounds | Contributing particle clusters |
| `particle_mean_time_weighted_vT (<speed_unit>)` | Equal-particle tumble mean | Contributing particles |
| `particle_std_time_weighted_vT (<speed_unit>)` | Between-particle sample SD | Contributing particles |
| `particle_sem_time_weighted_vT (<speed_unit>)` | Between-particle SEM | Contributing particles |
| `pooled_speed_bootstrap_resamples` | Bootstrap setting | Not applicable |
| `pooled_speed_bootstrap_confidence_level` | Bootstrap setting | Not applicable |
| `pooled_speed_bootstrap_random_seed` | Bootstrap setting | Not applicable |
| `Mean tR (s)` | Pooled complete-run `t_R` | Complete uncensored runs |
| `±std` | Pooled complete-run `std_t_R` | Complete uncensored runs |
| `±sem` | Pooled complete-run `sem_t_R` | Complete uncensored runs |
| `Mean tT (s)` | Pooled complete-tumble `t_T` | Complete uncensored tumbles |
| `std_t_T (s)` | Pooled complete-tumble sample SD | Complete uncensored tumbles |
| `p`, `std_p` | Pooled within-run persistence and sample SD | Supported direction-cosine observations |
| `R`, `std_R` | Pooled consecutive-run persistence and sample SD | Supported fitted transition cosines |
| `Dr (rad²/s)` | Origin-constrained angular-MSD fit | Fitted lag-bin means |
| `Dr fit max lag (s)` | Requested fit ceiling from all reconstructed runs | Reconstructed runs, including censored runs |
| `Dr largest fitted lag (s)` | Largest supported lag actually retained | Fitted lag-bin means |
| `Dr fit lag count` | Number of fitted lag bins | Not applicable; count is in `Value` |
| `Dr fit direction-pair count` | Pair support summed over fitted lags | Not applicable; count is in `Value` |
| `Dr fit uncentered R^2` | `1 - residual_SS/sum(RMSD²)` for the origin fit | Fitted lag-bin means |
| `Dr fit status` | `fitted` or an explicit failure reason | Not applicable |

For uncertainty rows with `n<2`, `Value` is blank and `n_definition` states why
sample SD or SEM was not calculated. Najafi et al. did not publish these
uncertainty and audit rows; they are explicit MiMoSA extensions. The lag-level
angular-MSD curve is calculated internally but is not exported as a standalone
table.
