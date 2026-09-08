# Geometric-turn analysis parameters and calculations

This guide covers the thresholded geometric-turn analysis returned by
`calculate_turn_statistics()` and normally saved as:

- `06_Additional_Analysis/02_Turn_Summary.csv`
- `06_Additional_Analysis/03_Turn_Angles.csv`

This analysis detects large processed-path angles. It does not classify
run/tumble states and does not merge consecutive qualifying vertices into one
event. See [STATS_EXPORTED_DATA_FILES_SUMMARY.md](STATS_EXPORTED_DATA_FILES_SUMMARY.md)
for conventions shared by every export.

## Calculation inputs

The current method uses these exact argument names:

| Argument | Default | Effect |
|---|---:|---|
| `particle_ids` | `None` | All particles, one ID, or a list/tuple of IDs. |
| `minimum_turn_angle_degrees` | `15.0` | Inclusive absolute-angle threshold for `is_turn`. Must be between 0 and 180. |
| `smoothing_method` | `"sg_filter_rdp"` | `moving_average`, `triangular_smoothing`, or `sg_filter_rdp`. |
| `moving_average_window` | `5` | Complete centered odd moving-average span. |
| `triangular_smoothing_window` | `17` | Odd triangular-filter span with edge-value padding. |
| `sg_filter_window_length` | `5` | Odd Savitzky-Golay span. |
| `sg_filter_polyorder` | `2` | SG polynomial order; must be smaller than the SG span. |
| `rdp_epsilon_pixels` | `5.0` | Ramer-Douglas-Peucker perpendicular-distance tolerance after SG smoothing. |
| `distance_unit` | `"scale_units"` | `pixels`, `scale_units`, or the configured physical unit. |
| `max_frame_gap` | `1` | Split a track when the adjacent frame difference is larger. `None` permits all gaps. |
| `discard_initial_frames` | `10` | Initial per-particle source-frame window removed. |
| `discard_final_frames` | `10` | Final per-particle source-frame window removed. |

Only the settings for the selected smoother affect the path. Unused smoother
columns remain blank in the output. A warning is emitted when
`max_frame_gap=None` or is greater than 1 because the smoothing window then
operates over detection order despite irregular source-frame sampling.

## Processing sequence

For each selected particle, the method:

1. removes the requested initial and final source-frame windows;
2. splits the retained track at adjacent differences larger than
   `max_frame_gap`;
3. converts Tracker image coordinates to Cartesian analysis coordinates using
   `x=centroid_y` and `y=-centroid_x`;
4. smooths each segment independently;
5. uses the frame-aligned smoothed path for moving-average or triangular modes,
   but uses the sparse RDP-retained vertices for `sg_filter_rdp`;
6. calculates a signed angle at every finite interior processed vertex; and
7. marks a vertex as a turn when
   `abs(turn_angle_degrees) >= minimum_turn_angle_degrees`.

For incoming processed vector `v1` and outgoing processed vector `v2`, the
signed two-dimensional angle is:

```text
cross = v1_x * v2_y - v1_y * v2_x
dot   = v1_x * v2_x + v1_y * v2_y
turn_angle_degrees = degrees(atan2(cross, dot))
```

The result lies in `[-180, 180]`. Positive angles are counterclockwise in the
analysis coordinate system; negative angles are clockwise.

Worked angle example: processed positions `(0,0) -> (1,0) -> (1,1)` give
`v1=(1,0)` and `v2=(0,1)`. The cross product is 1 and the dot product is 0, so
the signed turn angle is `atan2(1,0)=+90 degrees`. It is a turn for a 30-degree
threshold. If two adjacent interior vertices both exceed the threshold, the
summary adds two turns.

## `02_Turn_Summary.csv`

One row summarizes one particle.

| Column | Calculation or provenance |
|---|---|
| `strain` | Label supplied when `Stats` was initialized. |
| `source_dataframe` | Resolved Tracker snapshot. |
| `particle` | Linked trajectory ID. |
| `first_frame`, `last_frame` | First and last retained frames after boundary-window removal. |
| `detection_count` | Number of retained source detections. |
| `segment_count` | Number of continuous pieces after applying `max_frame_gap`. |
| `processed_point_count` | Sum of processed path points across segments. Moving-average and triangular modes count finite smoothed points; SG/RDP mode counts sparse RDP vertices. |
| `angle_count` | Number of finite interior-vertex angles across processed segments. A segment with `m` usable processed points normally supports at most `m-2` angles. |
| `number_of_turns` | Count of finite angle rows satisfying the inclusive absolute threshold. Consecutive qualifying rows remain separate counts. |
| `mean_absolute_turn_angle_degrees` | Arithmetic mean of `absolute_turn_angle_degrees` over detected turns only; blank if none are detected. |
| `total_path_length` | Sum of raw centroid step lengths within retained segments, converted to `distance_unit`. It excludes gap-crossing steps and is not the length of the smoothed/RDP path. |
| `observed_duration_seconds` | Sum over retained segments of `(segment_last_frame - segment_first_frame) / FPS`. Excluded inter-segment gaps do not contribute. |
| `turns_per_distance` | `number_of_turns / total_path_length`; blank when path length is zero. |
| `turns_per_second` | `number_of_turns / observed_duration_seconds`; blank when observed duration is zero or FPS is unavailable. |
| `distance_unit` | Unit of `total_path_length` and denominator unit of `turns_per_distance`. |
| `minimum_turn_angle_degrees` | Requested inclusive absolute-angle threshold. |
| `smoothing_method` | Selected smoother. |
| `moving_average_window` | Requested moving-average span when selected; otherwise blank. |
| `triangular_smoothing_window` | Requested triangular span when selected; otherwise blank. |
| `sg_filter_window_length`, `sg_filter_polyorder` | Requested SG span and polynomial order when `sg_filter_rdp` is selected; otherwise blank. |
| `rdp_epsilon_pixels` | Requested RDP tolerance when `sg_filter_rdp` is selected; otherwise blank. |
| `max_frame_gap` | Largest permitted within-segment frame difference; blank represents `None`. |
| `discard_initial_frames`, `discard_final_frames` | Requested boundary-window widths. |

Worked rate example: a particle has four qualifying vertices, two retained
segments with durations 0.8 and 1.2 seconds, and 10 um total raw path length.
Then `observed_duration_seconds=2.0`, `turns_per_second=4/2=2 s^-1`, and
`turns_per_distance=4/10=0.4 turns/um`.

## `03_Turn_Angles.csv`

One row is one finite angle at an interior processed-path vertex. This is the
long-form evidence beneath the particle summary.

| Column | Calculation or provenance |
|---|---|
| `strain`, `source_dataframe`, `particle` | Dataset label, resolved source, and trajectory ID. |
| `segment_id` | One-based segment number within the particle after gap splitting. |
| `processed_vertex_index` | Zero-based index in that segment's processed path. The first possible interior vertex is index 1. In SG/RDP mode this is an RDP-path index, not a source-row index. |
| `frame` | Original source frame aligned to the processed vertex. SG/RDP vertices retain their source-frame identity. |
| `elapsed_time_seconds` | `(frame - particle_first_retained_frame) / FPS`. It is a source-frame-axis timestamp and can therefore span time omitted between segments. |
| `centroid_x_pixels`, `centroid_y_pixels` | Original unsmoothed Tracker centroid at the aligned source frame. |
| `turn_angle_degrees` | Signed processed-path angle calculated with `atan2(cross, dot)`. |
| `absolute_turn_angle_degrees` | `abs(turn_angle_degrees)`. |
| `is_turn` | True when the absolute angle is greater than or equal to the configured threshold. |
| `cumulative_distance` | Cumulative distance along the processed, not raw, path through this vertex, converted to `distance_unit`. Segment processed-path totals are joined without a gap-crossing edge. |
| `distance_unit` | Unit of `cumulative_distance`. |
| `smoothing_method` | Smoother used for this path. |
| `effective_smoothing_window` | Window actually supported by the segment. It can be smaller than requested for short segments or blank when no window applies. |

Worked cumulative-distance example: the first processed segment has step
lengths 1 and 2 um, and the next segment begins after an excluded frame gap
with a first processed step of 0.5 um. Cumulative distance reaches 3 um at the
end of segment 1 and then 3.5 um after the first step in segment 2; no distance
is added across the missing gap.

## Smoothing interpretation

- `moving_average` requires a complete centered window. Unsupported boundary
  positions are omitted from the processed turn path.
- `triangular_smoothing` uses edge-value padding. Short segments use the
  largest supported odd span, recorded in `effective_smoothing_window`.
- `sg_filter_rdp` applies Savitzky-Golay smoothing and then RDP simplification.
  The geometric-turn analysis uses the sparse retained vertices directly, so
  increasing `rdp_epsilon_pixels` can reduce both `processed_point_count` and
  `angle_count`.

Because smoothing changes the processed geometry and available vertices,
turn-count changes between methods or windows are expected and should be
reported with the configuration columns.
