# MiMoSA basic and advanced tracker comparison

Both workflows start with the same region-properties dataframe and return rows
with particle identifiers. They differ in how detections are assigned across
frames.

| Capability | Basic tracker | Advanced tracker |
|---|---|---|
| Entry point | `Tracker.link_particles(...)` | `Tracker.link_particles_with_features(...)` |
| Assignment engine | `trackpy.link_df` | Hungarian assignment over a custom cost matrix when advanced options are enabled |
| Position input | Two centroid columns | Two centroid columns |
| Distance control | TrackPy `search_range` | Predicted-position distance plus optional hard last-step distance |
| Missing detections | One global TrackPy memory | Global memory with optional shorter memory near image edges |
| Morphology | Not used | Optional area, axis lengths, or other numeric feature columns with per-feature weights |
| Recent motion | Managed by TrackPy | Optional momentum prediction and direction-change cost |
| Feature stability | Not used | Optional past-track and future-candidate feature averaging windows |
| Assignment audit | Basic settings metadata | Advanced settings, feature weights, windows, and edge parameters in export metadata |
| Compatibility mode | Native behavior | With no advanced option enabled, delegates to TrackPy and records `advanced_trackpy_compatible` |

## When to use the basic tracker

Use the basic tracker when detections are sparse, frame-to-frame displacement
is modest, and spatial proximity alone separates candidates reliably. It has
fewer parameters, is easier to tune, and provides the clearest baseline for
comparison with the upstream RABiTPy workflow.

```python
linked_basic = tracker_basic.link_particles(
    max_distance=20,
    max_memory=30,
    position_columns=["centroid_x", "centroid_y"],
)
```

## When to use the advanced tracker

Use the advanced tracker when nearby detections cross, morphology helps retain
identity, motion direction is informative, long jumps must be rejected, or
edge exits cause stale identifiers to be reused.

```python
linked_advanced = tracker_advanced.link_particles_with_features(
    max_distance=20,
    max_memory=30,
    max_step_distance=20,
    position_columns=["centroid_x", "centroid_y"],
    feature_columns=["area", "major_axis_length", "minor_axis_length"],
    feature_weights={
        "area": 0.0,
        "major_axis_length": 50.0,
        "minor_axis_length": 1.0,
    },
    feature_averaging_window=5,
    future_feature_averaging_window=5,
    momentum_weight=2.0,
    direction_weight=2.0,
    momentum_window=3,
    edge_max_memory=0,
    edge_margin=15.0,
)
```

These values are examples from one recording, not universal defaults. Retune
them for the frame rate, spatial calibration, density, cell morphology, and
motion regime of each dataset.

## Interpreting advanced costs

- `max_distance` gates candidate distance from the predicted position.
- `max_step_distance` rejects a candidate whose distance from the last
  observed centroid is too large, even if prediction places it nearby.
- Feature costs compare numeric candidate morphology with recent track means.
  A weight of zero disables that feature's contribution.
- Momentum cost favors candidates near the predicted position.
- Direction cost penalizes abrupt changes relative to recent travel.
- Future feature averaging changes relative candidate cost; it is not a hard
  morphology threshold and it never bridges a missing frame during lookahead.
- `edge_max_memory` can expire tracks near the field boundary sooner than
  interior tracks, reducing accidental identifier inheritance after exits.

## Shared downstream improvements

Both linked results can use the same filtering, manual relabeling, export,
reload, plotting, collision review, video overlay, and statistics APIs. The
full, filtered, and manually updated tables remain separate so that corrections
do not destroy the original links.
