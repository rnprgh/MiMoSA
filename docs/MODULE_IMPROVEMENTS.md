# MiMoSA module improvements and quality-of-life changes

This document summarizes user-facing additions in MiMoSA relative to the
upstream RABiTPy workflow. It is not an exhaustive API reference.

## Capture

- Keeps all captured frames separate from the selected working-frame range.
- Supports explicit frame-indexing and inclusive-range selection.
- Loads video, image folders, and TIFF stacks while retaining FPS and spatial
  calibration on the `Capture` object.
- Can restore acquisition properties from an exported metadata workbook or
  from manual FPS, pixel-scale, and unit values.
- Includes a Fiji helper for exporting ND2 metadata without loading pixels.

## Identify

- Keeps all masks separate from masks aligned to the working frame range.
- Supports saved-mask import and export in addition to thresholding and
  Omnipose segmentation.
- Adds Gaussian centroid refinement with parameter visualization.
- Preserves full and filtered region-property tables separately.
- Exports region data and filter settings in reusable formats.

## Tracker

- Retains the original TrackPy linker as a reproducible baseline.
- Adds morphology-, momentum-, direction-, hard-step-, and edge-aware linking.
- Separates full, filtered, and manually updated tracks instead of overwriting
  the source table.
- Distinguishes cumulative observed path length from endpoint displacement
  during filtering and records an eliminated-particle audit.
- Supports collision-review reports, particle lookup, range-aware manual
  linking/deletion/splitting, and adding filtered-out tracks back from full
  data.
- Saves and reloads linked tables, filter settings, and tracker parameters in
  CSV, Excel, pickle, and NumPy forms.
- Provides named dataframe selection for plots and videos, exact particle-ID
  selectors, empty-overlay exports, and batched overlay workflows.

## Stats

- Selects `active`, `full`, `filtered`, or `updated` tracker data explicitly
  and records the resolved source in exported tables.
- Applies Capture FPS, pixel scale, and spatial units consistently.
- Adds interval metrics, particle characteristics, trajectory-from-origin
  plots, speed-feature correlations, MSD, turns, and angle- and
  velocity-based run/tumble analyses.
- Treats `max_frame_gap` as the largest accepted adjacent frame-number
  difference and excludes larger gaps from observed motion.
- Records calibration, smoothing, trimming, state, count, and censoring
  provenance so blank or aggregated results can be audited.
- Centralizes plot styling and exports analysis tables for reuse.

## ComparativeStats

- Loads multiple runs with explicit dataset and strain identities.
- Preserves dataset, strain, and particle provenance during pooling.
- Validates units, calibration, parameter ranges, companion identity coverage,
  and compatible schemas before comparison.
- Produces machine-readable validation reports and cross-run comparison tables
  and figures for turns, run/tumble behavior, speeds, and MSD.

## Utility

- Combines compatible linked CSV files into a `Tracker` object for downstream
  workflows while retaining the class-level interface derived from RABiTPy.
