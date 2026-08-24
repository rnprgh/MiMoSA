# Relationship to RABiTPy

MiMoSA (Microbial Motion, Segmentation and Analysis) is based on the upstream
[RABiTPy project](https://github.com/indraneel207/RABiTPy). It preserves the
core object flow from image capture through identification, tracking, and
statistics, while substantially expanding the available tracking, correction,
provenance, analysis, and cross-run comparison workflows.

MiMoSA is an independent expanded framework, not an official upstream
RABiTPy release. Upstream authorship, licensing information, and the associated
publication are recorded in [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## Retained foundation

- The `Capture`, `Identify`, `Tracker`, and `Stats` class organization is
  derived from RABiTPy.
- `Tracker.link_particles(...)` retains the TrackPy position-only workflow as
  a reproducible baseline.
- Coordinate conventions and compatible tabular export structures are retained
  where required for continuity and reproducibility.

## MiMoSA extensions

- Feature-, momentum-, direction-, hard-step-, and edge-aware tracking.
- Separate full, filtered, manually corrected, and eliminated-particle data.
- Reloadable calibration, filter, tracking, and analysis provenance.
- Expanded trajectory, speed, morphology, MSD, turning, and run-and-tumble
  analyses.
- Cross-run and cross-strain validation, aggregation, and visualization through
  `ComparativeStats`.

See [`TRACKER_COMPARISON.md`](TRACKER_COMPARISON.md) and
[`MODULE_IMPROVEMENTS.md`](MODULE_IMPROVEMENTS.md) for implementation-oriented
comparisons.

## Python namespace migration

MiMoSA deliberately uses its own package namespace and does not install a
`RABiTPy` compatibility shim. A shim could shadow the upstream package when
both projects are installed in one environment and would make provenance less
clear.

Update imports as follows:

```python
# Before
from RABiTPy import Capture, Identify, Stats, Tracker
from RABiTPy.constants import AvailableOperations, AvailableProps, PropsThreshold

# MiMoSA
from MiMoSA import Capture, Identify, Stats, Tracker
from MiMoSA.constants import AvailableOperations, AvailableProps, PropsThreshold
```

The public class and method names remain stable unless a MiMoSA-specific API is
being used. `ComparativeStats`, for example, is imported directly from MiMoSA:

```python
from MiMoSA import ComparativeStats
```

Tabular exports and metadata files remain the preferred migration surface.
Arbitrary Python object pickles can encode the original module namespace and
are therefore not guaranteed to unpickle under the renamed package.
