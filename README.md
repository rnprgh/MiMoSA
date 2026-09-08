# MiMoSA

**Microbial Motion, Segmentation and Analysis**

*A Python framework for microbial motion segmentation, tracking, and analysis.*

MiMoSA is based on the upstream
[RABiTPy project](https://github.com/indraneel207/RABiTPy) and preserves its
core `Capture`, `Identify`, `Tracker`, and `Stats` workflow. MiMoSA substantially
extends that foundation with feature-aware tracking, auditable filtering and
manual corrections, reloadable exports, expanded motion statistics, and
cross-run comparisons. The broader tracking and analysis scope motivates the
independent MiMoSA name; MiMoSA is not an official upstream RABiTPy release.

## Run and Tumble Analysis

The statistics workflow provides two complementary run-and-tumble analysis
methods. **Angle-based run-and-tumble analysis** is implemented as a
two-dimensional adaptation of Turner *et al.* (2016),
[*Visualizing Flagella while Tracking Bacteria*](https://doi.org/10.1016/j.bpj.2016.05.053),
in which successive changes in swimming direction are used to classify
trajectory intervals as runs or tumbles. **Velocity-based run-and-tumble
analysis** follows [Najafi *et al.* (2018)](https://doi.org/10.1126/sciadv.aar6425),
in which tumbles are identified from temporally coincident marked decreases in
translational speed and increases in angular velocity.

Both methods support three trajectory-smoothing options: a centered moving
average (`moving_average`), triangular smoothing (`triangular_smoothing`), and
Savitzky-Golay filtering followed by Ramer-Douglas-Peucker path simplification
(`sg_filter_rdp`). The moving average is the Turner-aligned default, while
triangular smoothing is the Najafi-aligned default.

Velocity-based exports name their aggregation level explicitly: `point_*`
metrics weight classified points equally, `time_weighted_*` metrics weight
same-state intervals within each particle by elapsed time, `pooled_time_weighted_*`
metrics pool interval support across particles, and
`particle_mean_time_weighted_*` metrics give each contributing particle equal
weight. `ComparativeStats` normalizes documented legacy velocity names while
loading older CSVs; rerun the producing analysis to create exports with the
canonical names. See the statistics guides under `docs/` for formulas,
uncertainty definitions, censoring rules, and worked examples.

## Python compatibility

This project has been tested only with CPython 3.10.11. No other Python
versions have been tested, and compatibility with them is not claimed. The
package metadata therefore limits installation to the Python 3.10 series.

## Repository layout

- [`src/MiMoSA/`](src/MiMoSA/): package source.
- [`examples/MiMoSA_Sample_Workflow/`](examples/MiMoSA_Sample_Workflow/):
  reviewed, partially executed example notebook, licensed
  sample video, ND2-derived frames, and calibration metadata.
- [`examples/MiMoSA_Comparative_Stats/`](examples/MiMoSA_Comparative_Stats/):
  portable, unexecuted multi-run
  comparison notebook; source run exports are not included.
- [`docs/RABITPY_LINEAGE.md`](docs/RABITPY_LINEAGE.md): relationship to RABiTPy
  and namespace migration.
- [`docs/TRACKER_COMPARISON.md`](docs/TRACKER_COMPARISON.md): basic-versus-advanced
  tracker comparison.
- [`docs/MODULE_IMPROVEMENTS.md`](docs/MODULE_IMPROVEMENTS.md): module-level
  quality-of-life improvements.
- [`docs/STATS_EXPORTED_DATA_FILES_SUMMARY.md`](docs/STATS_EXPORTED_DATA_FILES_SUMMARY.md):
  concise statistics export and calculation index.
- [`docs/STATS_BASIC_STATS_PARAMETERS.md`](docs/STATS_BASIC_STATS_PARAMETERS.md),
  [`docs/STATS_TURN_ANALYSIS_PARAMETERS.md`](docs/STATS_TURN_ANALYSIS_PARAMETERS.md),
  [`docs/STATS_ANGLE_BASED_ANALYSIS_PARAMETERS.md`](docs/STATS_ANGLE_BASED_ANALYSIS_PARAMETERS.md),
  and [`docs/STATS_VELOCITY_BASED_ANALYSIS_PARAMETERS.md`](docs/STATS_VELOCITY_BASED_ANALYSIS_PARAMETERS.md):
  detailed formulas, parameter defaults, schemas, provenance, and worked
  examples.
- [`docs/SAMPLE_MEDIA_PROVENANCE.md`](docs/SAMPLE_MEDIA_PROVENANCE.md):
  sample-media provenance, technical details, licensing, and upstream-video
  exclusion record.
- [`CHANGELOG.md`](CHANGELOG.md): release changes and legacy velocity-schema
  migration notes.
- [`licenses/`](licenses/): exact upstream license artifact retained for audit.
- [`tools/fiji/`](tools/fiji/): ND2 metadata export helper and instructions.

The published MP4 and ND2-derived TIFF frames are tracked with Git LFS.
Regenerable masks, linked tables, plots, overlay videos, and local environments
remain excluded from version control.

## Installation

### macOS with Homebrew and Miniconda

1. Install [Homebrew](https://brew.sh/) if it is not already available:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the installer instructions to add Homebrew to the shell environment,
then open a new Terminal window.

2. Use Homebrew to install Git, Git LFS, and Miniconda:

```bash
brew install git git-lfs
brew install --cask miniconda
```

3. Initialize Conda for the current shell, then close and reopen Terminal:

```bash
conda init "$(basename "${SHELL}")"
```

4. Create and activate an isolated Conda virtual environment using the only
   tested Python version:

```bash
conda create --name mimosa python=3.10.11 pip
conda activate mimosa
python --version
```

The final command should report `Python 3.10.11`.

5. Initialize Git LFS, clone the repository, and retrieve the LFS-managed
   sample video and TIFF sequence:

```bash
git lfs install
git clone https://github.com/rnprgh/MiMoSA.git
cd MiMoSA
git lfs pull
```

6. Install the package and its dependencies into the active environment:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

To open the included notebooks in JupyterLab, install it in the same active
environment:

```bash
python -m pip install jupyterlab
```

The pinned environment used for the reviewed workflow is also recorded in
`requirements.txt`. Omnipose and its machine-learning dependencies can be
large and may require platform-specific setup.

## Example notebooks

### Sample tracking workflow

The reviewed, partially executed example is
[`examples/MiMoSA_Sample_Workflow/01_MiMoSA_Sample_Walkthrough.ipynb`](examples/MiMoSA_Sample_Workflow/01_MiMoSA_Sample_Walkthrough.ipynb).
It is configured to use the included 100-frame, 16-bit TIFF sequence exported
from a Dubnau Lab ND2 recording. A 100-frame MP4 viewing copy and the ND2
calibration workbook are included alongside the frames.

The notebook retains reviewed outputs from the completed workflow, except for
the velocity-analysis cells whose saved tables and dependent plots used the
0.1.0 schema; those cells are intentionally cleared for regeneration with
MiMoSA 0.2.0. The retained outputs report an 851-frame source run, while the
bundled inputs are a 100-frame publication excerpt, so rerunning the notebook
on the distributed inputs will not reproduce the same observation counts.
External files under `02_Outputs/` are ignored because they can be regenerated
by running the notebook.

`Strain_A` is an intentional generic placeholder and should not be interpreted
as the biological identity of the supplied sample.

The original upstream `sample.avi` remains excluded because no explicit
file-specific redistribution permission was found. It is not the source of
the included Dubnau Lab sample. See
[`docs/SAMPLE_MEDIA_PROVENANCE.md`](docs/SAMPLE_MEDIA_PROVENANCE.md).

### Comparative statistics workflow

The portable comparative example is
`examples/MiMoSA_Comparative_Stats/00_Comparative_Stats.ipynb`. It shows how
to load independent run exports with stable dataset and strain identities,
review validation findings, compare turn and run-and-tumble tables, and
generate fitted-speed, run-and-tumble, translational-MSD, and angular-MSD
plots.

The comparative source datasets are not included, so the notebook is shipped
without execution output. Configure its placeholder run folders before
running it. See
[`examples/MiMoSA_Comparative_Stats/README.md`](examples/MiMoSA_Comparative_Stats/README.md)
for required inputs and usage.

## Choosing a tracker

Use `Tracker.link_particles(...)` for the original TrackPy position-only
workflow retained from RABiTPy. Use
`Tracker.link_particles_with_features(...)` when morphology, momentum,
direction, hard step-distance limits, or edge-specific memory are needed.
Calling the advanced entry point without advanced options delegates to
TrackPy-compatible linking.

See [the detailed comparison](docs/TRACKER_COMPARISON.md) before tuning the
advanced cost weights.

## Relationship to RABiTPy

MiMoSA uses a distinct Python namespace so that the expanded framework is
clearly differentiated from upstream RABiTPy and does not shadow an installed
copy of it. Existing scripts can retain their class and method calls while
changing imports:

```python
# Upstream RABiTPy
from RABiTPy import Capture, Identify, Stats, Tracker

# MiMoSA
from MiMoSA import Capture, Identify, Stats, Tracker
```

The `ComparativeStats` API is specific to the expanded MiMoSA workflow:

```python
from MiMoSA import ComparativeStats
```

The tabular, Excel, and NumPy export mechanisms remain available, but MiMoSA
0.2.0 gives velocity fields explicit canonical aggregation names.
`ComparativeStats` normalizes documented 0.1.0 velocity names in memory when it
loads older CSVs; direct column-name consumers must use the migration table in
[`CHANGELOG.md`](CHANGELOG.md), and existing files are not rewritten
automatically. Python object pickles that encode the old
`RABiTPy.<module>` namespace are not guaranteed to load as MiMoSA objects; use
tabular and metadata exports for migration. See
[`docs/RABITPY_LINEAGE.md`](docs/RABITPY_LINEAGE.md) for details.

## Provenance and licensing

MiMoSA code and documentation contributions in this repository are licensed
under the [MIT License](LICENSE), with copyright attributed to Dubnau Lab,
Rutgers University. The included video, TIFF frames, and media-derived
notebook output are licensed separately under
[CC BY 4.0](MEDIA_LICENSE.md).

The upstream RABiTPy README and Python package metadata describe that project
as MIT-licensed, but its `LICENSE` file is a license-catalog JSON object rather
than the MIT license grant and GitHub reports `NOASSERTION` for the repository
license. The upstream source-code license has since been cleared for public
redistribution. The nonstandard artifact and its history remain documented in
`THIRD_PARTY_NOTICES.md`; MiMoSA's MIT license does not erase or replace
third-party notices. This clearance does not extend to the excluded upstream
`sample.avi`, for which no file-specific permission was found.

## Acknowledgements

We extend our thanks to the developers and maintainers of RABiTPy, Omnipose,
and TrackPy. Their work provides essential foundations and functionality for
MiMoSA.
