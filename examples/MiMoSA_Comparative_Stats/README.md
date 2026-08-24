# ComparativeStats example

`00_Comparative_Stats.ipynb` demonstrates cross-run and cross-strain analysis
with `MiMoSA.ComparativeStats`. It loads every independent run separately,
preserves dataset and cell provenance, exports validation findings, compares
turn and run-and-tumble tables, and creates fitted-speed, run-and-tumble,
translational-MSD, and angular-MSD plots.

## Required inputs

The source run exports are not distributed with this repository, so this
notebook is intentionally unexecuted. Each configured input may be a complete
tracking-run folder or its `02_Outputs` folder. The folder must contain the
statistics CSV files needed by the requested comparisons. See
[`../../docs/STATS_EXPORTED_DATA_FILES_SUMMARY.md`](../../docs/STATS_EXPORTED_DATA_FILES_SUMMARY.md)
for the export schemas and file locations.

Edit `runs_root` and `runs` in the first configuration cell. Every entry must
have this form:

```python
(run_folder, strain, unique_dataset_name)
```

Load every independent run exactly once. Runs from the same strain should use
the same `strain` value but different `unique_dataset_name` values.

## Running the notebook

1. Install the project as described in the top-level `README.md`.
2. Start Jupyter from this directory so the relative `02_Outputs` path is
   created here.
3. Replace the placeholder folders and strain labels in the configuration
   cell.
4. Execute the notebook from top to bottom and review the dataset registry and
   validation report before interpreting pooled comparisons.

From the repository root, launch it with:

```bash
cd examples/MiMoSA_Comparative_Stats
jupyter lab 00_Comparative_Stats.ipynb
```

Generated tables and plots are written under `02_Outputs/`, which is excluded
from version control. Missing legacy provenance may produce a
`not_verifiable` warning; contradictory metadata or incompatible companion
cell identities can block the affected comparison and should be corrected in
the source run exports.
