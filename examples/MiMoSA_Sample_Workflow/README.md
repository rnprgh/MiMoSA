# Example workflow

`01_MiMoSA_Sample_Walkthrough.ipynb` demonstrates capture,
segmentation, feature measurement, advanced tracking, manual track updates,
overlays, statistics, and additional analyses. It contains the reviewed
execution output from the Dubnau Lab source run. Velocity-analysis cells whose
saved results used the 0.1.0 column schema are intentionally cleared and must
be regenerated with MiMoSA 0.2.0. The remaining retained output reports 851
frames, while the bundled publication inputs contain a 100-frame excerpt;
rerunning on the distributed excerpt will therefore produce different
observation counts.

The notebook was reviewed in CPython 3.10.11 only. No other Python versions
have been tested.

The notebook's `Strain_A` value is an intentional generic placeholder, not a
statement of the supplied sample's biological identity.

The example uses these supplied inputs:

```text
01_Inputs/01_Video/01_Sample_Video.mp4
01_Inputs/02_ND2_Frames/frame_001.tif ... frame_100.tif
01_Inputs/FPS_Details.xlsx
```

The MP4 and TIFF sequence are stored using Git LFS. Run `git lfs pull` after
cloning. The notebook loads the TIFF sequence by default and uses the workbook
for the FPS and µm-per-pixel calibration. The MP4 is provided as a
viewing copy and as an optional frame source.

The sample media and media-derived notebook output are © 2026 Dubnau Lab,
Rutgers University and licensed under CC BY 4.0. See `../../MEDIA_LICENSE.md`
and `../../docs/SAMPLE_MEDIA_PROVENANCE.md`.

MiMoSA is derived from RABiTPy, but the original upstream RABiTPy sample video
is not included. External files in
`02_Outputs/` are regenerable and intentionally ignored.

Before adapting the notebook to another recording, review every manual input,
especially FPS, pixel scale, units, frame ranges, segmentation thresholds,
feature filters, tracking distances, morphology weights, and manual particle
updates. Particle IDs and tuned parameters are specific to this sample.
