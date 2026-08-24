# Sample media provenance and license

Updated: 2026-08-21

## Included Dubnau Lab sample

Dubnau Lab, Rutgers University supplied and owns the publication sample. The
TIFF sequence was exported from the laboratory's ND2 recording with
Fiji/ImageJ. The source ND2 file itself is not distributed.

Included inputs:

- `01_Inputs/01_Video/01_Sample_Video.mp4`
  - Size: 33,291,750 bytes
  - SHA-256:
    `5c3582f5e8b7a1f41fbffb73957311b77950cb129718543fec2db5c79ac535cf`
  - Resolution: 2044 × 2048 pixels
  - Frames: 100
  - Frame rate: 28.279 FPS
- `01_Inputs/02_ND2_Frames/frame_001.tif` through `frame_100.tif`
  - Count: 100
  - Total size: 837,420,238 bytes
  - Per-file size range: 8,372,668–8,374,220 bytes
  - Frame shape: 2048 × 2044 pixels
  - Pixel type: unsigned 16-bit
- `01_Inputs/FPS_Details.xlsx`
  - Size: 3,834 bytes
  - SHA-256:
    `5175fabbd542863cacfe4803b6287846f370ca31e548fb9a65e55e41d6d0dc41`

The workbook was re-exported without Excel's machine-specific absolute-path
metadata; its displayed values, formulas, and formatting are unchanged.

The retained notebook output reports an 851-frame source run at approximately
28.279 FPS with a spatial scale of 0.1625 µm per pixel and `um` as its stored
unit string. The distributed TIFF sequence and workbook contain a 100-frame
publication excerpt, so the embedded output is not a direct execution on only
the bundled frames. The organism/strain is intentionally represented by the
generic `Strain_A` placeholder. The acquisition date is not specified in this
repository.

Individual file hashes are recorded in
`examples/MiMoSA_Sample_Workflow/01_Inputs/SHA256SUMS`.

## License and attribution

The included MP4, TIFF frames, and media-derived output embedded in the
executed notebook are:

> © 2026 Dubnau Lab, Rutgers University. Licensed under CC BY 4.0.

Reuse requires attribution, a link to
<https://creativecommons.org/licenses/by/4.0/>, and an indication of whether
changes were made. The preferred attribution is recorded in
`MEDIA_LICENSE.md`.

## Storage

The MP4 and TIFF files are tracked with Git LFS. The 101 LFS objects contain
approximately 871 MB of binary data. `FPS_Details.xlsx` and the executed
notebook are stored directly in Git. Regenerable files under `02_Outputs/`
remain ignored.

## Excluded upstream sample

The upstream RABiTPy `Additional examples/sample.avi` remains excluded. The
previous working copy was identical to the upstream file:

- Size: 17,439,420 bytes
- SHA-256:
  `40c1c870bbd0f77da856b54e281c0be706e9630fc11fb9359c54d3a403fc9f19`
- Git blob SHA: `9bf13ee6a0a0ae5e5f41141d7a931d17af260055`

The upstream README and package metadata describe RABiTPy generally as MIT,
but neither the repository nor the associated article provides explicit
file-specific redistribution permission for `sample.avi`. The upstream
license artifact is JSON metadata rather than the standard MIT grant, and
GitHub reports `NOASSERTION`. The upstream source-code license has been cleared
separately, but that clearance does not provide file-specific permission for
the AVI. The article's CC BY-NC-ND 4.0 license is not treated as permission for
the separately hosted AVI.

Sources reviewed:

- <https://github.com/indraneel207/RABiTPy>
- <https://github.com/indraneel207/RABiTPy/tree/main/Additional%20examples>
- <https://github.com/indraneel207/RABiTPy/blob/main/LICENSE>
- <https://doi.org/10.1186/s12859-025-06145-w>
