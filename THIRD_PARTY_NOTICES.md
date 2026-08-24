# Third-party notices and provenance

## Upstream RABiTPy

MiMoSA contains extensively modified code derived from:

- Project: RABiTPy
- Repository: https://github.com/indraneel207/RABiTPy
- PyPI: https://pypi.org/project/RABiTPy/
- Authors listed upstream: Indraneel Vairagare, Samyabrata Sen, and Abhishek
  Shrivastava
- Article: Sen et al., *BMC Bioinformatics* 26, 127 (2025),
  https://doi.org/10.1186/s12859-025-06145-w

The upstream README and package metadata describe RABiTPy as MIT-licensed.
Note however that the upstream `LICENSE` file is a JSON description of the MIT license,
not the standard MIT license grant with a copyright notice. GitHub therefore
reports the repository license as `NOASSERTION`. The exact license file shipped
with the installed RABiTPy 1.2.0 package is still preserved at
`licenses/UPSTREAM_RABITPY_LICENSE_AS_PUBLISHED.json`.

The upstream source-code license has been cleared for public redistribution.
MiMoSA is an independent expanded framework and is not an official upstream
RABiTPy release.

## Upstream sample video

The original `Additional examples/sample.avi` is not included. The local
working copy was byte-for-byte equivalent to upstream Git blob
`9bf13ee6a0a0ae5e5f41141d7a931d17af260055`, but no file-specific license,
permission, or attribution record was found. See
`docs/SAMPLE_MEDIA_PROVENANCE.md`.

The replacement MP4, ND2-derived TIFF frames, and media-derived notebook
output are original materials supplied by Dubnau Lab, Rutgers University; they
are not upstream RABiTPy assets. They are licensed under CC BY 4.0 as described
in `MEDIA_LICENSE.md`.

## Dependencies

MiMoSA uses third-party libraries including NumPy, pandas, SciPy,
Matplotlib, seaborn, scikit-image, TrackPy, OpenCV, Omnipose, Cellpose Omni,
distfit, tifffile, and tqdm. Their licenses are not reproduced here. Dependency
versions are listed in `requirements.txt`; review each dependency's license
before redistributing bundled binaries, models, or source.
