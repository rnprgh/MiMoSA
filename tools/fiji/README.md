# Export ND2 metadata in Fiji

`Export_ND2_Metadata.groovy` reads acquisition metadata directly from an ND2
file with Fiji's bundled Bio-Formats reader. It writes `FPS_Details.xlsx` beside
the ND2 file without loading the image pixels.

## Run it

1. Open Fiji and choose **File > New > Script**.
2. Set **Language** to **Groovy**.
3. Open `Export_ND2_Metadata.groovy` and click **Run**.
4. Choose the ND2 file. For the usual single-series ND2 files, leave
   **Series** set to `1` and **Output filename** set to `FPS_Details.xlsx`.
5. Enable **Overwrite existing workbook** only when the old workbook should be
   replaced.

The workbook contains Resolution, Frames, Pixel Scale Factor, Scale Units,
Timestamp #First, Timestamp #Last, elapsed time, and FPS. Pixel size is
normalized to micrometres (`um`), and timestamps are normalized to seconds.

FPS intentionally uses the same calculation as the project's existing manual
workbooks:

```text
FPS = Frames / (Timestamp #Last - Timestamp #First)
```

The workbook can be loaded directly with
`Capture.load_capture_properties_from_file(...)`.
