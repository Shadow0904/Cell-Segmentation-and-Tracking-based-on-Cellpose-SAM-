# Cell-Segmentation-and-Tracking-based-on-Cellpose-SAM-
A cellpose SAM based pipeline to segment, extract and analyze multiplexed fluorescence time series data from live-cell imaging

Run the scripts in this order:

1. Image Concatenation: builds mosaics from single-cell frames
2. Cellpose-SAM Parameter Test: check if your segmentation looks good
3. Segmentation (Multi-GPU): run full segmentation
4. Tracking & Stitching: link cells across frames
5. Fluorescence Extraction: get per-cell intensity values

To address a few pecularities:
Cellpose-SAM works better on multi-cell images, so single-cell frames are arranged into a near-square mosaic before segmentation. Skip this step if you don't need it.
Segmentation runs cpsam with some light preprocessing (scaling + Gaussian blur), which you can fine tune to get the best results for your data. The pre-processing is only for the masking step. Masks are mapped back to original frames for fluorescence extraction.
Tracking uses a Kalman filter + Hungarian matching to keep cell identities consistent over time.

Output:
Masks (stitched + unstitched)
Cell trajectories
Fluorescence arrays: (Channel, Frame, Cell ID)

Notes:
Pick the channel with the best signal to noise ratio for segmentation
