# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 12:25:48 2025

@author: chakrabarty
"""
"""
 -Determines the best dimensions for the concatenation so that the final concatenated
  image is closest to a square
 -Concatenates tiffs and saves them as one file with dimensions (frame, channel, H, W)
 
"""

import os, math
import numpy as np
import tifffile
from glob import glob

# ---------------- Config ----------------
folder_path = "" #input image folder path here
output_folder = os.path.join(folder_path, "processed_output")
os.makedirs(output_folder, exist_ok=True)

# ---------------- Helpers ----------------
def best_grid(n):
    best = None
    limit = math.ceil(math.sqrt(n))
    for r in range(1, limit+1):
        c = math.ceil(n / r)
        diff = abs(c - r)
        area = r * c
        if best is None or (diff, area) < (best[2], best[3]):
            best = (r, c, diff, area)
    return best[0], best[1]

def ensure_4d(arr):
    """Ensure array is (C, T, H, W). If (C, H, W), add T=1 axis."""
    if arr.ndim == 3:
        return arr[:, None, :, :]  # insert time axis at position 1
    elif arr.ndim == 4:
        return arr
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")

def concat_tifs(list_tiffs, nrows, ncols):
    """Concatenate list of (C, T, H, W) arrays into a grid."""
    prestacks = []
    for row in range(nrows):
        prestacks.append(
            np.concatenate(
                [list_tiffs[i] for i in range(row*ncols, row*ncols + ncols)], axis=3  # concat width
            )
        )
    return np.concatenate(prestacks, axis=2)  # concat height

# ---------------- Main ----------------
if __name__ == '__main__':
    tiff_files = sorted(glob(os.path.join(folder_path, "*.tif")))
    if not tiff_files:
        raise FileNotFoundError("No .tif files found")

    # Read and normalize shapes
    list_tiffs = [ensure_4d(np.array(tifffile.imread(tf))) for tf in tiff_files]

    nrows, ncols = best_grid(len(list_tiffs))
    while len(list_tiffs) < nrows * ncols:
        list_tiffs.append(np.zeros_like(list_tiffs[-1]))

    combined_tiffs = concat_tifs(list_tiffs, nrows, ncols)

    out_path = os.path.join(output_folder, "combined_tiffs.tif")
    tifffile.imwrite(out_path, combined_tiffs)
    print(f"✅ Saved combined TIFF to {out_path} | shape={combined_tiffs.shape}")
