# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 13:23:14 2025

@author: chakrabarty
"""

#!/usr/bin/env python3
"""
Segment a concatenated/combined TIFF with Cellpose SAM across multiple GPUs
and stitch masks over time. Saves both stitched and unstitched maskes separately.

Expected input shapes:
- [T, C, H, W]    -> choose a channel via SEGMENTATION_CHANNEL (0-based)
- [T, H, W]       -> channel selection is ignored
"""

import os
import math
from glob import glob

import cv2
import numpy as np
import tifffile
import torch
import multiprocessing as mp

from cellpose import models, transforms, utils


# User settings (edit here, I recommend using the settings that worked best for Parameter Test) 

# Path to combined_tiffs.tif (shape [T, C, H, W] or [T, H, W])
INPUT_TIFF = r""

# Optional: where to save outputs. If None, saves next to input in "processed_output"
OUTPUT_FOLDER = None

# Channel index to segment when TIFF is [T, C, H, W] (0-based)
SEGMENTATION_CHANNEL = 1

# Resize so the longer side equals this many pixels before segmentation (e.g., 500)
RESIZE_FACTOR = 1000

# Threshold for stitching labels across time
STITCH_THRESHOLD = 0.25

# Force CPU even if CUDA is available
FORCE_CPU = False

# Cellpose SAM model identifier
PRETRAINED_MODEL = "cpsam"


# ---------------- Helper functions ----------------

def preprocess_frame(frame): # Use same as the Parameter Test
    # Scale and blur to stabilize segmentation
    frame = 150 * frame.astype(np.float32)
    frame = cv2.GaussianBlur(frame, (5, 5), sigmaX=1)
    frame = np.clip(frame, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    return frame


def image_resize(frame, resize_factor):
    # Keep aspect ratio; set the longer side to resize_factor
    H, W = frame.shape[:2]
    if W >= H:
        new_W = resize_factor
        new_H = int(round(H * (resize_factor / float(W))))
    else:
        new_H = resize_factor
        new_W = int(round(W * (resize_factor / float(H))))
    new_H = max(1, new_H)
    new_W = max(1, new_W)
    return transforms.resize_image(frame, Ly=new_H, Lx=new_W)


def mask_resize(mask, H, W):
    # Resize label mask back to the original frame size
    return transforms.resize_image(mask, Ly=int(H), Lx=int(W))


def _segment_single_frame(model, frame, resize_factor):
    pre = preprocess_frame(frame)
    H, W = pre.shape
    pre_small = image_resize(pre, resize_factor)
    # For single-channel images, channels=[0, 0] is appropriate
    masks, _, _ = model.eval([pre_small], channels=[0, 0], do_3D=False, batch_size=1)
    mask = masks[0]
    mask_full = mask_resize(mask, H, W)
    return mask_full


def segment_worker(gpu_idx, frame_indices, seg_stack, resize_factor, use_gpu, return_dict):
    device = f"cuda:{gpu_idx}" if use_gpu else "cpu"
    model = models.CellposeModel(
        gpu=use_gpu,
        device=torch.device(device),
        pretrained_model=PRETRAINED_MODEL,
    )

    results = []
    for i in frame_indices:
        mask = _segment_single_frame(model, seg_stack[i], resize_factor)
        results.append((int(i), mask.astype(np.int32)))  # label masks are integers

    return_dict[gpu_idx] = results


# ---------------- Main execution ----------------

def main():
    if not INPUT_TIFF or not os.path.exists(INPUT_TIFF):
        raise FileNotFoundError(f"INPUT_TIFF not found: {INPUT_TIFF}")

    # Resolve output folder
    if OUTPUT_FOLDER is None:
        parent = os.path.dirname(os.path.abspath(INPUT_TIFF))
        output_folder = os.path.join(parent, "processed_output")
    else:
        output_folder = OUTPUT_FOLDER
    os.makedirs(output_folder, exist_ok=True)

    # Device configuration
    cuda_available = torch.cuda.is_available()
    use_gpu = (not FORCE_CPU) and cuda_available
    gpu_count = torch.cuda.device_count() if use_gpu else 0
    print(f"Available GPUs: {gpu_count} | Using GPU: {use_gpu}")

    # Load combined TIFF
    print(f"Loading combined TIFF: {INPUT_TIFF}")
    combined = tifffile.imread(INPUT_TIFF)
    combined = np.asarray(combined)

    # Validate shape and select channel
    if combined.ndim == 4:
        # Expect [T, C, H, W]
        num_frames, num_channels, H, W = combined.shape
        if not (0 <= SEGMENTATION_CHANNEL < num_channels):
            raise ValueError(
                f"SEGMENTATION_CHANNEL={SEGMENTATION_CHANNEL} out of range for C={num_channels}"
            )
        seg_stack = combined[:, SEGMENTATION_CHANNEL, :, :]
    elif combined.ndim == 3:
        # Assume [T, H, W]
        num_frames, H, W = combined.shape
        seg_stack = combined
    else:
        raise ValueError(f"Unsupported TIFF shape {combined.shape}. Expected [T,C,H,W] or [T,H,W].")

    print(f"Data shape — frames: {num_frames}, height: {H}, width: {W}")

    # Build frame index chunks
    frame_indices = np.arange(num_frames)
    worker_count = max(gpu_count, 1)
    chunks = [c for c in np.array_split(frame_indices, worker_count) if len(c) > 0]

    # Multiprocessing setup
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    return_dict = manager.dict()
    jobs = []

    print(f"Spawning {len(chunks)} worker(s)...")
    for gpu_idx, chunk in enumerate(chunks):
        p = mp.Process(
            target=segment_worker,
            args=(gpu_idx, chunk, seg_stack, RESIZE_FACTOR, use_gpu, return_dict)
        )
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    # Collect masks in frame order
    all_masks = [None] * num_frames
    for results in return_dict.values():
        for idx, mask in results:
            all_masks[idx] = mask

    # Sanity check for missing results
    missing = [i for i, m in enumerate(all_masks) if m is None]
    if missing:
        raise RuntimeError(f"Missing masks for frames: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    # Convert to array stack [T, H, W]
    all_masks = np.asarray(all_masks, dtype=np.int32)

    # Stitch masks across time for consistent labels
    print("Stitching masks across time...")
    stitched_masks = utils.stitch3D(all_masks, stitch_threshold=STITCH_THRESHOLD)

    # Save outputs
    base = os.path.splitext(os.path.basename(INPUT_TIFF))[0]
    per_frame_npy = os.path.join(output_folder, f"{base}_per_frame_masks.npy")
    per_frame_tif = os.path.join(output_folder, f"{base}_per_frame_masks.tif")
    stitched_npy = os.path.join(output_folder, f"{base}_stitched_masks.npy")
    stitched_tif = os.path.join(output_folder, f"{base}_stitched_masks.tif")

    np.save(per_frame_npy, all_masks)
    tifffile.imwrite(per_frame_tif, all_masks.astype(np.int32), photometric="minisblack")

    np.save(stitched_npy, stitched_masks)
    tifffile.imwrite(stitched_tif, stitched_masks.astype(np.int32), photometric="minisblack")

    print(f"Saved per-frame masks: {per_frame_npy}")
    print(f"Saved per-frame masks (TIFF): {per_frame_tif}")
    print(f"Saved stitched masks: {stitched_npy}")
    print(f"Saved stitched masks (TIFF): {stitched_tif}")
    print("Done.")


if __name__ == "__main__":
    main()
