# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 10:55:15 2025

@author: chakrabarty
"""

"""
Single-frame segmentation with Cellpose-SAM:
- Lazily loads one frame/channel from a multi-channel TIFF via zarr
- Preprocesses (scale + blur) and resizes for segmentation
- Runs Cellpose 'cpsam' on the resized image
- Resizes mask back to original and saves a side-by-side overlay with the preprocessed frame

Note:
- TIFF is assumed shaped as (T, C, Y, X). Adjust indices if your order differs.
- 'cpsam' requires a recent Cellpose version that includes the SAM-backed model.
"""

import os
import numpy as np
import tifffile
import zarr
import cv2
import torch
import matplotlib.pyplot as plt
from cellpose import models, transforms, utils
import inspect


# ----------------------------
# Configuration (edit these)
# ----------------------------
tiff_path = r'' # enter tif path for concatenated tif
frame_index = 200        # 0-based index
channel_index = 1          # 0-based index
resize_factor = 1000    # longest side in pixels for segmentation
overlay_output = r""

# Cellpose / device settings
use_gpu = torch.cuda.is_available()
device = "cuda" if use_gpu else "cpu"
cellprob_threshold = 0.0   # adjust as needed
flow_threshold = 0.4       # adjust as needed
min_size = 15              # remove tiny masks (pixels); adjust for your data


# ----------------------------
# I/O and preprocessing
# ----------------------------
def load_single_frame_lazy(tiff_path: str, frame_idx: int, channel_idx: int) -> np.ndarray:
    """
    Lazily opens TIFF as zarr and returns one 2D frame (Y, X) as uint16.
    Assumes array shape (T, C, Y, X). Prints detected shape.
    """
    with tifffile.TiffFile(tiff_path) as tif:
        store = tif.series[0].aszarr()
        zarr_array = zarr.open(store, mode="r")
        print("TIFF shape:", zarr_array.shape)

        # Safety checks
        if len(zarr_array.shape) < 4:
            raise ValueError(f"Expected (T, C, Y, X), got {zarr_array.shape}")

        T, C, H, W = zarr_array.shape[:4]
        if not (0 <= frame_idx < T):
            raise IndexError(f"frame_index {frame_idx} out of range 0..{T-1}")
        if not (0 <= channel_idx < C):
            raise IndexError(f"channel_index {channel_idx} out of range 0..{C-1}")

        frame = zarr_array[frame_idx, channel_idx, :, :]
        frame_np = np.array(frame, dtype=np.uint16)

    print(f"Loaded frame {frame_idx}, channel {channel_idx}: shape={frame_np.shape}, dtype={frame_np.dtype}")
    return frame_np


def preprocess_frame(frame: np.ndarray) -> np.ndarray: # preprocess has been optimized for my images, feel free to play around
    """
    Scale intensity then Gaussian blur; returns uint16.
    """
    frame = 150 * frame.astype(np.float32)
    frame = cv2.GaussianBlur(frame, (5, 5), sigmaX=1)
    frame = np.clip(frame, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    return frame


def compute_resize_dims(H: int, W: int, target_long: int) -> tuple[int, int]:
    """
    Keeps aspect ratio; longest side becomes target_long.
    """
    if W >= H:
        new_W = target_long
        new_H = int(round(H * (target_long / float(W))))
    else:
        new_H = target_long
        new_W = int(round(W * (target_long / float(H))))
    new_H = max(1, new_H)
    new_W = max(1, new_W)
    return new_H, new_W


def resize_image_cellpose(img: np.ndarray, new_H: int, new_W: int) -> np.ndarray:
    """
    Uses cellpose.transforms.resize_image with Ly/Lx semantics.
    """
    return transforms.resize_image(img, Ly=new_H, Lx=new_W)


# ----------------------------
# Segmentation and postprocess
# ----------------------------
import inspect

def run_cellpose_cpsam(img_float01: np.ndarray, model: models.CellposeModel) -> np.ndarray:
    """
    Runs Cellpose on a single 2D gray image in [0,1], returns mask (int labels).
    Adapts to different Cellpose versions: supports 3- or 4-value returns and
    only passes eval() kwargs supported by the installed version.
    """
    # Common args
    base_args = {
        "channels": [0, 0],            # grayscale
        "cellprob_threshold": cellprob_threshold,
        "flow_threshold": flow_threshold,
        "min_size": min_size,
        "normalize": False,            # already normalized
        "augment": False,
        "progress": False,
    }

    # Only pass kwargs that exist in this install
    eval_sig = inspect.signature(model.eval).parameters
    call_args = dict(base_args)
    if "channel_axis" in eval_sig:
        call_args["channel_axis"] = None
    if "tile" in eval_sig:
        call_args["tile"] = False
    if "net_avg" in eval_sig:
        call_args["net_avg"] = False

    # Eval
    result = model.eval([img_float01], **call_args)

    # Handle 3- vs 4-item returns
    if not isinstance(result, tuple):
        raise RuntimeError(f"Unexpected return type from Cellpose eval: {type(result)}")
    if len(result) == 4:
        masks_list, flows, styles, diams = result
    elif len(result) == 3:
        masks_list, flows, styles = result
        diams = None
    else:
        raise RuntimeError(f"Unexpected number of return values from Cellpose eval: {len(result)}")

    # Extract first image’s mask
    masks = masks_list[0] if isinstance(masks_list, (list, tuple)) else masks_list
    return masks

def mask_resize_nearest(mask: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Resizes labeled mask back to (H, W) using nearest-neighbor to preserve integer labels.
    """
    return cv2.resize(mask.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)


# ----------------------------
# Visualization
# ----------------------------
def show_side_by_side_overlay_filled(preproc_frame_u16: np.ndarray, mask: np.ndarray,
                                     color=(1.0, 0.0, 0.0), alpha=0.3) -> None:
    """
    Left: preprocessed frame (grayscale).
    Right: same image with transparent color fill over each mask region.
    
    color: (R, G, B) in 0-1 float range
    alpha: transparency, 0=fully transparent, 1=opaque
    """
    img_gray = preproc_frame_u16.astype(np.float32)
    img_gray /= (img_gray.max() + 1e-8)

    # RGB version of grayscale
    overlay = np.dstack([img_gray]*3)

    # Boolean mask for regions (mask > 0)
    mask_bool = mask.astype(bool)

    # Fill masked area with chosen color, using alpha blending
    for c in range(3):
        overlay[..., c] = np.where(
            mask_bool,
            (1 - alpha) * overlay[..., c] + alpha * color[c],
            overlay[..., c]
        )

    plt.figure(figsize=(12, 6))
    # Left: grayscale
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(img_gray, cmap="gray")
    ax1.set_title("Preprocessed frame")
    ax1.axis("off")

    # Right: transparent overlay fill
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(overlay)
    ax2.set_title("Mask fill overlay")
    ax2.axis("off")

    plt.tight_layout()
    plt.show()

# ----------------------------
# Main
# ----------------------------
def main():
    # 1) Load single frame/channel
    raw_frame = load_single_frame_lazy(tiff_path, frame_index, channel_index)
    H, W = raw_frame.shape

    # 2) Preprocess
    preproc = preprocess_frame(raw_frame)

    # 3) Resize for segmentation
    new_H, new_W = compute_resize_dims(H, W, resize_factor)
    preproc_resized = resize_image_cellpose(preproc, new_H, new_W)

    # Convert to float [0,1] for model
    img_float01 = preproc_resized.astype(np.float32)
    img_float01 = img_float01 / (img_float01.max() + 1e-8)

    # 4) Create Cellpose-SAM model and segment
    print(f"Using device: {device} (gpu={use_gpu})")
    model = models.CellposeModel(gpu=use_gpu, device=torch.device(device), pretrained_model="cpsam")
    mask_resized = run_cellpose_cpsam(img_float01, model)

    # 5) Resize mask back to original size
    mask_fullres = mask_resize_nearest(mask_resized, H, W)

    # 6) Save side-by-side overlay
    show_side_by_side_overlay_filled(preproc, mask_fullres, color=(0.0, 1.0, 0.0), alpha=0.4)

if __name__ == "__main__":
    main()
