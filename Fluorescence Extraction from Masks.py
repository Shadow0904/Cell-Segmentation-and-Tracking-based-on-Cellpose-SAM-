# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 10:52:12 2025

@author: chakrabarty

Extracts average per area fluorescence from tifs within masked region
Saves extracted raw fluorescence values as numpy array of shape (Channel, F, Cell ID)
"""

import numpy as np
import tifffile as tiff
import os

# Optional progress support (tqdm if available, otherwise a simple stdout bar)
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

class _SimpleBar:
    def __init__(self, total, desc=""):
        self.total = int(total) if total else 0
        self.count = 0
        self.desc = desc
        self._last_pct = -1

    def update(self, n=1):
        if self.total <= 0:
            return
        self.count += n
        pct = int(self.count * 100 / self.total)
        if pct != self._last_pct:
            self._last_pct = pct
            print(f"\r{self.desc} {pct}% ({self.count}/{self.total})", end="", flush=True)

    def close(self):
        if self.total > 0:
            print("\r" + " " * 60, end="\r")
            print(f"{self.desc} done.")

def _make_pbar(total, desc):
    if tqdm is not None:
        return tqdm(total=int(total), desc=desc, smoothing=0.1, unit="step", leave=False)
    return _SimpleBar(total=int(total), desc=desc)


def load_tiff_TCHW(path, time_axis=0, channel_axis=1):
    """
    Load a TIFF and return (T, C, H, W).
    - If the TIFF is 3D (T, H, W) and channel_axis is None, assumes single channel and adds C=1.
    - time_axis and channel_axis refer to axes in the raw TIFF array BEFORE reordering.
    """
    arr = tiff.imread(path)

    if arr.ndim == 2:
        raise ValueError("TIFF appears to be a single 2D image; expected time-lapse (>=3D).")

    if arr.ndim == 3:
        if channel_axis is not None:
            raise ValueError("3D TIFF given with channel_axis; ambiguous. Use channel_axis=None for single-channel (T,H,W).")
        # Move time to axis 0, assume shape is (T, H, W) after move
        arr = np.moveaxis(arr, time_axis, 0)
        if arr.ndim != 3:
            raise ValueError(f"After moving time axis, expected 3D array, got {arr.shape}")
        T, H, W = arr.shape
        return arr.reshape(T, 1, H, W)

    if arr.ndim == 4:
        # Move time to axis 0 first
        arr = np.moveaxis(arr, time_axis, 0)
        if channel_axis is None:
            # Try to infer channel as the smallest non-spatial axis (heuristic)
            if arr.shape[-1] <= 8:
                arr = np.moveaxis(arr, -1, 1)
            else:
                raise ValueError("4D TIFF with channel_axis=None is ambiguous. Please specify channel_axis.")
        else:
            ch_ax = channel_axis
            if channel_axis > time_axis:
                ch_ax -= 1
            arr = np.moveaxis(arr, ch_ax, 1)
        if arr.ndim != 4:
            raise ValueError(f"Expected (T,C,H,W) after axis moves, got {arr.shape}")
        return arr

    raise ValueError(f"Unsupported TIFF dimensionality: {arr.ndim}D. Expect 3D or 4D.")


def find_persistent_ids_full(st_masks, min_fraction=1.0, show_progress=False):
    """
    Compute IDs present in at least min_fraction of frames (default 100%).
    st_masks: (T, H, W) integer labels; 0 is background.
    Returns: sorted ndarray of persistent IDs (excluding 0).
    """
    if st_masks.ndim != 3:
        raise ValueError("Stitched masks must be 3D (T, H, W).")
    if st_masks.dtype.kind not in "iu":
        raise ValueError(f"Masks must be integer-labeled; got dtype {st_masks.dtype}")

    T = st_masks.shape[0]
    max_id = int(st_masks.max())
    if max_id == 0:
        return np.array([], dtype=int)

    presence = np.zeros(max_id + 1, dtype=np.int32)

    pbar = _make_pbar(T, "Scanning presence") if show_progress else None
    try:
        for t in range(T):
            ids = np.unique(st_masks[t])
            ids = ids[ids != 0]
            presence[ids] += 1
            if pbar:
                pbar.update(1)
    finally:
        if pbar:
            pbar.close()

    required = int(np.ceil(min_fraction * T))
    persistent = np.flatnonzero(presence >= required)
    return np.sort(persistent[persistent != 0])


def extract_avg_fluorescence(st_masks, img_TCHW, persistent_ids, channels_to_use, show_progress=False):
    """
    Compute average fluorescence per (channel, time, cell) only for persistent_ids.
    st_masks: (T, H, W) int labels
    img_TCHW: (T, C, H, W) intensities
    persistent_ids: 1D sorted array of cell IDs
    channels_to_use: list of channel indices to process, in desired order
    Returns: fluo (len(channels_to_use), T, N_cells) float32
    """
    Tm, Hm, Wm = st_masks.shape
    Ti, Ci, Hi, Wi = img_TCHW.shape
    if (Hm, Wm) != (Hi, Wi):
        raise ValueError(f"Mask and image sizes differ: masks {(Hm,Wm)} vs image {(Hi,Wi)}")

    # Align time by truncation to shortest
    T = min(Tm, Ti)
    st = st_masks[:T]
    img = img_TCHW[:T]

    # Validate channels
    channels_to_use = list(channels_to_use)
    if len(channels_to_use) == 0:
        raise ValueError("channels_to_use cannot be empty.")
    if min(channels_to_use) < 0 or max(channels_to_use) >= Ci:
        raise ValueError(f"channels_to_use {channels_to_use} out of bounds for {Ci} channels.")

    pids = np.asarray(persistent_ids, dtype=int)
    N = len(pids)
    out = np.empty((len(channels_to_use), T, N), dtype=np.float32)

    if N == 0:
        return out  # shape (Csel, T, 0)

    max_id = int(st.max())
    pbar = _make_pbar(T, "Extracting fluorescence") if show_progress else None

    try:
        for t in range(T):
            labels_flat = st[t].ravel()
            # Area per ID
            areas = np.bincount(labels_flat, minlength=max_id + 1).astype(np.float64)
            denom = areas[pids]
            zero_mask = denom == 0.0
            denom_safe = denom.copy()
            denom_safe[zero_mask] = 1.0  # avoid div-by-zero

            for ci, c in enumerate(channels_to_use):
                plane = img[t, c].astype(np.float64, copy=False)
                sums = np.bincount(labels_flat, weights=plane.ravel(), minlength=max_id + 1)
                vals = sums[pids] / denom_safe
                if zero_mask.any():
                    vals[zero_mask] = np.nan
                out[ci, t] = vals.astype(np.float32)

            if pbar:
                pbar.update(1)
    finally:
        if pbar:
            pbar.close()

    return out


def main(stitched_masks_npy,
         combined_tiff_path,
         channels_to_use,
         min_fraction=1.0,
         time_axis=0,
         channel_axis=1,
         save_output_path=None,
         show_progress=False):

    # Load masks
    st = np.load(stitched_masks_npy)
    if st.ndim != 3:
        raise ValueError(f"Stitched masks must be (T,H,W), got {st.shape}")
    if st.dtype.kind not in "iu":
        raise ValueError(f"Stitched masks must be integer-labeled; got dtype {st.dtype}")

    # Load TIFF as (T, C, H, W)
    img = load_tiff_TCHW(combined_tiff_path, time_axis=time_axis, channel_axis=channel_axis)

    # Time-align
    T = min(st.shape[0], img.shape[0])
    st = st[:T]
    img = img[:T]

    # Identify persistent cells
    persistent_ids = find_persistent_ids_full(st, min_fraction=min_fraction, show_progress=show_progress)

    # Extract fluorescence
    fluo = extract_avg_fluorescence(st, img, persistent_ids, channels_to_use, show_progress=show_progress)

    # Report
    print(f"Frames used: {T}")
    print(f"Channels used: {channels_to_use}  (total in TIFF: {img.shape[1]})")
    print(f"Persistent cells: {len(persistent_ids)}")
    if len(persistent_ids) > 0:
        print(f"First IDs: {persistent_ids[:min(10, len(persistent_ids))].tolist()}")

    # --- Save in same folder as stitched masks ---
    folder = os.path.dirname(os.path.abspath(stitched_masks_npy))
    base = os.path.join(folder, "fluorescence.npy")
    ids_base = os.path.join(folder, "fluorescence_cell_ids.npy")

    np.save(base, fluo)
    np.save(ids_base, persistent_ids)
    print(f"Saved fluorescence: {base}")
    print(f"Saved matching IDs: {ids_base}")

    return fluo, persistent_ids

if __name__ == "__main__":
    # Example usage (edit paths/axes/channels as needed)
    fluo, cell_ids = main(
        stitched_masks_npy='',
        combined_tiff_path='',
        channels_to_use=[0, 1, 2, 3],
        min_fraction=0.96,   # require presence in specified frame fraction
        time_axis=1,        # set according to your TIFF layout
        channel_axis=0,     # set None for single-channel 3D TIFFs
        save_output_path="fluorescence.npy",
        show_progress=True  # turn on progress bars
    )
