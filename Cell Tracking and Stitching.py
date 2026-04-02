# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 10:59:37 2025

@author: chakrabarty
"""
"""
Cell 1: Stitches per frame unstitched masks based on centroid tracking
Cell 2: Visualizes mask centroid trajectories with overlay of a chosen frame from time series
""" 

import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import os
import numpy as np

# =========================
# Geometry and mask helpers
# =========================

def iou_from_bboxes(b1, b2):
    """
    Compute IoU between two bounding boxes.
    Each box is (x1, y1, x2, y2) with inclusive integer coords.
    """
    x1, y1, x2, y2 = map(float, b1)
    a1, b1_, a2, b2_ = map(float, b2)

    # Intersection coords
    ix1 = max(x1, a1)
    iy1 = max(y1, b1_)
    ix2 = min(x2, a2)
    iy2 = min(y2, b2_)

    iw = max(0.0, ix2 - ix1 + 1.0)
    ih = max(0.0, iy2 - iy1 + 1.0)
    inter = iw * ih

    # Areas
    area1 = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    area2 = (a2 - a1 + 1.0) * (b2_ - b1_ + 1.0)

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def extract_props(mask_2d: np.ndarray) -> Dict[int, dict]:
    """
    Returns dict: local_id -> {'centroid': (x, y), 'area': int, 'bbox': (x1,y1,x2,y2)}
    Uses integer labels in mask_2d; background 0 is ignored.
    """
    props = {}
    ids = np.unique(mask_2d)
    ids = ids[ids != 0]
    for lid in ids:
        ys, xs = np.where(mask_2d == lid)
        if xs.size == 0:
            continue
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        cx, cy = xs.mean(), ys.mean()
        props[int(lid)] = {
            'centroid': (float(cx), float(cy)),     # (x, y)
            'area': int(xs.size),
            'bbox': (int(x1), int(y1), int(x2), int(y2))
        }
    return props

def bbox_union(b1: Tuple[int,int,int,int], b2: Tuple[int,int,int,int]) -> Tuple[int,int,int,int]:
    x1, y1, x2, y2 = b1
    a1, b1_, a2, b2_ = b2
    return (min(x1, a1), min(y1, b1_), max(x2, a2), max(y2, b2_))

def iou_local(frame_prev: np.ndarray, id_prev: int, bbox_prev: Tuple[int,int,int,int],
              frame_curr: np.ndarray, id_curr: int, bbox_curr: Tuple[int,int,int,int]) -> float:
    """
    Compute IoU between two instances via local crops to the union bbox (fast, no skimage).
    """
    ux1, uy1, ux2, uy2 = bbox_union(bbox_prev, bbox_curr)

    # Prev crop
    px1, py1, px2, py2 = bbox_prev
    p_crop = frame_prev[uy1:uy2+1, ux1:ux2+1]  # union window
    # Build boolean mask: True where equals id_prev AND within its bbox region
    p_mask = np.zeros_like(p_crop, dtype=bool)
    p_slice = (slice(py1-uy1, py2-uy2+uy2-uy1+1), slice(px1-ux1, px2-ux2+ux2-ux1+1))
    # safer slice calc:
    p_mask_y0 = py1 - uy1; p_mask_y1 = py2 - uy2 + (uy2 - uy1) + 1
    p_mask_x0 = px1 - ux1; p_mask_x1 = px2 - ux2 + (ux2 - ux1) + 1
    p_mask[p_mask_y0:p_mask_y1, p_mask_x0:p_mask_x1] = (frame_prev[py1:py2+1, px1:px2+1] == id_prev)

    # Curr crop
    cx1, cy1, cx2, cy2 = bbox_curr
    c_mask = np.zeros_like(p_crop, dtype=bool)
    c_mask_y0 = cy1 - uy1; c_mask_y1 = cy2 - uy2 + (uy2 - uy1) + 1
    c_mask_x0 = cx1 - ux1; c_mask_x1 = cx2 - ux2 + (ux2 - ux1) + 1
    c_mask[c_mask_y0:c_mask_y1, c_mask_x0:c_mask_x1] = (frame_curr[cy1:cy2+1, cx1:cx2+1] == id_curr)

    inter = np.logical_and(p_mask, c_mask).sum()
    union = np.logical_or(p_mask, c_mask).sum()
    return float(inter) / float(union) if union > 0 else 0.0

def euclidean(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    ax, ay = a; bx, by = b
    return float(np.hypot(ax - bx, ay - by))

# =========================
# Kalman filter (CV model)
# =========================

class KalmanFilterCV:
    """
    Constant-velocity Kalman filter on (x, y, vx, vy).
    """
    def __init__(self, dt=1.0, process_var=1.0, meas_var=25.0):
        self.dt = dt
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1,  0],
                           [0, 0, 0,  1]], dtype=float)
        q = process_var
        dt2 = dt*dt
        dt3 = dt2*dt/2.0
        dt4 = dt2*dt2/4.0
        self.Q = np.array([[dt4,   0,  dt3,   0],
                           [  0, dt4,    0, dt3],
                           [ dt3,  0,  dt2,   0],
                           [  0, dt3,    0, dt2]], dtype=float) * q
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)
        r = meas_var
        self.R = np.eye(2) * r
        self.I = np.eye(4)
        self.x = None
        self.P = None

    def initiate(self, x, y, init_var_pos=100.0, init_var_vel=1000.0):
        self.x = np.array([x, y, 0.0, 0.0], dtype=float)
        self.P = np.diag([init_var_pos, init_var_pos, init_var_vel, init_var_vel])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z_xy: Tuple[float,float]):
        z = np.asarray(z_xy, dtype=float)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P

    def state_xy(self) -> Tuple[float,float]:
        return float(self.x[0]), float(self.x[1])

# =========================
# Track and Tracker classes
# =========================

class Track:
    min_hits_global = 3  # set by Tracker

    def __init__(self, track_id: int, x: float, y: float, bbox, frame_idx: int, dt=1.0):
        self.id = track_id
        self.kf = KalmanFilterCV(dt=dt)
        self.kf.initiate(x, y)
        self.bbox = bbox
        self.age = 1
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.state = 'Tentative'
        self.history = [(frame_idx, x, y)]  # list of (t, x, y)
        self.last_frame = frame_idx

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, x: float, y: float, bbox, frame_idx: int):
        self.kf.update((x, y))
        self.bbox = bbox
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.history.append((frame_idx, x, y))
        self.last_frame = frame_idx
        if self.state == 'Tentative' and self.hits >= Track.min_hits_global:
            self.state = 'Confirmed'

    def position(self) -> Tuple[float,float]:
        return self.kf.state_xy()

class Tracker:
    def __init__(self,
                 dist_threshold: float = 60.0,
                 iou_weight: float = 0.6,
                 area_weight: float = 0.3,
                 max_age: int = 8,
                 min_hits: int = 3,
                 iou_gate: float = 0.1,
                 dt: float = 1.0):
        """
        dist_threshold: gate for centroid distance after prediction (pixels).
        iou_weight, area_weight: cost weights; distance weight = 1 - (iou_weight + area_weight).
        max_age: frames to keep a track without updates.
        min_hits: frames before confirming a track.
        iou_gate: minimum IoU to consider a match (helps avoid identity switches).
        """
        assert 0.0 <= iou_weight <= 1.0 and 0.0 <= area_weight <= 1.0
        assert iou_weight + area_weight <= 1.0
        self.dist_threshold = dist_threshold
        self.iou_weight = iou_weight
        self.area_weight = area_weight
        self.dist_weight = 1.0 - (iou_weight + area_weight)
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_gate = iou_gate
        self.dt = dt

        Track.min_hits_global = min_hits
        self.tracks: List[Track] = []
        self.next_id = 1
        self.t = -1  # frame counter

    def _build_cost(self, frame_prev: np.ndarray, prev_props: Dict[int, dict],
                    frame_curr: np.ndarray, curr_props: Dict[int, dict]) -> Tuple[np.ndarray, List[int], List[int]]:
        """
        Build cost matrix between existing tracks (predicted) and current detections.
        Returns cost, list(track_indices), list(curr_local_ids)
        """
        trk_indices = list(range(len(self.tracks)))
        det_ids = list(curr_props.keys())
        if len(trk_indices) == 0 or len(det_ids) == 0:
            return np.zeros((0,0)), trk_indices, det_ids

        cost = np.ones((len(trk_indices), len(det_ids)), dtype=float)
        for i, ti in enumerate(trk_indices):
            trk = self.tracks[ti]
            pred_xy = trk.position()
            tb = trk.bbox
            # gate by distance
            for j, cid in enumerate(det_ids):
                c = curr_props[cid]
                cx, cy = c['centroid']
                dist = euclidean(pred_xy, (cx, cy))
                if dist > self.dist_threshold:
                    cost[i, j] = 1.0  # disallow
                    continue
                # IoU between previous bbox and current bbox as a proxy (fast)
                # For tighter IoU, you can switch to mask IoU via iou_local using frames.
                iou_bbox = iou_from_bboxes(tb, c['bbox'])
                if iou_bbox < self.iou_gate:
                    cost[i, j] = 1.0
                    continue
                # Area similarity
                prev_area = (tb[2]-tb[0]+1)*(tb[3]-tb[1]+1)
                curr_area = (c['bbox'][2]-c['bbox'][0]+1)*(c['bbox'][3]-c['bbox'][1]+1)
                area_diff = abs(prev_area - curr_area) / max(prev_area, curr_area)

                # Normalize distance
                dist_cost = min(dist / self.dist_threshold, 1.0)
                iou_cost = 1.0 - iou_bbox

                cost[i, j] = self.iou_weight * iou_cost + self.area_weight * area_diff + self.dist_weight * dist_cost

        return cost, trk_indices, det_ids

    def predict(self):
        for trk in self.tracks:
            trk.predict()

    def update(self, frame_prev: Optional[np.ndarray], prev_props: Optional[Dict[int, dict]],
               frame_curr: np.ndarray, curr_props: Dict[int, dict]):
        """
        Update tracker with current frame detections.
        Returns: dict mapping current local IDs -> persistent global IDs.
        """
        self.t += 1
        self.predict()
    
        # Build cost matrix
        cost, trk_indices, det_ids = self._build_cost(frame_prev, prev_props, frame_curr, curr_props)
    
        matches: List[Tuple[Track, int]] = []  # (Track object, detection local ID)
        unmatched_trk = trk_indices.copy()
        unmatched_det = list(range(len(det_ids)))
    
        if cost.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost)
            matched_trk_idx = set()
            matched_det_idx = set()
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < 1.0:  # passed gating
                    track_obj = self.tracks[trk_indices[r]]
                    det_cid = det_ids[c]
                    matches.append((track_obj, det_cid))
                    matched_trk_idx.add(r)
                    matched_det_idx.add(c)
            unmatched_trk = [i for i in trk_indices if i not in matched_trk_idx]
            unmatched_det = [j for j in range(len(det_ids)) if j not in matched_det_idx]
    
        # Update matched tracks
        for trk, cid in matches:
            cx, cy = curr_props[cid]['centroid']
            trk.update(cx, cy, curr_props[cid]['bbox'], self.t)
    
        # Age unmatched tracks
        for idx in unmatched_trk:
            trk = self.tracks[idx]
            trk.hit_streak = 0  # break streak
    
        # Create new tracks for unmatched detections
        for j in unmatched_det:
            cid = det_ids[j]
            cx, cy = curr_props[cid]['centroid']
            bbox = curr_props[cid]['bbox']
            trk = Track(self.next_id, cx, cy, bbox, self.t, dt=self.dt)
            if self.min_hits == 1:
                trk.state = 'Confirmed'
            self.tracks.append(trk)
            self.next_id += 1
    
        # Build mapping BEFORE pruning
        gid_for_cid = {cid: trk.id for trk, cid in matches}
    
        # Prune old tracks
        alive = []
        for trk in self.tracks:
            if trk.time_since_update > self.max_age:
                trk.state = 'Deleted'
            if trk.state != 'Deleted':
                alive.append(trk)
        self.tracks = alive
    
        return gid_for_cid

# =========================
# End-to-end tracking API
# =========================

def track_mask_stack(masks: np.ndarray,
                     dist_threshold: float = 60.0,
                     iou_weight: float = 0.6,
                     area_weight: float = 0.3,
                     max_age: int = 8,
                     min_hits: int = 3,
                     iou_gate: float = 0.1,
                     dt: float = 1.0):
    """
    masks: (T, H, W) integer-labeled masks per frame (per-frame local IDs).
    Returns:
      - tracked_masks: (T, H, W) with persistent global IDs.
      - tracks: list of Track objects with trajectory history.
    """
    T, H, W = masks.shape
    tracked_masks = np.zeros_like(masks, dtype=np.int32)
    tracker = Tracker(dist_threshold, iou_weight, area_weight, max_age, min_hits, iou_gate, dt)

    prev_frame = None
    prev_props = None

    for t in range(T):
        frame = masks[t]
        curr_props = extract_props(frame)
        gid_for_cid = tracker.update(prev_frame, prev_props, frame, curr_props)

        # Write global IDs for this frame
        for cid, gid in gid_for_cid.items():
            tracked_masks[t][frame == cid] = gid

        prev_frame = frame
        prev_props = curr_props

    # After loop, collect tracks
    return tracked_masks, tracker.tracks

# =========================
# Trajectory plotting
# =========================

def plot_trajectories_with_overlay(tracks, masks, frame_idx=0,
                                   frame_range=(0, 121),
                                   persistent_min_hits=3,
                                   min_length=5,
                                   invert_y=True):
    # 1. Show the mask image as a light grayscale background
    plt.imshow(masks[frame_idx], cmap='gray', alpha=0.3)  # alpha controls transparency

    # 2. Loop through tracks and plot those that meet criteria
    for tr in tracks:
        # Filter by persistence
        if tr.hits < persistent_min_hits:
            continue

        # Extract positions within frame_range
        xs, ys, fs = [], [], []
        for f, x, y in tr.history:
            if frame_range[0] <= f <= frame_range[1]:
                xs.append(x)
                ys.append(y)
                fs.append(f)

        # Filter by length
        if len(xs) < min_length:
            continue

        plt.plot(xs, ys, lw=1)

    # 3. Match orientation to image coordinates
    if invert_y:
        plt.gca().invert_yaxis()

    plt.title(f"Trajectories over frame {frame_idx}")
    plt.axis('off')
    plt.show()


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    # Load your per-frame/ unstitched local masks (T, H, W)
    mask_path = ''
    masks = np.load(mask_path)  # replace with your path

    tracked_masks, tracks = track_mask_stack(
        masks,
        dist_threshold=50.0,
        iou_weight=0.6,
        area_weight=0.3,
        max_age=8,
        min_hits=3,
        iou_gate=0.15,
        dt=1.0
    )

    # Ensure dtype and shape match the original
    tracked_masks = tracked_masks.astype(masks.dtype)
    assert tracked_masks.shape == masks.shape, "Shape mismatch — something went wrong"
    
    # Build save path in the same folder as the original
    folder = os.path.dirname(mask_path)
    base = os.path.splitext(os.path.basename(mask_path))[0]
    save_path = os.path.join(folder, f"{base}_tracked.npy")
    
    # # Save
    np.save(save_path, tracked_masks)
    print(f"Tracked masks saved to: {save_path}")


#%%
# === Animation of trajectories over mask background with outlined, offset ID labels ===
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as pe  # for text outline

# --- Settings ---
frame_range = (0, 280)       # Experiment frames
persistent_min_hits = 3      # Filter: minimum hits
min_length = 269             # Filter: minimum trajectory length
invert_y = False             # Match image orientation
bg_frame_idx = 30             # Which frame to use as background
label_offset = (2, -2)       # (x, y) pixel offset for labels

# --- Prepare figure ---
fig, ax = plt.subplots()
bg = ax.imshow(masks[bg_frame_idx], cmap='gray', alpha=0.5, vmin=0, vmax=1)
if invert_y:
    ax.invert_yaxis()
ax.set_axis_off()

# --- Prepare line and text objects for each track ---
lines = []
texts = []
valid_tracks = []  # list of (pts, track_id)
for tr in tracks:
    if tr.hits >= persistent_min_hits:
        pts = [(f, x, y) for f, x, y in tr.history
               if frame_range[0] <= f <= frame_range[1]]
        if len(pts) >= min_length:
            line, = ax.plot([], [], lw=1)
            text = ax.text(
                0, 0, str(tr.id),
                fontsize=6,
                color=line.get_color(),
                ha='left', va='bottom',
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")]
            )
            lines.append(line)
            texts.append(text)
            valid_tracks.append((pts, tr.id))

# --- Init function ---
def init():
    for line, text in zip(lines, texts):
        line.set_data([], [])
        text.set_position((0, 0))
        text.set_visible(False)
    return lines + texts

# --- Update function ---
def update(frame):
    for (pts, tid), line, text in zip(valid_tracks, lines, texts):
        xs = [x for f, x, y in pts if f <= frame]
        ys = [y for f, x, y in pts if f <= frame]
        if xs and ys:
            line.set_data(xs, ys)
            # Offset label position slightly from last point
            text.set_position((xs[-1] + label_offset[0], ys[-1] + label_offset[1]))
            text.set_visible(True)
        else:
            text.set_visible(False)
    return lines + texts

# --- Create animation ---
ani = FuncAnimation(
    fig, update,
    frames=range(frame_range[0], frame_range[1] + 1),
    init_func=init,
    blit=True,
    interval=5  # ms between frames
)

plt.tight_layout()
plt.show()

