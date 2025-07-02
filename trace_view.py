# trace_view.py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def view_traces(trace_path, rois_path="rois.npz", last_n=100, max_rois=10):
    # sanity checks
    if not os.path.exists(rois_path):
        sys.exit("❌ No rois.npz found. Run ROI discovery/refinement first.")
    if not os.path.exists(trace_path):
        sys.exit(f"❌ No trace file found at {trace_path}")

    # load label map & ROI IDs
    lab = np.load(rois_path)["labels"].astype(int)
    ids = np.unique(lab)
    ids = ids[ids >0]
    if ids.size == 0:
        sys.exit("❌ ROI mask contains zero ROIs.")

    # load the rolling‐trace array (T × N_rois or N_rois × T)
    traces = np.load(trace_path)
    if traces.ndim != 2:
        sys.exit("❌ Trace file must be 2D (frames x ROIs).")
    # ensure shape is (T, N)

    if traces.shape[0] < traces.shape[1]:
        traces = traces.T

    T, N = traces.shape
    if last_n > T:
        last_n = T

    shown_rois = min(max_rois, N)
    trace_block = traces[-last_n:, :shown_rois]
    x = np.arange(-last_n + 1, 1, 1)

    # quick preview
    plt.figure(figsize=(12, 6))

    for i in range(shown_rois):
        plt.plot(x, trace_block[:, i], label=f"ROI {i+1}")
    plt.xlabel("Frames ago")
    plt.ylabel("Mean intensity")
    plt.title(f"Live ROI Traces — last {last_n} frames, {shown_rois} ROIs")
    plt.legend(loc="upper right", fontsize="small", ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()