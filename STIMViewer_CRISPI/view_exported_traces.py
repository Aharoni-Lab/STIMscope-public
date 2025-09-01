import numpy as np
import matplotlib.pyplot as plt
import sys

def view_exported_traces(traces_file="live_traces.npy", roi_info_file="roiprint_export.npz"):

    traces = np.load(traces_file, allow_pickle=True).item()

    print(f"\nLoaded traces from {traces_file}: {len(traces)} ROIs")
    for key, arr in traces.items():
        print(f"  {key}: {len(arr)} frames")


    roi_info = np.load(roi_info_file)
    ids = roi_info["ids"]
    roi_sizes = roi_info["roi_sizes"]
    shape = roi_info["shape"]

    print(f"\nROI metadata from {roi_info_file}:")
    print(f"  IDs: {ids}")
    print(f"  Sizes: {roi_sizes}")
    print(f"  Image shape: {shape}")


    plt.figure(figsize=(12, 6))
    for i, (roi_key, roi_trace) in enumerate(traces.items()):
        plt.plot(roi_trace, label=f"{roi_key} (size={roi_sizes[i]:.1f})")

    plt.xlabel("Frame index")
    plt.ylabel("Mean intensity")
    plt.title("Exported ROI Traces")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 3:
        view_exported_traces(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python3 view_exported_traces.py live_traces.npy roiprint_export.npz")
