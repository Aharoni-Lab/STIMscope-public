#!/usr/bin/env python3
"""
Minimal UI for calibration utilities with an "ASIFT Calibration" button
placed to the right of a "Troubleshooting" button.

Actions:
- ASIFT Calibration: computes 3x3 H using ASIFT-style matching (fallback SIFT),
  sends H to the projector over ZMQ, and writes H to the provided text file.
- Troubleshooting: opens the local synchronization summary if present.
"""

import os
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox

from ZMQ_sender_mask.asift_calibration import run_asift_ui_action


def browse_open(entry_widget, title="Select file", filetypes=(("All files", "*.*"),)):
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    if path:
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, path)


def browse_save(entry_widget, title="Save H text file", defaultextension=".txt"):
    path = filedialog.asksaveasfilename(title=title, defaultextension=defaultextension,
                                        filetypes=(("Text", "*.txt"), ("All files", "*.*")))
    if path:
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, path)


def on_asift_calibration(ref_entry, cam_entry, h_entry):
    ref_path = ref_entry.get().strip()
    cam_path = cam_entry.get().strip()
    h_txt_path = h_entry.get().strip()

    if not ref_path or not os.path.isfile(ref_path):
        messagebox.showerror("ASIFT Calibration", "Please select a valid reference image path.")
        return
    if not cam_path or not os.path.isfile(cam_path):
        messagebox.showerror("ASIFT Calibration", "Please select a valid camera image path.")
        return
    if not h_txt_path:
        messagebox.showerror("ASIFT Calibration", "Please choose where to save the H text file.")
        return

    try:
        ok, H = run_asift_ui_action(ref_path, cam_path, h_txt_path, endpoint="tcp://127.0.0.1:5560")
        if ok:
            messagebox.showinfo("ASIFT Calibration", f"Calibration OK.\nSaved: {h_txt_path}")
        else:
            messagebox.showwarning("ASIFT Calibration", "Calibration failed: insufficient matches or no homography.")
    except Exception as e:
        messagebox.showerror("ASIFT Calibration", f"Error: {e}")


def on_troubleshooting():
    # Try to open a local summary if present
    summary = os.path.join(os.path.dirname(__file__), "..", "SYNCHRONIZATION_FIXES_SUMMARY.md")
    summary = os.path.abspath(summary)
    if os.path.isfile(summary):
        try:
            subprocess.Popen(["xdg-open", summary])
            return
        except Exception:
            pass
    messagebox.showinfo("Troubleshooting", "No troubleshooting summary found in this repository.")


def build_ui():
    root = tk.Tk()
    root.title("Calibration Tools")

    # Top button row: Troubleshooting (left), ASIFT Calibration (right)
    btn_row = tk.Frame(root)
    btn_row.pack(fill=tk.X, padx=10, pady=10)

    btn_trouble = tk.Button(btn_row, text="Troubleshooting", command=on_troubleshooting)
    btn_trouble.pack(side=tk.LEFT, padx=(0, 8))

    btn_asift = tk.Button(btn_row, text="ASIFT Calibration")
    btn_asift.pack(side=tk.LEFT)

    # Paths section
    form = tk.Frame(root)
    form.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Reference image
    tk.Label(form, text="Reference image:").grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
    ent_ref = tk.Entry(form, width=60)
    ent_ref.grid(row=0, column=1, sticky=tk.W, padx=4, pady=4)
    tk.Button(form, text="Browse", command=lambda: browse_open(ent_ref, title="Select reference image",
              filetypes=(("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("All files", "*.*")))).grid(row=0, column=2, padx=4, pady=4)

    # Camera image
    tk.Label(form, text="Camera image:").grid(row=1, column=0, sticky=tk.W, padx=4, pady=4)
    ent_cam = tk.Entry(form, width=60)
    ent_cam.grid(row=1, column=1, sticky=tk.W, padx=4, pady=4)
    tk.Button(form, text="Browse", command=lambda: browse_open(ent_cam, title="Select camera image",
              filetypes=(("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("All files", "*.*")))).grid(row=1, column=2, padx=4, pady=4)

    # H text output
    tk.Label(form, text="Save H (txt):").grid(row=2, column=0, sticky=tk.W, padx=4, pady=4)
    ent_h = tk.Entry(form, width=60)
    ent_h.grid(row=2, column=1, sticky=tk.W, padx=4, pady=4)
    tk.Button(form, text="Browse", command=lambda: browse_save(ent_h, title="Save H text file"))
    .grid(row=2, column=2, padx=4, pady=4)

    # Wire ASIFT callback now that entries exist
    btn_asift.configure(command=lambda: on_asift_calibration(ent_ref, ent_cam, ent_h))

    return root


def main():
    root = build_ui()
    root.mainloop()


if __name__ == "__main__":
    main()


