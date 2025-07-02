# roi_editor.py
import numpy as np, cupy as cp, napari
from magicgui import magicgui
from roi_thresh import threshold_patch

def refine_rois(mean, labels, return_viewer=False):
    # Stream 5400 frames average them and return a 2-D float array as the background
    #mean  = compute_mean_projection(load_movie("cropped.avi"), calib_frames=5400)

    # --- 1) initial masks ----------------------------------------
    stack = labels
    # stack = np.load("rois.npz")["masks"]   # load the initial masks where rois.npz is the initial thresholding mask from main.py with N boolean masks
    # labels0 = np.zeros(mean.shape, np.int16) # empty integer image same dims as mean as the label map
    labels0 = np.zeros(mean.shape, np.int16)
    for i, m in enumerate(stack, 1): # start enumerating masks skipping label 0 
        if m.shape != mean.shape: # check the size
            print(f"Shape mismatch at index {i}: mask {m.shape}, mean {mean.shape}")
            raise ValueError("mask shape mismatch")
        labels0[m.astype(bool) & (labels0 == 0)] = i   # write label i into pixels where mask is True and no occupied hence (labels0==0)

    # Check if labels are assigned
    print("unique IDs now:", np.unique(labels0)[:20])
    viewer = napari.current_viewer() or napari.Viewer()  # get the current viewer or create a new one
# Open window
    viewer.mouse_double_click_callbacks.clear() # Clear double click zoom in feature
    # viewer.add_image(mean, name="mean", colormap="gray", blending="additive")
    vmin, vmax = np.percentile(mean, (1, 99.5))   # Stretch the contrast of the mean image for clearer mean image
    viewer.add_image(mean.astype("float32"), # add the mean image as a grayscale
                    name="mean",
                    colormap="gray",
                    contrast_limits=(vmin, vmax),
                    blending="additive") # adds label map as a semi transparent overlay
    lbl = viewer.add_labels( # use a copy of the labels for future use
        labels0.copy(),
        name="ROIs",
        opacity=0.6,         # 60% visible
        blending="translucent",
    )

    qt_canvas = viewer.window.qt_viewer.canvas
    from PyQt5.QtCore import QTimer
    QTimer.singleShot(
        0,
        lambda: (
            qt_canvas.update(),            # VisPy repaint
            qt_canvas.native.update(),     # Qt widget repaint
        )
    )

    lbl.contour = 1 # Give each roi a random color contour
    PAD      = 5        # pixels around ROI when cropping
    @lbl.mouse_double_click_callbacks.append ## appends mouse double clicks to the refine one function
    def refine_one(layer, event):
        event.handled = True # clear any previous double clicks
        r, c = map(int, event.position) # pick the pixel and roi id
        rid  = layer.data[r, c]
        if rid == 0:
            return
        # get the current mask for that roi and its bounding box
        mask = layer.data == rid
        ys, xs = np.where(mask)
        if ys.size == 0:        # no pixels (shouldn’t happen, but safe)
            return
        # extract the mean image patch with padding so the ROI mask is not cut off from the box
        # PAD is the pixel margin on all sides, too small threshold might cut off edges of the cell
        # too big might have neighboring cells invade the patch
        # what if there is a way we can just extract the roi mask shape
        # put that into an empty black box and then run thresholding on that and then replace the old shape
        # instead of cutting out a box from the original mask which might have neighboring ROIs
        y0, y1 = ys.min() - PAD, ys.max() + PAD + 1
        x0, x1 = xs.min() - PAD, xs.max() + PAD + 1
        patch  = mean[y0:y1, x0:x1]
        if patch.size == 0:
            return
        # call threshold path to get a better mask, we can also change this for what type of ROI we clicked on
        # so we can call a custom threshold for all types of ROIs
        new_masks, _ = threshold_patch(patch)
        if not new_masks:
            viewer.status = "No new mask found"
            return  
        # which ever mask has a better i o u we update the old mask
        # iou is the overlap between two masks, shared pixels/pixels in either mask
        # so if the new mask shares more pixels with the old one its a better fit because it fit tighter over the original ROI
        # but may want to change this for different ROI fits, some original masks might be missing part of the ROI
        best, best_iou = None, 0
        for m in new_masks:
            iou = (m & mask[y0:y1,x0:x1]).sum() / (m | mask[y0:y1,x0:x1]).sum()
            if iou > best_iou:
                best, best_iou = m, iou 

        layer.data[mask] = 0 # live label image napari is displaying
        # mask is a boolean array same shape as layer.data that is True for every pixel of the old ROI
        # setting them to 0 erases the previous ROI from the map
        layer.data[y0:y1, x0:x1][best] = rid # then we select the cropped region around the ROI the same coordinates to build the patch
        # then those are set to the original ROI ID rid

        # quick flash
        old_opacity = layer.opacity # store the current opacity
        layer.opacity = min(1.0, old_opacity + 0.4) # make the label brighter 
        from PyQt5.QtCore import QTimer # import short timer
        QTimer.singleShot(400, lambda: setattr(layer, "opacity", old_opacity))
        # call back 400 ms later that sets the layer opacity to the original value
        # single shot ensures we dont have a persistent timer object
        viewer.status = f"ROI {rid} refined (IoU {best_iou:.2f})"
            # update the status bar of napari

    # ---------------- selection widgets --------------------------
    from PyQt5.QtWidgets import QLabel
    from PyQt5.QtCore    import Qt

    selected: set[int] = set()   # hold all ROI IDs the user marked 

    # create label that shows what ROIs where selected
    sel_label = QLabel('Selected ROIs: none')
    sel_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    sel_label.setWordWrap(True)           # allow multi-line text

    # show the current selected ROIs
    def refresh_sel_label() -> None:
        """Update the label to show current keep-list."""
        text = ', '.join(map(str, sorted(selected))) or 'none'
        sel_label.setText(f'Selected ROIs: {text}')


    # uses magic gui to build a QT button to toggle the ROI based on the function its over
    @magicgui(call_button='Toggle select ROI')
    def toggle(id: int = 1):
        """Add or remove an ROI ID from the keep-list."""
        selected.symmetric_difference_update([id])
        refresh_sel_label()

    # if nothing selected shows status
    @magicgui(call_button='Keep only selected')
    def keep():
        """Hide every ROI *not* in the keep-list."""
        if not selected:
            viewer.status = "Nothing selected."
            return
        lbl.data[~np.isin(lbl.data, list(selected))] = 0 # build a boolean mask of pixels whose labels are in selected and uses ~ to invert them
        # those are set to 0 all unselected ROis disappear
        viewer.status = f"Kept {len(selected)} ROIs"

    # restores the original label map and empties selected map
    @magicgui(call_button='Reset masks')
    def reset():
        """Restore original masks and clear the keep-list."""
        lbl.data = labels0.copy()
        selected.clear()
        refresh_sel_label()
        viewer.status = "Mask reset"


    @magicgui(call_button='Export → trace_view')
    def export():
        try:
            """Write the current label map for the next stage and update the projection."""
            import numpy as np
        

            # Save the current label map
            np.savez_compressed("rois.npz", labels=lbl.data)
            viewer.status = "Exported rois.npz"
        except Exception as e:
            viewer.status = f"❌ Export failed: {e}"



    # ---- dock widgets -------------------------------------------
    for w in (toggle, keep, reset, export):
        viewer.window.add_dock_widget(w, area='right')

    viewer.window.add_dock_widget(sel_label, area='right')
    refresh_sel_label()        # initialize text once
    if return_viewer:
        return labels0, viewer
    return labels0
