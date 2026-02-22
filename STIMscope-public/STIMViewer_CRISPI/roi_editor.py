
from pathlib import Path
import os




import numpy as np


try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None

try:
    import napari
    NAPARI_AVAILABLE = True
except ImportError:
    NAPARI_AVAILABLE = False
    napari = None

try:
    import magicgui
    MAGICGUI_AVAILABLE = True
    print("✅ MagicGUI imported successfully")
except ImportError as e:
    MAGICGUI_AVAILABLE = False
    magicgui = None
    print(f"❌ MagicGUI import failed: {e}")

try:
    from roi_thresh import threshold_patch
    THRESHOLD_AVAILABLE = True
except ImportError:
    THRESHOLD_AVAILABLE = False
    threshold_patch = None




def refine_rois(mean, labels, return_viewer=False, on_close_callback=None):
    print("test")
    

    if not NAPARI_AVAILABLE:
        print("❌ Napari not available, cannot launch ROI editor")
        return labels
    
    if not MAGICGUI_AVAILABLE:
        print("⚠️ MagicGUI not available, some widgets may not work")
    
    if not THRESHOLD_AVAILABLE:
        print("⚠️ threshold_patch not available, ROI refinement will be disabled")
    




    stack = labels


    labels0 = np.zeros(mean.shape, np.int16)
    for i, m in enumerate(stack, 1):
        if m.shape != mean.shape: 
            print(f"Shape mismatch at index {i}: mask {m.shape}, mean {mean.shape}")
            raise ValueError("mask shape mismatch")
        labels0[m.astype(bool) & (labels0 == 0)] = i   


    print("unique IDs now:", np.unique(labels0)[:20])
    viewer = napari.current_viewer() or napari.Viewer()  
    print("passed napari current")

    viewer.mouse_double_click_callbacks.clear()

    vmin, vmax = np.percentile(mean, (1, 99.5))   
    viewer.add_image(mean.astype("float32"), 
                    name="mean",
                    colormap="gray",
                    contrast_limits=(vmin, vmax),
                    blending="additive")
    print("passed viewer add image")
    lbl = viewer.add_labels( 
        labels0.copy(),
        name="ROIs",
        opacity=0.6,         
        blending="translucent",
    )
    print("passed viewer add labels")


    from PyQt5.QtCore import QTimer
    QTimer.singleShot(
        0,
        lambda: viewer.window.qt_viewer.canvas.native.update() 
    )
    print("passed viewer add labels")


    lbl.contour = 1 
    PAD      = 5        
    

    def add_drawing_tools():
       
        try:

            shapes_layer = viewer.add_shapes(
                name="Drawing",
                edge_color="red",
                face_color="red",
                edge_width=2,
                opacity=0.7
            )
            

            next_roi_id = lbl.data.max() + 1 if lbl.data.max() > 0 else 1
            
            def on_shape_added(event):
               
                nonlocal next_roi_id
                try:

                    if len(shapes_layer.data) > 0:
                        shape = shapes_layer.data[-1]
                        if len(shape) > 2:

                            from skimage.draw import polygon
                            coords = np.array(shape)
                            rr, cc = polygon(coords[:, 0], coords[:, 1], shape=lbl.data.shape)
                            

                            valid_mask = (rr >= 0) & (rr < lbl.data.shape[0]) & (cc >= 0) & (cc < lbl.data.shape[1])
                            rr = rr[valid_mask]
                            cc = cc[valid_mask]
                            
                            if len(rr) > 0:

                                lbl.data[rr, cc] = next_roi_id
                                next_roi_id += 1
                                

                                shapes_layer.data = shapes_layer.data[:-1]
                                
                                print(f"✅ Added new ROI {next_roi_id - 1}")
                                viewer.status = f"Added ROI {next_roi_id - 1}"
                except Exception as e:
                    print(f"⚠️ Error adding shape: {e}")
            

            shapes_layer.events.data.connect(on_shape_added)
            

            def on_key_press(event):
               
                if event.key == 'd':

                    viewer.layers.selection = [shapes_layer]
                    viewer.status = "Drawing mode: Click to add points, double-click to finish"
                elif event.key == 'e':

                    viewer.layers.selection = [lbl]
                    viewer.status = "Erase mode: Click on ROIs to delete"
                elif event.key == 'r':

                    lbl.data = labels0.copy()
                    viewer.status = "Masks reset to original"
                elif event.key == 'c':

                    shapes_layer.data = []
                    viewer.status = "Drawing cleared"
            

            viewer.bind_key('d', lambda v: on_key_press(type('Event', (), {'key': 'd'})()))
            viewer.bind_key('e', lambda v: on_key_press(type('Event', (), {'key': 'e'})))
            viewer.bind_key('r', lambda v: on_key_press(type('Event', (), {'key': 'r'})))
            viewer.bind_key('c', lambda v: on_key_press(type('Event', (), {'key': 'c'})))
            
            print("✅ Drawing tools added (d=draw, e=erase, r=reset, c=clear)")
            
        except Exception as e:
            print(f"⚠️ Failed to add drawing tools: {e}")
    

    add_drawing_tools()


    
    @lbl.mouse_double_click_callbacks.append 
    def refine_one(layer, event):
        event.handled = True 
        r, c = map(int, event.position) 
        rid  = layer.data[r, c]
        if rid == 0:
            return

        mask = layer.data == rid
        ys, xs = np.where(mask)
        if ys.size == 0:       
            return






        y0, y1 = ys.min() - PAD, ys.max() + PAD + 1
        x0, x1 = xs.min() - PAD, xs.max() + PAD + 1
        patch  = mean[y0:y1, x0:x1]
        if patch.size == 0:
            return


        if not THRESHOLD_AVAILABLE:
            viewer.status = "threshold_patch not available"
            return
        new_masks, _ = threshold_patch(patch)
        if not new_masks:
            viewer.status = "No new mask found"
            return  




        best, best_iou = None, 0
        for m in new_masks:
            iou = (m & mask[y0:y1,x0:x1]).sum() / (m | mask[y0:y1,x0:x1]).sum()
            if iou > best_iou:
                best, best_iou = m, iou 

        layer.data[mask] = 0 


        layer.data[y0:y1, x0:x1][best] = rid



        old_opacity = layer.opacity
        layer.opacity = min(1.0, old_opacity + 0.4)
        from PyQt5.QtCore import QTimer 
        QTimer.singleShot(400, lambda: setattr(layer, "opacity", old_opacity))


        viewer.status = f"ROI {rid} refined (IoU {best_iou:.2f})"



    from PyQt5.QtWidgets import QLabel
    from PyQt5.QtCore    import Qt

    selected: set[int] = set() 


    sel_label = QLabel('Selected ROIs: none')
    sel_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    sel_label.setWordWrap(True)          


    def refresh_sel_label() -> None:
       
        text = ', '.join(map(str, sorted(selected))) or 'none'
        sel_label.setText(f'Selected ROIs: {text}')



    def toggle_base(id: int = 1):
       
        selected.symmetric_difference_update([id])
        refresh_sel_label()

    def keep_base():
       
        if not selected:
            viewer.status = "Nothing selected."
            return
        lbl.data[~np.isin(lbl.data, list(selected))] = 0 # build a boolean mask of pixels whose labels are in selected and uses ~ to invert them

        viewer.status = f"Kept {len(selected)} ROIs"

    def reset_base():
       
        lbl.data = labels0.copy()
        selected.clear()
        refresh_sel_label()
        viewer.status = "Mask reset"

    def export_base():
        try:
           
            import numpy as np

            np.savez_compressed("rois.npz", labels=lbl.data)
            viewer.status = "Exported rois.npz"
        except Exception as e:
            viewer.status = f"❌ Export failed: {e}"


    if MAGICGUI_AVAILABLE:
        print("🔄 Creating MagicGUI widgets...")
        toggle = magicgui.magicgui(call_button='Toggle select ROI')(toggle_base)
        keep = magicgui.magicgui(call_button='Keep only selected')(keep_base)
        reset = magicgui.magicgui(call_button='Reset masks')(reset_base)
        export = magicgui.magicgui(call_button='Export → trace_view')(export_base)
    else:

        toggle = toggle_base
        keep = keep_base
        reset = reset_base
        export = export_base




    if MAGICGUI_AVAILABLE:
        try:
            print("🔄 Adding widgets to viewer...")
            for i, w in enumerate([toggle, keep, reset, export]):
                viewer.window.add_dock_widget(w, area='right')
                print(f"✅ Added widget {i+1}/4")
            print("✅ All MagicGUI widgets added to viewer")
        except Exception as e:
            print(f"⚠️ Failed to add MagicGUI widgets: {e}")
    else:
        print("⚠️ MagicGUI not available, widgets not added to viewer")

    viewer.window.add_dock_widget(sel_label, area='right')
    refresh_sel_label()        # initialize text once
    

    def auto_save_on_close():
       
        try:
            np.savez_compressed("rois.npz", labels=lbl.data)
            print("✅ Auto-saved updated ROIs to rois.npz")
        except Exception as e:
            print(f"❌ Auto-save failed: {e}")
    

    try:

        original_close_event = viewer.window._qt_window.closeEvent
        
        def close_event_with_save(event):
           
            auto_save_on_close()
            

            if on_close_callback:
                try:
                    on_close_callback()
                except Exception as e:
                    print(f"⚠️ Error in close callback: {e}")
            

            if original_close_event:
                original_close_event(event)
            else:
                event.accept()
        

        viewer.window._qt_window.closeEvent = close_event_with_save
        print("✅ Auto-save and restore callback connected to Napari close event")
    except Exception as e:
        print(f"⚠️ Could not connect auto-save to close event: {e}")
    
    if return_viewer:
        return labels0, viewer
    return labels0
