# roi_thresh.py
import numpy as np, cupy as cp, cv2
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi

# def _to_cpu(a):              # If needed to convert a CuPy array into a numpy array
#     return cp.asnumpy(a) if isinstance(a, cp.ndarray) else a

# Main ROI finding routine but this is for each individual ROI the user clicks on in the editor
# Can define multiple functions and assign it to different ROIs, customizing the thresholding for certain ROIs
# or instead we can have the user draw it themselves 
def threshold_patch(img, # Same parameters as before
                    gauss_ksize=(3,3),
                    gauss_sigma=0.5,
                    min_area=5,
                    max_area=200):
    """Return list[mask] – tighter fit using local max-projection + erosion."""
    # 1) Convert to float32 and blur
    # Apply gaussian blur to suppress and noise
    blur = cv2.GaussianBlur(img.astype(np.float32), gauss_ksize, gauss_sigma)

    # 2) Normalize the blurred image needs to be between 0 and 225 for OpenCVs adaptive threshold
    norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)
    # Computes a local mean over 51 * 51 windows 
    # These methods need to be fixed, this is just for the general structure of the pipeline
    # And how we can have the user re-threshold ROIs
    bw   = cv2.adaptiveThreshold(norm.astype('uint8'), 1,
                                 cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 21, 0)

    # 3) Optional erosion to shrink halo to fit ROIs tighter around cells
    # but can also be bad for masks that are too tight and need to be loosened
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    bw = cv2.erode(bw, k3, iterations=1)

    # 4) remove noise inside the ROI
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k3)
    num, lab = cv2.connectedComponents(bw, 8) # labels every 8 connected blob
    # Which means treat any two foreground pixels that touch by an edge or corner
    # As the same object and assign each object an ID
    # can also be 4 which is only the pixels that share an edge
    # think of it as a 3 x3 square
    # in binary images pixels are either in the foreground or background

    masks, sizes = [], [] # lists for collecting boolean ROI masks and each mask's pixel count
    for lab_id in range(1, num): # skip label 0 and iterate over every connected component 
        m = lab == lab_id # build a boolean mask for the current blob
        # m is true where the component lives and False elsewhere
        # shape is lab.shape

        # measure the blob size
        pix = int(m.sum()) # m.sum() counts True pixels 
        if pix < min_area: # if too small discard them
            continue
        # if too big decide to split them to enter the watershed split routine
        if max_area and pix > max_area:
            # for every foreground pixel dist stores its distance to the nearest background pixel
            # ridges at the blob center and zeros at the edges
            # uint8 255 is an 8bit 0/255 image
            # cv2.DIST is the euclidean distance
            dist = cv2.distanceTransform(m.astype('uint8')*255, cv2.DIST_L2, 5)
            coords = peak_local_max(dist, min_distance=4, labels=m) # searches dist for peaks at least 4 pixels apart, tuned with min_distance
            # returns an Nx2 array of (row, col) coordinates
            peaks  = np.zeros_like(dist, bool) # builds a binary seed image with True only at peak pixels
            peaks[tuple(coords.T)] = True
            # converts the binary peaks image into integer seed labels 1,2 ,3...
            labels_ws = watershed(-dist, ndi.label(peaks)[0], mask=m) # floods the blob 

            for sub in np.unique(labels_ws)[1:]: # iterate over the watershed basins
                submask = labels_ws == sub # grab unique sub-labels
                if submask.sum() >= min_area: # build a boolean mask for each and keep only if passes min area threshold
                    masks.append(submask) # append to outputs
                    sizes.append(int(submask.sum()))
        else: # when blob is within min or max  adds entire component to the results
            masks.append(m)
            sizes.append(pix)
    return masks, sizes # return 2-d bool arrays, each separate ROI
    # and sizes which is list of integers area of each ROI
