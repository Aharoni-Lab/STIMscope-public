# online_main.py
import time # For timer

from otsu_thresh import ( # Import helper functions from otsu_thresh.py
    load_movie, 
    compute_mean_projection,
    denoise_and_threshold_gpu,
    save_rois
)

def main():
    time_start = time.time() # Capture the current time
    print("Starting ROI discovery pipeline...")
    # -------------------------------
    # 1) Load the movie
    # -------------------------------
    # Change movie_path to point at your data.
    # If using HDF5, supply dataset_name; for .avi/.mp4 leave it None.
    movie_path   = "synthetic_calcium_movie (1).mp4" # Path to movie, better if in root directory like it is here
    movie = load_movie(movie_path, "None")

    # -------------------------------
    # 2) Compute mean projection
    # -------------------------------
    # calib_frames: number of frames to average (reduce if memory‐bound)
    # chunk_size:   frames read per chunk (increase to lower RAM usage)
    mean_img = compute_mean_projection(
        movie,
        calib_frames=5400, # Average all the frames
        chunk_size=200    # How many frames to read at once
    )

    # -------------------------------
    # 3) Threshold & extract ROIs
    # -------------------------------
    # min_area: minimum pixel count to keep a mask (raise to drop
    # tiny specks; lower to catch very small cells)
    masks, sizes = denoise_and_threshold_gpu(
        mean_img,
        gauss_ksize=(5,5), # Kernel size in pixels, higher if image is grainy, lower for high SNR
        # Wider kernel is more smoothing (risks merging small cells)
        gauss_sigma=1.5, # blur strengths, std-dev of the gaussian distribution, if 5x5 kernel use 1-2
        min_area=60, # Throw away cells smaller than 60 pixels area
        # Maybe can alter algorithm to run for different min-areas after you study the data
        # or change it for each set of frames
        max_area=300 # try 50–200 depending on typical cell size
        # Any component bigger than this will be split into more rois using watershed
    )
    # Relationship between radius of cells, gauss sigma, and kernel size
    # radius = 3 * sigma
    # ksize = (2 * radius) + 1
    # So here it should be 11,11 for the kernel size instead but we don't know the average radius of the cells
    # We can adjust it accordingly or devise an algorithm that does that
    
    print(f"Detected {len(masks)} ROIs") # print how many rois found

    # -------------------------------
    # 4) Save your ROI masks
    # -------------------------------
    # output: path for your compressed masks + sizes
    output_path = "rois.npz" # Used for future operations, only changes if movie changes by running main.py again
    save_rois(masks, sizes, output_path)
    print(f"Saved ROI masks and sizes to {output_path}")
    time_end = time.time() # End timer
    print(f"ROI discovery pipeline completed in {time_end - time_start:.2f} seconds")

if __name__ == '__main__':
    main()