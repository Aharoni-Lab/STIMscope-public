# This Actor is the simulation of the offline initialization phase 
# This is where we flood the neurons with light and capture their fl signals
# We then take this as a recording and process it through a segmentation algo like Cellpose, OTSU, Suite2P
# This tells every actor where neurons are and what pixels they occupy
# This allows us to later choose what neuron we want to record from and what neurons we want to stimulate
# There are many ways to store or process a binary mask, and there this system uses binary masks in different contexts
# So sometimes we need a mask for projection, sometimes for recording, sometimes for generating the final mask of CS

# First need to load or generate the neuron positions

# Then need to load or generate the neuron footprints

# Then need to build a global imaging mask - this is from offline setup

# Build the MaskGenerator - this is the template the CS stimulation matrix will take, we need to configure it appropriatley

# Then we need to store this in the improv Redis database under their corresponding keys

# Signal that setup is complete
