import numpy as np

dt = 0.01              # Time step
t_cue = 1.0            # Duration of cue presentation
t_delay = 8.0          # Duration of delay period between cue and decision
cue_scale = 1.0        # How strong the cuelus is from the visual system
misperceive = 0.1      # ???
time_scale = 0.4       # ???
steps = np.arange(750) # Steps to use
noise_wm = 0.005       # Standard deviation of white noise added to WM
noise_decision = 0.005 # Standard deviation of white noise added to decision
neurons_decide = 100   # Number of neurons for decision
neurons_inputs = 100   # Number of neurons for inputs ensemble
neurons_wm = 100       # Number of neurons for working memory ensemble
tau_wm = 0.1           # Synapse on recurrent connection in wm
tau = 0.01             # Synaptic time constant between ensembles
