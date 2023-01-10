"""
# Villin test
- System: villin (PDB ID: 1YRF)  in implicit solvent (Gbn2)
- Initial configuration: from PDB
- CVs: all alpha carbon distances
- Reduced dimensions (if applicable): 8 (softmax output)

Check the code below for settings.
"""

import os
import mdtraj as md
import numpy as np

from Simulation import ImplicitSim

# Define where the job is going to run
root = os.getcwd()

# Define simulation system (simulation details hidden under the hood)
system = ImplicitSim("../villin.pdb", platform="CPU")

# Define features
indices = np.asarray([ [i, j] for i in range(35) for j in range(i+3, 35) ])
features = [ lambda x, i=idx: md.compute_contacts(x, scheme='ca', contacts=[i])[0].flatten() for idx in indices ]

# Define some settings
tstep = 2e-15  # 2 fs is the default timestep
traj_len = int(10e-9 / tstep)  # 10 ns per individual trajectory
save_rate = int(traj_len / 1e4)  # Save 10000 frames per traj
trajs_per_round = 10  # 10 trajectories per round
num_rounds = 50  # 50*10*10 ns = 5 us total simulated time
ndim = 8  # 8 states
lagtime = 100  # 10*save_rate*tstep = 100 ps
n_agents = 1  # single agent (not used)

# Define initial state
init_states = [
    dict(fname="../villin.pdb",
         frame_idx=0,
         top_file=system.top_file,
         agent_idx=0), # Ignored
]