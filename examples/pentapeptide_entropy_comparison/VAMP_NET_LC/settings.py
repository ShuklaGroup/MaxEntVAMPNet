"""
# Pentapeptide test
- System: TRP-LEU-ALA-LEU-LEU pentapeptide in implicit solvent (Gbn2)
- Initial configurations: state_{1, 2}.pdb --> stable states from long simulation
- CVs: 4 $\phi$ dihedrals + 4 $\psi$ dihedrals
- Reduced dimensions (if applicable): 8

Check the code below for settings.
"""

import os
import mdtraj as md

from Simulation import ImplicitSim

# Define where the job is going to run
root = os.getcwd()

# Define simulation system (simulation details hidden under the hood)
system = ImplicitSim("../state_1.pdb")

# Define features

features = [
    lambda x: md.compute_phi(x, periodic=False)[1].T[0],
    lambda x: md.compute_phi(x, periodic=False)[1].T[1],
    lambda x: md.compute_phi(x, periodic=False)[1].T[2],
    lambda x: md.compute_phi(x, periodic=False)[1].T[3],
    lambda x: md.compute_psi(x, periodic=False)[1].T[0],
    lambda x: md.compute_psi(x, periodic=False)[1].T[1],
    lambda x: md.compute_psi(x, periodic=False)[1].T[2],
    lambda x: md.compute_psi(x, periodic=False)[1].T[3],
]

# Define some settings
tstep = 2e-15  # 2 fs is the default timestep
traj_len = int(2e-9 / tstep)  # 2 ns per individual trajectory --> MFPT between states varies from ~10-500 ns
save_rate = int(traj_len / 1e3)  # Save 1000 frames per traj
trajs_per_round = 5  # 5 trajectories per round
num_rounds = 60  # 5*100*2 ns = 1 us total simulated time
ndim = 8  # 2 reduced dimensions
lagtime = 10  # 10*save_rate*tstep = 20 ps
n_agents = 2  # single agent (not used)

# Some VampNet settings
vnet_batch_size = 2048
vnet_num_threads = 8

# Define initial state
init_states = [
    dict(fname="../state_1.pdb",
         frame_idx=0,
         top_file=system.top_file,
         agent_idx=0),

    dict(fname="../state_2.pdb",
         frame_idx=0,
         top_file=system.top_file,
         agent_idx=1),
]
