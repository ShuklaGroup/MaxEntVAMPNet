import sys
import os
from tqdm import tqdm
import torch.nn as nn
from deeptime.util.torch import MLP
from glob import glob
from natsort import natsorted
import dill as pickle

sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append("/path/to/repository/files/")

from AdaptiveSampling import VampNetLeastCounts
from settings import *

replicate = int(sys.argv[1])
root = os.path.join(root, "replicate_{}".format(replicate))

checkpoint_files = natsorted(glob(os.path.join(root, "checkpoint_*")))

if not checkpoint_files:
    print('No checkpoints found, start new run.')
    vnet_lobe = MLP(units=[len(features), 512, 256, 128, 64, 32, 16, ndim], nonlinearity=nn.ReLU)
    # Initialize adaptive sampling object
    adaptive_run = VampNetLeastCounts(system=system,
                                      root=root,
                                      basename="villin",
                                      save_rate=save_rate,
                                      features=features,
                                      save_info=True,
                                      lagtime=lagtime,
                                      ndim=ndim,
                                      vnet_batch_size=1024,
                                      vnet_num_threads=8,
                                      vnet_lobe=vnet_lobe)  # Using default values for some parameters
    # Obtain initial data
    restart_round = 0
    adaptive_run.collect_initial_data(init_states, n_steps=traj_len, n_repeats=1)
else:
    with open(checkpoint_files[-1], 'rb') as infile:
        adaptive_run = pickle.load(infile)
    restart_round = int(checkpoint_files[-1].split("_")[-1])
    print('Available checkpoints:', checkpoint_files)
    print('Restart from:', restart_round)

# Run simulations
checkpoint = 3 # Save checkpoint every 3 rounds

for i in tqdm(range(restart_round, num_rounds)):
    checkpoint_fname = os.path.join(root, "checkpoint_{}".format(i))
    if ((i+1) % checkpoint == 0) and (not os.path.exists(checkpoint_fname)):
        adaptive_run.save(checkpoint_fname)
    adaptive_run.run_round(n_select=trajs_per_round, n_steps=traj_len, n_repeats=1)
