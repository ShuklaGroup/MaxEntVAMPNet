import sys
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from deeptime.util.torch import MLP


sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append("/path/to/repository/files/")

from AdaptiveSampling import VampNetLeastCounts
from settings import *

replicate = int(sys.argv[1])
root = os.path.join(root, "replicate_{}".format(replicate))

vnet_lobe = MLP(units=[len(features), 16, 32, 64, 128, 256, 128, 64, 32, 16, ndim], nonlinearity=nn.ReLU)
assert (len(features) == ndim == 8)

# Initialize adaptive sampling object
adaptive_run = VampNetLeastCounts(system=system,
                                  root=root,
                                  basename="pentapeptide",
                                  save_rate=save_rate,
                                  features=features,
                                  save_info=True,
                                  lagtime=lagtime,
                                  ndim=ndim,
                                  vnet_batch_size=vnet_batch_size,
                                  vnet_num_threads=vnet_num_threads,
                                  vnet_lobe=vnet_lobe)  # Using default values for some parameters

# Obtain initial data
adaptive_run.collect_initial_data(init_states, n_steps=traj_len, n_repeats=1)

# Run simulations
checkpoint = 10 # Save checkpoint every 10 rounds
for i in tqdm(range(num_rounds)):
    if (i+1) % checkpoint == 0:
        adaptive_run.save(os.path.join(root, "checkpoint_{}".format(i)))
    adaptive_run.run_round(n_select=trajs_per_round, n_steps=traj_len, n_repeats=1)
