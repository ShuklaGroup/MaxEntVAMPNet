import sys
import os
from tqdm import tqdm

sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append("/path/to/repository/files/")

from AdaptiveSampling import VampNetLeastCounts
from settings import *

replicate = int(sys.argv[1])
root = os.path.join(root, "replicate_{}".format(replicate))

# Initialize adaptive sampling object
adaptive_run = VampNetLeastCounts(system=system,
                                  root=root,
                                  basename="pentapeptide",
                                  save_rate=save_rate,
                                  features=features,
                                  save_info=True,
                                  lagtime=lagtime,
                                  ndim=ndim,
                                  vnet_batch_size=1024,
                                  vnet_num_threads=8)  # Using default values for some parameters

# Obtain initial data
adaptive_run.collect_initial_data(init_states, n_steps=traj_len, n_repeats=1)

# Run simulations
for i in tqdm(range(num_rounds)):
    adaptive_run.run_round(n_select=trajs_per_round, n_steps=traj_len, n_repeats=1)
