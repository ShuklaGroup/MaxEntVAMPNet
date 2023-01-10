import sys
import os
from tqdm import tqdm

sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append("/path/to/repository/files/")

from EntropyBased import EntropyBasedSampling
from settings import *

from glob import glob
from natsort import natsorted

import dill as pickle

replicate = int(sys.argv[1])
root = os.path.join(root, "replicate_{}".format(replicate))

checkpoint_files = natsorted(glob(os.path.join(root, "checkpoint_*")))

if not checkpoint_files:
    print('No checkpoints found, start new run.')

    # Initialize adaptive sampling object
    adaptive_run = EntropyBasedSampling(system=system,
                                        root=root,
                                        basename="pentapeptide",
                                        save_rate=save_rate,
                                        features=features,
                                        save_info=True,
                                        vnet_batch_size=vnet_batch_size,
                                        vnet_num_threads=vnet_num_threads, )  # Using default values for some parameters

    # Obtain initial data
    restart_round = 0  # restart from 0
    adaptive_run.collect_initial_data(init_states, n_steps=traj_len, n_repeats=1)
else:
    with open(checkpoint_files[-1], 'rb') as infile:
        adaptive_run = pickle.load(infile)
    restart_round = int(checkpoint_files[-1].split("_")[-1])
    print('Available checkpoints:', checkpoint_files)
    print('Restart from:', restart_round)

# Run simulations
checkpoint = 10  # Save checkpoint every 10 rounds

for i in tqdm(range(restart_round, num_rounds)):
    checkpoint_fname = os.path.join(root, "checkpoint_{}".format(i))
    if ((i + 1) % checkpoint == 0) and (not os.path.exists(checkpoint_fname)):
        adaptive_run.save(checkpoint_fname)
    adaptive_run.run_round(n_select=trajs_per_round, n_steps=traj_len, n_repeats=1)
