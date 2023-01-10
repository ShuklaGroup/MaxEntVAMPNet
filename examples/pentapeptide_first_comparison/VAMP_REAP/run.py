import sys
import os
from tqdm import tqdm

sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append("/path/to/repository/files/")

from VampReap import MultiagentVampReap
from settings import *

replicate = int(sys.argv[1])
root = os.path.join(root, "replicate_{}".format(replicate))

# Initialize adaptive sampling object
adaptive_run = MultiagentVampReap(system=system,
                                  root=root,
                                  basename="pentapeptide",
                                  save_rate=save_rate,
                                  features=features,
                                  save_info=True,
                                  lagtime=lagtime,
                                  cv_weights=[0.5, 0.5],
                                  n_agents=2,
                                  stakes_method="equal")  # Using default values for some parameters

# Obtain initial data
adaptive_run.collect_initial_data(init_states, n_steps=traj_len, n_repeats=1)

# Run simulations
for i in tqdm(range(num_rounds)):
    adaptive_run.run_round(n_select=trajs_per_round, n_steps=traj_len, n_repeats=1)
