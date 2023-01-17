# MaxEntVAMPNet

Codes for [*Active Learning of the Conformational Ensemble of Proteins using Maximum Entropy VAMPNets*](https://www.biorxiv.org/content/10.1101/2023.01.12.523801v1.full).

## Citation

If using the code in this repository, please include the following in your citations:

```bibtex
@article{kleiman2023active,
  title={Active Learning of the Conformational Ensemble of Proteins using Maximum Entropy VAMPNets},
  author={Kleiman, Diego E and Shukla, Diwakar},
  journal={bioRxiv},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```

## Environment

The `clean_env.yml` file can be used with `conda` to recreate the environment used while conducting the research.

`conda env create -f clean_env.yml`

## Description

This repository contains codes implementing different adaptive sampling methods (`AdaptiveSampling.py`, `EntropyBased.py`, `Reap.py`, `VaeReap.py`, `VampNetReap.py`, `VampReap.py`) and supporting objects or methods (`Agent.py`, `Simulations.py`, `utils.py`).

The `examples` directory contains usage examples to run molecular dynamics simulations. See the README in the directory for more details. Check the `Short example` down below for a general idea of how to use the codes.

## Methods

There are many different adaptive sampling methods implemented in this repository as Python classes. Check the table below for a quick summary of the main ones. The kinetic models correspond to those implemented in [deeptime](https://deeptime-ml.github.io/latest/index.html). 

| Name  | Selection strategy | Kinetic model | File | Main references |
| ----- | ------------------ | ------------- | ---- | --------- |
| LeastCounts  | LeastCounts  | None | `AdaptiveSampling.py` | Bowman et al. J. Chem. Theory Comput. 2010, 6, 3, 787–794.  |
| VampLeastCounts  | LeastCounts  | VAMP | `AdaptiveSampling.py` | Bowman et al. J. Chem. Theory Comput. 2010, 6, 3, 787–794.<br />Wu et al. J. Nonlinear Sci. 2019, 30, 23–66.|
| VampNetLeastCounts  | LeastCounts  | VAMPNet | `AdaptiveSampling.py` | Bowman et al. J. Chem. Theory Comput. 2010, 6, 3, 787–794.<br />Mardt et al. Nat. Commun. 2018, 9.|
| VaeLeastCounts  | LeastCounts  | TVAE | `AdaptiveSampling.py` | Bowman et al. J. Chem. Theory Comput. 2010, 6, 3, 787–794.<br />Wehmeyer et al. Chem. Phys. 2018, 148, 241703.|
| MultiagentReap  | MA REAP  | None | `Reap.py` | Kleiman et al. J. Chem. Theory Comput. 2022, 18, 9, 5422–5434.<br />|
| MultiagentVampReap  | MA REAP  | VAMP | `VampReap.py` | Kleiman et al. J. Chem. Theory Comput. 2022, 18, 9, 5422–5434.<br />Wu et al. J. Nonlinear Sci. 2019, 30, 23–66.|
| MultiagentVampNetReap  | MA REAP  | VAMPNet | `VampNetReap.py` | Kleiman et al. J. Chem. Theory Comput. 2022, 18, 9, 5422–5434.<br />Mardt et al. Nat. Commun. 2018, 9.|
| MultiagentVaeReap  | MA REAP  | TVAE | `VaeReap.py` | Kleiman et al. J. Chem. Theory Comput. 2022, 18, 9, 5422–5434.<br />Wehmeyer et al. Chem. Phys. 2018, 148, 241703.|
| EntropyBasedSampling  | MaxEnt  | VAMPNet | `EntropyBased.py` | Kleiman et al. bioRxiv. 2023, 10.1101/2023.01.12.523801.<br />Mardt et al. Nat. Commun. 2018, 9.|

# Short example

The Python classes offered here provide functionalities that correspond to different aspects of executing an adaptive sampling run. 

A `Simulation` object packages the necessary information of the system to run the simulations. **Currently it only works with OpenMM and the trajectories are performed serially (intended for testing purposes)**. You can override the methods in the corresponding class or contribute your own to add parallelization or use a different MD engine.

The `Agent` object provides the scoring method used to rank structures for seeding of new trajectories.

The `FileHandler` object organizes the files that will be created by the run. The user should not need to interact with this class.

Finally, the `AdaptiveSampling` class implements the selection strategy and takes an instance of the kinetic model to be trained.

The following is a minimal example (may need some small modifications) for running 10 rounds of simulation of the villin headpiece protein in implicit solvent using the MaxEnt method.

```python
import numpy as np
import mdtraj as md
import torch.nn as nn
from deeptime.util.torch import MLP
from Simulation import ImplicitSim
from EntropyBased import EntropyBasedSampling

# Define simulation system (simulation details hidden under the hood)
system = ImplicitSim("../villin.pdb", platform="CPU")

# Define features --> All alpha C pairwise distances (residues separated by 3 amino acids or more)
indices = np.asarray([ [i, j] for i in range(35) for j in range(i+3, 35) ])
features = [ lambda x, i=idx: md.compute_contacts(x, scheme='ca', contacts=[i])[0].flatten() for idx in indices ]

# Define some settings
tstep = 2e-15  # 2 fs is the default timestep --> Used to calculate traj_len
traj_len = int(10e-9 / tstep)  # 10 ns per individual trajectory
save_rate = int(traj_len / 1e4)  # Save 10000 frames per traj
trajs_per_round = 10  # 10 trajectories per round
num_rounds = 10  # 10*10*10 ns = 1 us total simulated time
ndim = 8  # 8 output states
lagtime = 100  # 100*save_rate*tstep = 10 ps

# Define initial state
init_states = [
    dict(fname="../villin.pdb",
         frame_idx=0,
         top_file=system.top_file,
         agent_idx=0), # Only single agent for MaxEnt
]



# Silence deprecation warning
def softmax():
    return nn.Softmax(dim=0)

vnet_lobe = MLP(units=[len(features), 512, 256, 128, 64, 32, 16, ndim], nonlinearity=nn.ReLU, output_nonlinearity=softmax)

# Initialize adaptive sampling object

adaptive_run = EntropyBasedSampling(system=system,
                                    root="./",
                                    basename="villin",
                                    save_rate=save_rate,
                                    features=features,
                                    save_info=True,
                                    lagtime=lagtime,
                                    device='cpu',
                                    vnet_batch_size=1024,
                                    vnet_num_threads=8,
                                    vnet_lobe=vnet_lobe,
                                    vnet_output_states=ndim)  # Using default values for some parameters
# Obtain initial data
adaptive_run.collect_initial_data(init_states, n_steps=traj_len, n_repeats=1)

# Run adaptive sampling runs
for i in range(num_rounds):
  adaptive_run.run_round(n_select=trajs_per_round, n_steps=traj_len, n_repeats=1)
```

