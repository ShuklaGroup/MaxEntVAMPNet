import numpy as np
import mdtraj as md
import openmm as mm
import dill as pickle
from matplotlib import pyplot as plt
import seaborn as sbn
from simtk.unit import *


def get_openmm_positions(state):
    """Converts a state dictionary into OpenMM-compatible positions for a simulation.
    """
    fname, frame_idx, top_file = state["fname"], state["frame_idx"], state["top_file"]
    xyz = md.load(fname, top=top_file).xyz[frame_idx]

    return Quantity(value=[mm.vec3.Vec3(*atom_pos) for atom_pos in xyz],
                    unit=nanometer)


def load_pickle(filename):
    with open(filename, 'rb') as infile:
        return pickle.load(infile)


def plot_free_e_heatmap(trajs, save_name='', n_bins=100, cscale=[0, 3]):
    hist, _, _ = np.histogram2d(trajs[:, 0], trajs[:, 1], density=True, bins=n_bins)
    free_e = -np.log(hist)
    free_e -= np.min(free_e)
    sbn.heatmap(np.rot90(free_e, 1), vmin=cscale[0], vmax=cscale[1])
    plt.xticks(np.arange(0, n_bins + 1, (n_bins + 1) // 8))
    plt.yticks(np.arange(0, n_bins + 1, (n_bins + 1) // 8)[::-1])
    if save_name:
        plt.savefig(save_name, dpi=150)
    plt.show()
