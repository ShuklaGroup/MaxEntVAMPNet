"""
Definition of the Simulation object for REAP-like implementations.
A Simulation instance is passed to an AdaptiveSampling-like instance to provide the simulation methods.

Simulation contains the information about the system that will be simulated. This includes all the information required
for OpenMM to launch a trajectory.
"""

from abc import ABC, abstractmethod
import numpy as np
import openmm as mm
import openmm.app as app
from simtk.unit import *
import mdtraj as md
import torch


class Simulation(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def _set_system(self):
        """
        _set_system prepares the system before a trajectory runs.
        """
        pass

    @abstractmethod
    def launch_trajectory(self):
        """
        _launch_trajectory executes a trajectory. OpenMM reporters should be set here as well.
        """
        pass

    # @abstractmethod
    # def save_state(self):
    #     """
    #     Save a single frame to a file containing coordinates and velocities.
    #     :return:
    #     """

    # @abstractmethod
    # def spawn_trajectories(self):
    #     """
    #     spawn_trajectories runs an ensemble of trajectories from different initial conditions.
    #     """
    #     pass

    @abstractmethod
    def project_cvs(self):
        """
    project_system projects trajectories into the desired space for analysis.
        """
        pass


class InVacuoSim(Simulation):
    """
    Simulation object for OpenMM simulations in vacuum.
    """

    def __init__(self, top_file, forcefield="amber14/protein.ff14SB.xml", temp=300 * kelvin, press=1 * bar,
                 nonbondedMethod=app.NoCutoff, constraints=app.HBonds, collision_freq=1 / picosecond,
                 timestep=0.002 * picosecond, platform="CUDA"):
        self.top_file = top_file
        self.forcefield = forcefield
        self.temp = temp
        self.press = press  # Ignored for this class, but used in derived classes
        self.nonbondedMethod = nonbondedMethod
        self.constraints = constraints
        self.collision_freq = collision_freq
        self.timestep = timestep
        if torch.cuda.is_available() and platform == "CUDA":
            print("CUDA is available and will be utilized.", flush=True)
        self.platform = platform if torch.cuda.is_available() else "CPU"

    def _set_system(self):
        """
        Convenience function to set the system before running a simulation. The reason this is not done in __init__() is
        that this method creates a fresh Integrator object (otherwise we get an error because the OpenMM Integrator is
        already bound to a context).
        """
        pdb = app.PDBFile(self.top_file)
        forcefield = app.ForceField(self.forcefield)
        system = forcefield.createSystem(pdb.topology,
                                         nonbondedMethod=self.nonbondedMethod,
                                         constraints=self.constraints)
        integrator = mm.LangevinIntegrator(self.temp, self.collision_freq, self.timestep)
        # system.addForce(mm.MonteCarloBarostat(self.press, self.temp)) # Can't use this in a nonperiodic system

        self.system = system
        self.topology = pdb.topology
        self.integrator = integrator
        self.init_pos = pdb.positions

    def launch_trajectory(self, positions, n_steps, out_name, save_rate, velocities=None):
        self._set_system()
        mm.Platform.getPlatformByName(self.platform)
        simulation = app.Simulation(self.topology, self.system, self.integrator)
        simulation.context.setPositions(positions)
        if velocities:
            simulation.context.setVelocities(velocities)
        simulation.reporters.append(app.DCDReporter(out_name, save_rate))
        simulation.step(n_steps)

    def project_cvs(self, features, traj_names):
        """
        Project the system into the desired collective variables defined in `features`. MDTraj functions can be used to
        define the features.

        :param traj_names: array of paths to trajectory files.
        :type features: list[callable]
        :param features: array of callables that process a trajectory object and return the data.
        :return: list[np.ndarray] containing the data for each trajectory.
        """
        # I will assume for now that the trajectories fit in memory
        trajs = [md.load(traj_file, top=self.top_file) for traj_file in traj_names]
        data = []

        for traj in trajs:
            data_feat = np.empty((traj.n_frames, len(features)))
            for j, feature in enumerate(features):
                data_feat[:, j] = feature(traj)
            data.append(data_feat)

        return data


class ImplicitSim(InVacuoSim):
    def __init__(self, top_file, forcefield="amber14/protein.ff14SB.xml", temp=300 * kelvin, press=1 * bar,
                 nonbondedMethod=app.NoCutoff, constraints=app.HBonds, collision_freq=1 / picosecond,
                 timestep=0.002 * picosecond, platform="CUDA", implicit_solvent="implicit/gbn2.xml"):
        InVacuoSim.__init__(self, top_file, forcefield=forcefield, temp=temp, press=press,
                            nonbondedMethod=nonbondedMethod, constraints=constraints, collision_freq=collision_freq,
                            timestep=timestep, platform=platform)
        self.implicit_solvent = implicit_solvent

    def _set_system(self):
        """
        Convenience function to set the system before running a simulation. The reason this is not done in __init__() is
        that this method creates a fresh Integrator object (otherwise we get an error because the OpenMM Integrator is
        already bound to a context).
        """
        pdb = app.PDBFile(self.top_file)
        forcefield = app.ForceField(self.forcefield, self.implicit_solvent)
        system = forcefield.createSystem(pdb.topology,
                                         nonbondedMethod=self.nonbondedMethod,
                                         constraints=self.constraints)
        integrator = mm.LangevinIntegrator(self.temp, self.collision_freq, self.timestep)
        # system.addForce(mm.MonteCarloBarostat(self.press, self.temp)) # Can't use this in a nonperiodic system

        self.system = system
        self.topology = pdb.topology
        self.integrator = integrator
        self.init_pos = pdb.positions


class LangevinSim(InVacuoSim):
    """
    Simulation object with external custom force.
    """

    def __init__(self, top_file, force_equation, temp=300 * kelvin, collision_freq=1 / picosecond,
                 timestep=0.002 * picosecond, platform="CUDA"):
        InVacuoSim.__init__(self, top_file, temp=temp, collision_freq=collision_freq, timestep=timestep,
                            platform=platform)
        self.force_equation = force_equation

    def _set_system(self):
        """
        Convenience function to set the system before running a simulation. The reason this is not done in __init__() is
        that this method creates a fresh Integrator object (otherwise we get an error because the OpenMM Integrator is
        already bound to a context).
        """
        pdb = app.PDBFile(self.top_file)  # Dummy topology
        system = mm.System()
        system.addParticle(100)  # Add particle with mass of 100 amu
        force = mm.CustomExternalForce(self.force_equation)  # Defines the potential
        force.addParticle(0, [])
        system.addForce(force)
        integrator = mm.LangevinIntegrator(self.temp, self.collision_freq, self.timestep)

        self.system = system
        self.topology = pdb.topology
        self.integrator = integrator
        self.init_pos = pdb.positions

# class CustomSDESim(Simulation):
#     """
#     Simulation object based on deeptime.data.custom_sde.
#     """
#
#     def __init__(self, top_file, force_equation, temp=300 * kelvin, collision_freq=1 / picosecond,
#                  timestep=0.002 * picosecond, platform="CUDA"):
#         InVacuoSim.__init__(self, top_file, temp=temp, collision_freq=collision_freq, timestep=timestep,
#                             platform=platform)
#         self.force_equation = force_equation
#
#     def _set_system(self):
#         """
#         Convenience function to set the system before running a simulation. The reason this is not done in __init__() is
#         that this method creates a fresh Integrator object (otherwise we get an error because the OpenMM Integrator is
#         already bound to a context).
#         """
#         pdb = app.PDBFile(self.top_file)  # Dummy topology
#         system = mm.System()
#         system.addParticle(100)  # Add particle with mass of 100 amu
#         force = mm.CustomExternalForce(self.force_equation)  # Defines the potential
#         force.addParticle(0, [])
#         system.addForce(force)
#         integrator = mm.LangevinIntegrator(self.temp, self.collision_freq, self.timestep)
#
#         self.system = system
#         self.topology = pdb.topology
#         self.integrator = integrator
#         self.init_pos = pdb.positions
