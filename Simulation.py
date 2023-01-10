"""Definition of the Simulation object for adaptive sampling implementations.
A Simulation instance is passed to an AdaptiveSampling instance to provide the simulation methods.

Simulation contains the information about the system. This includes all the information required for OpenMM to launch a
trajectory.
"""

from abc import ABC, abstractmethod
import numpy as np
import openmm as mm
import openmm.app as app
from simtk.unit import *
import mdtraj as md
import torch


class Simulation(ABC):
    """Abstract simulation class.
    """
    def __init__(self):
        pass

    @abstractmethod
    def _set_system(self):
        """_set_system prepares the system before a trajectory run.
        """
        pass

    @abstractmethod
    def launch_trajectory(self):
        """_launch_trajectory executes a trajectory. OpenMM reporters should be set here as well.
        """
        pass

    @abstractmethod
    def project_cvs(self):
        """project_cvs projects trajectories onto the desired space for analysis.
        """
        pass


class InVacuoSim(Simulation):
    """Simulation object for OpenMM simulations in vacuum.
    """

    def __init__(self, top_file, forcefield="amber14/protein.ff14SB.xml", temp=300 * kelvin, press=1 * bar,
                 nonbondedMethod=app.NoCutoff, constraints=app.HBonds, collision_freq=1 / picosecond,
                 timestep=0.002 * picosecond, platform="CUDA"):
        """Constructor for base InVacuoSim class.

        :param top_file: str.
            Topology file.
        :param forcefield: str, default = 'amber14/protein.ff14SB.xml'
            Path to .xml file that specifies an OpenMM compatible forcefield.
        :param temp: float * unit, default = 300 * kelvin.
            Temperature in OpenMM format.
        :param press: float * unit, default = 1 * bar.
            Pressure in OpenMM format. Only used for periodic simulation boxes.
        :param nonbondedMethod: see OpenMM documentation, default = openmm.app.NoCutoff
            Non-bonded interactions cutoff.
        :param constraints: see OpenMM documentation, default = openmm.app.HBonds
            Constraints.
        :param collision_freq: float / unit, default = 1 / picosecond
            Collision frequency in OpenMM format.
        :param timestep: float * unit, default = 0.002 * picosecond
            Timestep in OpenMM format.
        :param platform: str, default = "CUDA"
            See OpenMM for available options.
        """
        self.top_file = top_file
        self.forcefield = forcefield
        self.temp = temp
        self.press = press  # Ignored for this class, but may be used in derived classes
        self.nonbondedMethod = nonbondedMethod
        self.constraints = constraints
        self.collision_freq = collision_freq
        self.timestep = timestep
        if torch.cuda.is_available() and platform == "CUDA":
            print("CUDA is available and will be utilized.", flush=True)
        self.platform = platform if torch.cuda.is_available() else "CPU"

    def _set_system(self):
        """Convenience function to set the system before running a simulation. The reason this is not done in __init__()
        is that this method creates a fresh Integrator object (otherwise we get an error because the OpenMM Integrator
        is already bound to a context).

        :return: None.
        """
        pdb = app.PDBFile(self.top_file)
        forcefield = app.ForceField(self.forcefield)
        system = forcefield.createSystem(pdb.topology,
                                         nonbondedMethod=self.nonbondedMethod,
                                         constraints=self.constraints)
        integrator = mm.LangevinIntegrator(self.temp, self.collision_freq, self.timestep)
        self.system = system
        self.topology = pdb.topology
        self.integrator = integrator
        self.init_pos = pdb.positions

    def launch_trajectory(self, positions, n_steps, out_name, save_rate, velocities=None):
        """Runs an OpenMM simulation.

        :param positions: openmm.vec3.Vec3.
            Initial atomic positions for simulation.
        :param n_steps: int.
            Number of steps.
        :param out_name: str.
            Trajectory filename.
        :param save_rate: int.
            Save rate for trajectory frames.
        :param velocities: openmm.vec3.Vec3 (optional).
            Initial velocities. If not set, then they are sampled from a Boltzmann distribution.
        :return: None.
        """
        self._set_system()
        mm.Platform.getPlatformByName(self.platform)
        simulation = app.Simulation(self.topology, self.system, self.integrator)
        simulation.context.setPositions(positions)
        if velocities:
            simulation.context.setVelocities(velocities)
        simulation.reporters.append(app.DCDReporter(out_name, save_rate))
        simulation.step(n_steps)

    def project_cvs(self, features, traj_names):
        """Project the system into the desired collective variables defined in `features`.
        MDTraj functions can be used to define the features.

        :param traj_names: list[str].
            Array of paths to trajectory files.
        :param features: list[Callable].
            List of callables that take a trajectory file as input and return a real number per frame.
        :return: list[np.ndarray].
            List containing the data for each trajectory.
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
    """Simulation object for OpenMM simulations in implicit solvent.
    """
    def __init__(self, top_file, forcefield="amber14/protein.ff14SB.xml", temp=300 * kelvin, press=1 * bar,
                 nonbondedMethod=app.NoCutoff, constraints=app.HBonds, collision_freq=1 / picosecond,
                 timestep=0.002 * picosecond, platform="CUDA", implicit_solvent="implicit/gbn2.xml"):
        """Constructor for base ImplicitSim class.

        :param top_file: str.
            Topology file.
        :param forcefield: str, default = 'amber14/protein.ff14SB.xml'.
            Path to .xml file that specifies an OpenMM compatible forcefield.
        :param temp: float * unit, default = 300 * kelvin.
            Temperature in OpenMM format.
        :param press: float * unit, default = 1 * bar.
            Pressure in OpenMM format. Only used for periodic simulation boxes.
        :param nonbondedMethod: see OpenMM documentation, default = openmm.app.NoCutoff.
            Non-bonded interactions cutoff.
        :param constraints: see OpenMM documentation, default = openmm.app.HBonds.
            Constraints.
        :param collision_freq: float / unit, default = 1 / picosecond.
            Collision frequency in OpenMM format.
        :param timestep: float * unit, default = 0.002 * picosecond.
            Timestep in OpenMM format.
        :param platform: str, default = 'CUDA'.
            See OpenMM for available options.
        :param implicit_solvent: str, default = 'implicit/gbn2.xml'.
            Implicit solvent model. Must be compatible with OpenMM.
        """
        InVacuoSim.__init__(self, top_file, forcefield=forcefield, temp=temp, press=press,
                            nonbondedMethod=nonbondedMethod, constraints=constraints, collision_freq=collision_freq,
                            timestep=timestep, platform=platform)
        self.implicit_solvent = implicit_solvent

    def _set_system(self):
        """Convenience function to set the system before running a simulation. The reason this is not done in __init__()
        is that this method creates a fresh Integrator object (otherwise we get an error because the OpenMM Integrator
        is already bound to a context).

        :return: None.
        """
        pdb = app.PDBFile(self.top_file)
        forcefield = app.ForceField(self.forcefield, self.implicit_solvent)
        system = forcefield.createSystem(pdb.topology,
                                         nonbondedMethod=self.nonbondedMethod,
                                         constraints=self.constraints)
        integrator = mm.LangevinIntegrator(self.temp, self.collision_freq, self.timestep)
        self.system = system
        self.topology = pdb.topology
        self.integrator = integrator
        self.init_pos = pdb.positions


class LangevinSim(InVacuoSim):
    """Simulation object with external custom force for Langevin dynamics simulations.
    """

    def __init__(self, top_file, force_equation, temp=300 * kelvin, collision_freq=1 / picosecond,
                 timestep=0.002 * picosecond, platform="CUDA"):
        """Constructor for base LangevinSim class.

        :param top_file: str.
            Topology file.
        :param force_equation: str.
            Analytical potential function.
        :param temp: float * unit, default = 300 * kelvin.
            Temperature in OpenMM format.
        :param press: float * unit, default = 1 * bar.
            Pressure in OpenMM format. Only used for periodic simulation boxes.
        :param nonbondedMethod: see OpenMM documentation, default = openmm.app.NoCutoff.
            Non-bonded interactions cutoff.
        :param constraints: see OpenMM documentation, default = openmm.app.HBonds.
            Constraints.
        :param collision_freq: float / unit, default = 1 / picosecond.
            Collision frequency in OpenMM format.
        :param timestep: float * unit, default = 0.002 * picosecond.
            Timestep in OpenMM format.
        :param platform: str, default = 'CUDA'.
            See OpenMM for available options.
        """
        InVacuoSim.__init__(self, top_file, temp=temp, collision_freq=collision_freq, timestep=timestep,
                            platform=platform)
        self.force_equation = force_equation

    def _set_system(self):
        """Convenience function to set the system before running a simulation. The reason this is not done in __init__()
        is that this method creates a fresh Integrator object (otherwise we get an error because the OpenMM Integrator
        is already bound to a context).

        :return: None.
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
