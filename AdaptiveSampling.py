"""Definition of the base AdaptiveSampling class for REAP-like implementations.
Derived classes that implement more advanced adaptive sampling schemes will inherit from this class.
"""
import os
from abc import ABC, abstractmethod
from collections import Counter

import dill as pickle
import numpy as np
import torch
import torch.nn as nn
from deeptime.clustering import RegularSpace
from deeptime.decomposition import VAMP
from deeptime.decomposition.deep import VAMPNet, TVAEEncoder, TVAE
from deeptime.util.data import TrajectoryDataset
from deeptime.util.torch import MLP
from sklearn.cluster import MiniBatchKMeans, BisectingKMeans
from sklearn.metrics import pairwise_distances_argmin_min
from torch.utils.data import DataLoader

import utils
from FileHandler import FileHandler


class AdaptiveSampling(ABC):
    """Abstract class for adaptive sampling object.
    """

    def __init__(self):
        pass

    @abstractmethod
    def _reload(self):
        pass

    @abstractmethod
    def define_cvs(self):
        pass

    @abstractmethod
    def collect_initial_data(self):
        pass

    @abstractmethod
    def _cluster(self):
        pass

    @abstractmethod
    def _select_states(self):
        pass

    @abstractmethod
    def run_round(self):
        pass


class LeastCounts(AdaptiveSampling):
    """This class implements Least Counts adaptive sampling using KMeans clustering.
    """

    def __init__(self,
                 system=None,
                 root="",
                 basename="",
                 save_format='.dcd',
                 save_rate=100,
                 features=None,
                 save_info=False,
                 cluster_args=None,
                 log_file=None):
        """LeastCounts constructor.

        :param system: Simulation object.
            Object that implements the dynamics to be simulated.
        :param root: str.
            Path to root directory where data will be saved.
        :param basename: str.
            Basename for saved trajectory files.
        :param save_format: str, default = ".dcd".
            Saved format to use for trajectories (not implemented yet).
        :param save_rate: int, default = 100.
            Save rate in frames for trajectory files.
        :param features: list[Callable].
            List of callables that take a trajectory file as input and return a real number per frame.
        :param save_info: Bool, default = False.
            Save logging info for each trajectory run.
        :param cluster_args: list[float, float, int].
            List of parameters for MiniBatchKMeans clustering (gamma, b, batch_size). Gamma and b affect the number of
            clusters utilized, while batch_size is the number of samples to use per mini-batch.
        :param log_file: str.
            Path to log_file. Passing this argument will supersede all other parameters.
        """
        self.system = system
        self.fhandler = FileHandler(root, basename, save_format)
        self.save_rate = save_rate
        self.n_round = 0
        self.features = features
        self.n_features = len(features)
        self.data = None
        self.concat_data = None
        self.states = None
        self._cached_trajs = set()  # Set of trajectory files already read into memory
        self._cached_trajs_ordered = []
        self._cluster_object = None
        self.save_info = save_info  # Used to output information about the run
        if cluster_args is None:
            self.cluster_args = [2e-3, 0.7, 10000]  # Depends on clustering algorithm --> This is (gamma, b, batch_size)
        else:
            self.cluster_args = cluster_args
        if isinstance(log_file, str):
            self._reload(log_file)

    def save(self, path):
        """Saves the object serialized with dill (similar to pickle).
        Warning: the object might take a large amount of memory.

        :param path: str.
            Path to output file.
        :return: None.
        """
        with open(path, "wb") as outfile:
            pickle.dump(self, outfile)

    def _save_logs(self, filename):
        """Save logging information of the run.

        :param filename: str.
            Name of output file.
        :return: None.
        """
        logs = dict(
            system=self.system,
            fhandler=self.fhandler,
            save_rate=self.save_rate,
            n_round=self.n_round,
            features=self.features,
            n_features=self.n_features,
            cluster_args=self.cluster_args,
            cluster_object=self._cluster_object,
            states=self.states,
        )
        root_dir = self.fhandler.root
        path = os.path.join(root_dir, filename)
        with open(path, "wb") as outfile:
            pickle.dump(logs, outfile)

    def _reload(self, log_file):
        """Reset simulation object to state in log_file. Called in constructor.

        For this method to work, trajectories should be found where self.fhandler expects them.

        :param log_file: str.
            Path to log file.
        """
        with open(log_file, "rb") as infile:
            logs = pickle.load(infile)
        self.system = logs['system']
        self.fhandler = logs['fhandler']
        self.save_rate = logs['save_rate']
        self.n_round = logs['n_round']
        self.features = logs['features']
        self.n_features = logs['n_features']
        self.cluster_args = logs['cluster_args']
        self._cluster_object = logs['cluster_object']
        self.states = logs['states']
        self.save_info = True  # If loading from log file, assume that log files are required
        self.data = None
        self.concat_data = None
        self._cached_trajs = set()
        self._cached_trajs_ordered = []
        self._cluster_object = None
        self._update_data()

    # TODO: Define subsample function to use every time the total number of frames surpasses threshold.

    def define_cvs(self, features):
        """Use this function to redefine collective variables after a run has been started.

        :param features: list[Callable].
            List of callables that take a trajectory file as input and return a real number per frame.
        """
        self.features = features

    def collect_initial_data(self, init_states, n_steps, n_repeats=1):
        """Typically called when no preexisting data is available.
        Calling this function will update n_round.

        :param init_states: list[tuple(str, int, str)].
            Order for tuple elements in list: [(fname, frame_idx, top_file)].
        :param n_steps: int.
            Number of steps per trajectory.
        :param n_repeats: int.
            Number of trajectories.
        :return: None.
        """
        fnames = self.fhandler.generate_new_round(init_states, n_repeats, n_steps, self.n_round, self.save_rate)
        for fname in fnames:
            state = self.fhandler.init_states[fname]  # State is dict (see utils.py)
            positions = utils.get_openmm_positions(state)
            self.system.launch_trajectory(positions, n_steps, fname, self.save_rate)
        self.fhandler.check_files(self.system.top_file)
        self._update_data()  # Update data from new trajectories
        self.n_round += 1

    def _update_data(self):
        """Update data with newly saved trajectories.

        :return: None.
        """
        fnames = self.fhandler.list_all_files()
        new_fnames = []
        for fn in fnames:
            if fn not in self._cached_trajs:
                new_fnames.append(fn)
        if self.data is None:
            self.data = self.system.project_cvs(self.features, new_fnames)
        else:
            new_data = self.system.project_cvs(self.features, new_fnames)
            self.data.extend(new_data)
        self.concat_data = np.concatenate(self.data, axis=0)
        self._cached_trajs.update(new_fnames)
        self._cached_trajs_ordered.extend(new_fnames)

    def _cluster(self, min_clusters):
        """Perform clustering using MiniBatchKMeans.

        :param min_clusters: int.
            Minimum number of clusters required.
        :return: None.
        """
        total_frames = self.concat_data.shape[0]
        b, gamma, max_frames = self.clustering_args
        d = min(len(self.features), 3)
        n_clusters = int(b * (min(total_frames, max_frames) ** (gamma * d)))
        if n_clusters < min_clusters:
            n_clusters = min_clusters
        elif n_clusters > 1000:  # Set a limit to avoid slow clustering
            n_clusters = 1000
        elif n_clusters > total_frames:
            n_clusters = total_frames // 3  # In case of low data regime
        self._cluster_object = \
            MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', batch_size=max_frames).fit(self.concat_data)

    def _find_candidates(self, n_select):
        """Determine a representative simulation frame for each cluster.

        :param n_select: int.
            Maximum number of clusters to consider.
        :return: least_counts (np.ndarray), central_frames_indices (np.ndarray)
            least_counts: coordinates in collective variable space for representative frames.
            central_frames_indices: indices for representative frames.
        """
        self._cluster(n_select)
        counts = Counter(self._cluster_object.labels_)
        least_counts = np.asarray(counts.most_common()[::-1][:n_select])[:, 0]
        least_counts_centers = self._cluster_object.cluster_centers_[least_counts]
        central_frames_indices, _ = pairwise_distances_argmin_min(least_counts_centers,
                                                                  self.concat_data)

        return least_counts, central_frames_indices

    def _select_states(self, n_select):
        """Select states to restart simulations.

        :param n_select: int.
            Number of states to select.
        :return: starting_states_info (list[dict]).
            List containing information to read a state (fname, frame_idx, and top_file).
        """
        least_counts, central_frames_indices = self._find_candidates(n_select)
        starting_states = central_frames_indices  # In LeastCounts, no further sub-selection is performed
        starting_states_info = self.fhandler.find_state_info(starting_states,
                                                             self._cached_trajs_ordered,
                                                             self.system.top_file)

        return starting_states_info

    def run_round(self, n_select, n_steps, n_repeats=1):
        """Perform round of simulations.

        :param n_select: int.
            Number of trajectories to run.
        :param n_steps: int.
            Number of steps per trajectory.
        :param n_repeats: int.
            Number of clones per trajectory.
        :return: None.
        """
        starting_states_info = self._select_states(n_select)
        fnames = self.fhandler.generate_new_round(starting_states_info,
                                                  n_repeats,
                                                  n_steps,
                                                  self.n_round,
                                                  self.save_rate)
        for fname in fnames:
            state = self.fhandler.init_states[fname]
            # State is a dictionary that contains at least keys "fname", "frame_idx", and "top_file"
            positions = utils.get_openmm_positions(state)
            self.system.launch_trajectory(positions, n_steps, fname, self.save_rate)
        self.fhandler.check_files(self.system.top_file)
        self._update_data()  # Update data from new trajectories
        if self.save_info:
            self._save_logs("logs_round_{}.pkl".format(self.n_round))
        self.n_round += 1


class LeastCountsBis(LeastCounts):
    """This class implements Least Counts adaptive sampling using BisectingKMeans clustering.
    """

    def _cluster(self, min_clusters):
        """Perform clustering using BisectingKMeans.

        :param min_clusters: int.
            Minimum number of clusters required.
        :return: None.
        """
        total_frames = self.concat_data.shape[0]
        b, gamma, max_frames = self.clustering_args
        d = min(len(self.features), 3)
        n_clusters = int(b * (min(total_frames, max_frames) ** (gamma * d)))
        if n_clusters < min_clusters:
            n_clusters = min_clusters
        elif n_clusters > 1000:  # Set a limit to avoid slow clustering
            n_clusters = 1000
        elif n_clusters > total_frames:
            n_clusters = total_frames // 3  # In case of low data regime
        self._cluster_object = \
            BisectingKMeans(n_clusters=n_clusters,
                            init="k-means++",
                            bisecting_strategy="largest_cluster").fit(self.concat_data)


class LeastCountsRegSpace(LeastCounts):
    """This class implements Least Counts adaptive sampling using RegularSpace clustering.
    """

    def __init__(self,
                 system=None,
                 root="",
                 basename="",
                 save_format='.dcd',
                 save_rate=100,
                 features=None,
                 save_info=False,
                 cluster_args=None,
                 log_file=None):
        """Constructor for LeastCountsRegSpace.

        :param system: Simulation object.
            Object that implements the dynamics to be simulated.
        :param root: str.
            Path to root directory where data will be saved.
        :param basename: str.
            Basename for saved trajectory files.
        :param save_format: str, default = ".dcd".
            Saved format to use for trajectories (not implemented yet).
        :param save_rate: int, default = 100.
            Save rate in frames for trajectory files.
        :param features: list[Callable].
            List of callables that take a trajectory file as input and return a real number per frame.
        :param save_info: Bool, default = False.
            Save logging info for each trajectory run.
        :param cluster_args: list[float, int].
            List of parameters for RegularSpace clustering (dmin, max_centers). dmin is the minimum distance admissible
            between two centers. max_centers is the maximum number of clusters that can be created.
        :param log_file: str.
            Path to log_file. Passing this argument will supersede all other parameters.
        """
        LeastCounts.__init__(self,
                             system=system,
                             root=root,
                             basename=basename,
                             save_format=save_format,
                             save_rate=save_rate,
                             features=features,
                             save_info=save_info,
                             log_file=log_file)
        if log_file is None:
            if cluster_args is None:
                self.cluster_args = [1e-3, 10000]  # Depends on clustering algorithm --> This is (dmin, max_centers)
            else:
                self.cluster_args = cluster_args

    def _cluster(self, min_clusters):
        """Perform clustering using BisectingKMeans.

        :param min_clusters: int.
            Minimum number of clusters required.
        :return: None.
        """
        dmin, max_centers = self.cluster_args
        self._cluster_object = \
            RegularSpace(dmin=dmin,
                         max_centers=max_centers).fit(self.concat_data)

    def _find_candidates(self, n_select):
        """Determine a representative simulation frame for each cluster.

        :param n_select: int.
            Maximum number of clusters to consider.
        :return: least_counts (np.ndarray), central_frames_indices (np.ndarray)
            least_counts: coordinates in collective variable space for representative frames.
            central_frames_indices: indices for representative frames.
        """
        self._cluster(n_select)
        counts = Counter(self._cluster_object.model.transform(self.concat_data))
        least_counts = np.asarray(counts.most_common()[::-1][:n_select])[:, 0]
        cluster_centers = np.squeeze(np.asarray(self._cluster_object._clustercenters))
        least_counts_centers = cluster_centers[least_counts]
        central_frames_indices, _ = pairwise_distances_argmin_min(least_counts_centers,
                                                                  self.concat_data)
        return least_counts, central_frames_indices


class VampLeastCounts(LeastCountsRegSpace):
    """This class implements Least Counts adaptive sampling using RegularSpace clustering in combination with VAMP
    from the deeptime package.
    """

    def __init__(self,
                 system=None,
                 root="",
                 basename="",
                 save_format='.dcd',
                 save_rate=100,
                 features=None,
                 save_info=False,
                 cluster_args=None,
                 ndim=2,
                 lagtime=1,
                 propagation_steps=1,
                 log_file=None):
        """Constructor for VampLeastCounts.

        :param system: Simulation object.
            Object that implements the dynamics to be simulated.
        :param root: str.
            Path to root directory where data will be saved.
        :param basename: str.
            Basename for saved trajectory files.
        :param save_format: str, default = ".dcd".
            Saved format to use for trajectories (not implemented yet).
        :param save_rate: int, default = 100.
            Save rate in frames for trajectory files.
        :param features: list[Callable].
            List of callables that take a trajectory file as input and return a real number per frame.
        :param save_info: Bool, default = False.
            Save logging info for each trajectory run.
        :param cluster_args: list[float, int].
            List of parameters for RegularSpace clustering (dmin, max_centers). dmin is the minimum distance admissible
            between two centers. max_centers is the maximum number of clusters that can be created.
        :param ndim: int, default = 2.
            Number of reduction dimensions to use with VAMP.
        :param lagtime: int, default = 1.
            Lag time expressed as number of frames.
        :param propagation_steps: int, default = 1.
            This option allows to apply the Koopman operator propagation_steps times to the input conformation.
        :param log_file: str.
            Path to log_file. Passing this argument will supersede all other parameters.
        """
        LeastCountsRegSpace.__init__(self,
                                     system=system,
                                     root=root,
                                     basename=basename,
                                     save_format=save_format,
                                     save_rate=save_rate,
                                     features=features,
                                     save_info=save_info,
                                     cluster_args=cluster_args
                                     )
        if log_file is None:
            self.ndim = ndim
            self.lagtime = lagtime
            self.propagation_steps = propagation_steps
            self.estimator = VAMP(lagtime=self.lagtime, dim=self.ndim, epsilon=1e-18)  # Epsilon is set to a low
            # number to prevent errors, but this may result in poor quality models.
        elif isinstance(log_file, str):
            self._reload(log_file)

    def _save_logs(self, filename):
        """Save logging information of the run.

        :param filename: str.
            Name of output file.
        :return: None.
        """
        logs = dict(
            system=self.system,
            fhandler=self.fhandler,
            save_rate=self.save_rate,
            n_round=self.n_round,
            features=self.features,
            n_features=self.n_features,
            cluster_args=self.cluster_args,
            cluster_object=self._cluster_object,
            states=self.states,
            ndim=self.ndim,
            lagtime=self.lagtime,
            propagation_steps=self.propagation_steps,
            estimator=self.estimator,
        )
        root_dir = self.fhandler.root
        path = os.path.join(root_dir, filename)
        with open(path, "wb") as outfile:
            pickle.dump(logs, outfile)

    def _reload(self, log_file):
        """Reset simulation object to state in log_file. Called in constructor.

        For this method to work, trajectories should be found where self.fhandler expects them.

        :param log_file: str.
            Path to log file.
        """
        with open(log_file, "rb") as infile:
            logs = pickle.load(infile)
        self.system = logs['system']
        self.fhandler = logs['fhandler']
        self.save_rate = logs['save_rate']
        self.n_round = logs['n_round']
        self.features = logs['features']
        self.n_features = logs['n_features']
        self.cluster_args = logs['cluster_args']
        self._cluster_object = logs['cluster_object']
        self.states = logs['states']
        self.ndim = logs['ndim']
        self.lagtime = logs['lagtime']
        self.propagation_steps = logs['propagation_steps']
        self.estimator = logs['estimator']
        self.save_info = True  # If loading from log file, assume that log files are required
        self.data = None
        self.concat_data = None
        self._cached_trajs = set()
        self._cached_trajs_ordered = []
        self._cluster_object = None
        self._update_data()

    def _update_data(self):
        """Update data with newly saved trajectories.

        :return: None.
        """
        LeastCountsRegSpace._update_data(self)
        self.estimator.fit(self.data)
        self.concat_data = self._propagate_kinetic_model(np.concatenate(self.data, axis=0))

    def _propagate_kinetic_model(self, frames):
        """Propagate the candidate frames using the kinetic model.

        :param frames: np.ndarray of shape (n_frames, n_features).
            Frames to transform applying Koopman operator.
        :return: np.ndarray of shape (n_frames, ndim).
            Transformed frames.
        """
        model = self.estimator.fetch_model()
        inst_obs = model.transform(frames)
        propagated = inst_obs @ (model.operator ** self.propagation_steps)[:self.ndim, :self.ndim]
        return propagated


class VampNetLeastCounts(VampLeastCounts):
    """This class implements Least Counts adaptive sampling using RegularSpace clustering in combination with VAMPNets
    from the deeptime package.
    """

    def __init__(self,
                 system=None,
                 root="",
                 basename="",
                 save_format='.dcd',
                 save_rate=100,
                 features=None,
                 save_info=False,
                 cluster_args=None,
                 ndim=2,
                 lagtime=1,
                 device="cuda",
                 vnet_lobe=None,
                 vnet_learning_rate=1e-4,
                 vnet_batch_size=64,
                 vnet_epochs=100,
                 vnet_num_threads=1,
                 log_file=None):
        """Constructor for VampNetLeastCounts.

        :param system: Simulation object.
            Object that implements the dynamics to be simulated.
        :param root: str.
            Path to root directory where data will be saved.
        :param basename: str.
            Basename for saved trajectory files.
        :param save_format: str, default = ".dcd".
            Saved format to use for trajectories (not implemented yet).
        :param save_rate: int, default = 100.
            Save rate in frames for trajectory files.
        :param features: list[Callable].
            List of callables that take a trajectory file as input and return a real number per frame.
        :param save_info: Bool, default = False.
            Save logging info for each trajectory run.
        :param cluster_args: list[float, int].
            List of parameters for RegularSpace clustering (dmin, max_centers). dmin is the minimum distance admissible
            between two centers. max_centers is the maximum number of clusters that can be created.
        :param ndim: int, default = 2.
            Size of VAMPNet output layer.
        :param lagtime: int, default = 1.
            Lag time expressed as number of frames.
        :param device: str.
            Device where training of the VAMPNet will take place. See pytorch documentation for options.
        :param vnet_lobe: deeptime.util.torch.MLP.
            Multilayer perceptron model for VAMPNet. Lobe duplication is hard-coded.
        :param vnet_learning_rate: float, default 1e-4.
            Learning rate for VAMPNet.
        :param vnet_batch_size: int, default = 64.
            Batch size for VAMPNet.
        :param vnet_epochs: int, default = 100.
            Number of training epochs per adaptive sampling round.
        :param vnet_num_threads: int, default = 1.
            Number of threads available for VAMPNet fitting.
        :param log_file: str.
            Path to log_file. Passing this argument will supersede all other parameters.
        """
        VampLeastCounts.__init__(self,
                                 system=system,
                                 root=root,
                                 basename=basename,
                                 save_format=save_format,
                                 save_rate=save_rate,
                                 features=features,
                                 save_info=save_info,
                                 cluster_args=cluster_args,
                                 ndim=ndim,
                                 lagtime=lagtime)
        if log_file is None:
            self.batch_size = vnet_batch_size
            self.epochs = vnet_epochs
            self.device_name = device
            self.vnet_num_threads = vnet_num_threads
            self._set_device(device, vnet_num_threads)
            if vnet_lobe is None:
                vnet_lobe = MLP(units=[self.n_features, 15, 10, 10, 5, self.ndim],
                                nonlinearity=nn.ReLU)  # A default model
            vnet_lobe = vnet_lobe.to(self.device)
            self.estimator = VAMPNet(lobe=vnet_lobe, learning_rate=vnet_learning_rate, device=self.device)
        elif isinstance(log_file, str):
            self._reload(log_file)

    def _save_logs(self, filename):
        """Save logging information of the run.

        :param filename: str.
            Name of output file.
        :return: None.
        """
        logs = dict(
            system=self.system,
            fhandler=self.fhandler,
            save_rate=self.save_rate,
            n_round=self.n_round,
            features=self.features,
            n_features=self.n_features,
            cluster_args=self.cluster_args,
            cluster_object=self._cluster_object,
            states=self.states,
            ndim=self.ndim,
            lagtime=self.lagtime,
            batch_size=self.batch_size,
            epochs=self.epochs,
            device_name=self.device_name,
            vnet_num_threads=self.vnet_num_threads,
            estimator=self.estimator,
        )
        root_dir = self.fhandler.root
        path = os.path.join(root_dir, filename)
        with open(path, "wb") as outfile:
            pickle.dump(logs, outfile)

    def _reload(self, log_file):
        """Reset simulation object to state in log_file. Called in constructor.

        For this method to work, trajectories should be found where self.fhandler expects them.

        :param log_file: str.
            Path to log file.
        """
        with open(log_file, "rb") as infile:
            logs = pickle.load(infile)
        self.system = logs['system']
        self.fhandler = logs['fhandler']
        self.save_rate = logs['save_rate']
        self.n_round = logs['n_round']
        self.features = logs['features']
        self.n_features = logs['n_features']
        self.cluster_args = logs['cluster_args']
        self._cluster_object = logs['cluster_object']
        self.states = logs['states']
        self.ndim = logs['ndim']
        self.lagtime = logs['lagtime']
        self.device_name = logs['device_name']
        self.vnet_num_threads = logs['vnet_num_threads']
        self._set_device(self.device_name, self.vnet_num_threads)
        self.estimator = logs['estimator']
        self.save_info = True  # If loading from log file, assume that log files are required
        self.data = None
        self.concat_data = None
        self._cached_trajs = set()
        self._cached_trajs_ordered = []
        self._cluster_object = None
        self._update_data()

    def _set_device(self, device, num_threads):
        """Set the device for VAMPNet training.

        :param device: str.
            Device where training of the VAMPNet will take place. See pytorch documentation for options.
        :param num_threads: int, default = 1.
            Number of threads available for VAMPNet fitting.
        :return: None.
        """
        if (device == "cuda") and torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")
        torch.set_num_threads(num_threads)

    def _propagate_kinetic_model(self, frames):
        """Transform specified simulation frames using the learned VAMPNet.

        :param frames: np.ndarray of shape (n_frames, n_features).
            Features to transform.
        :return: np.ndarray of shape (n_frames, ndim).
            Transformed frames.
        """
        model = self.estimator.fetch_model()
        propagated = model.transform(frames)
        return propagated

    def _lagged_dataset(self, data):
        """Convert trajectories to deeptime.util.data.TrajectoryDataset object.

        :param data: list[np.ndarray].
            List of trajectories.
        :return: deeptime.util.data.TrajectoryDataset.
            Lagged dataset object.
        """
        data_float32 = list(map(lambda x: x.astype(np.float32), data))
        return TrajectoryDataset.from_trajectories(self.lagtime, data_float32)

    def _format_data(self, data, batch_size):
        """Format list of trajectories into the correct format to fit a VAMPNet.

        :param data: list[np.ndarray].
            List of trajectories.
        :param batch_size: int.
            Batch size for VAMPNet fitting.
        :return: torch.utils.data.dataloader.DataLoader.
            Data loader to train VAMPNet.
        """
        lagged_data = self._lagged_dataset(data)
        loader_train = DataLoader(lagged_data, batch_size=batch_size, shuffle=True)
        return loader_train

    def _fit_estimator(self):
        """Fit VAMPNet with list of trajectories currently stored in self.data.

        :return: None.
        """
        loader_train = self._format_data(self.data, self.batch_size)
        self.estimator.fit(loader_train, n_epochs=self.epochs)

    def _update_data(self):
        """Update data with newly saved trajectories.

        :return: None.
        """
        fnames = self.fhandler.list_all_files()
        new_fnames = []
        for fn in fnames:
            if fn not in self._cached_trajs:
                new_fnames.append(fn)
        if self.data is None:
            self.data = self.system.project_cvs(self.features, new_fnames)
        else:
            new_data = self.system.project_cvs(self.features, new_fnames)
            self.data.extend(new_data)
        self._fit_estimator()
        self.concat_data = self._propagate_kinetic_model(np.concatenate(self.data, axis=0))
        self._cached_trajs.update(new_fnames)
        self._cached_trajs_ordered.extend(new_fnames)


class VaeLeastCounts(VampLeastCounts):
    """This class implements Least Counts adaptive sampling using RegularSpace clustering in combination with TVAEs
    from the deeptime package.

    """

    def __init__(self,
                 system=None,
                 root="",
                 basename="",
                 save_format='.dcd',
                 save_rate=100,
                 features=None,
                 save_info=False,
                 cluster_args=None,
                 ndim=2,
                 lagtime=1,
                 device="cuda",
                 tvae_encoder=None,
                 tvae_decoder=None,
                 tvae_learning_rate=1e-4,
                 tvae_batch_size=64,
                 tvae_epochs=100,
                 tvae_num_threads=1,
                 log_file=None):
        """Constructor for VaeLeastCounts.

        :param system: Simulation object.
            Object that implements the dynamics to be simulated.
        :param root: str.
            Path to root directory where data will be saved.
        :param basename: str.
            Basename for saved trajectory files.
        :param save_format: str, default = ".dcd".
            Saved format to use for trajectories (not implemented yet).
        :param save_rate: int, default = 100.
            Save rate in frames for trajectory files.
        :param features: list[Callable].
            List of callables that take a trajectory file as input and return a real number per frame.
        :param save_info: Bool, default = False.
            Save logging info for each trajectory run.
        :param cluster_args: list[float, int].
            List of parameters for RegularSpace clustering (dmin, max_centers). dmin is the minimum distance admissible
            between two centers. max_centers is the maximum number of clusters that can be created.
        :param ndim: int, default = 2.
            Size of VAMPNet output layer.
        :param lagtime: int, default = 1.
            Lag time expressed as number of frames.
        :param device: str.
            Device where training of the VAMPNet will take place. See pytorch documentation for options.
        :param tvae_encoder: deeptime.decomposition.deep._tae.TVAEEncoder.
            Encoder for the TVAE. See deeptime documentation for details.
        :param tvae_decoder: typically deeptime.util.torch.MLP.
            Decoder for the TVAE. See deeptime documentation for details.
        :param tvae_learning_rate: float, default 1e-4.
            Learning rate for TVAE.
        :param tvae_batch_size: int, default = 64.
            Batch size for TVAE.
        :param tvae_epochs: int, default = 100.
            Number of training epochs per adaptive sampling round.
        :param tvae_num_threads: int, default = 1.
            Number of threads available for TVAE fitting.
        :param log_file: str.
            Path to log_file. Passing this argument will supersede all other parameters.
        """
        VampLeastCounts.__init__(self,
                                 system=system,
                                 root=root,
                                 basename=basename,
                                 save_format=save_format,
                                 save_rate=save_rate,
                                 features=features,
                                 save_info=save_info,
                                 cluster_args=cluster_args,
                                 ndim=ndim,
                                 lagtime=lagtime)
        if log_file is None:
            self.batch_size = tvae_batch_size
            self.epochs = tvae_epochs
            self._set_device(device, tvae_num_threads)
            # Set estimator
            if tvae_encoder is None:
                tvae_encoder = TVAEEncoder([self.n_features, 15, 10, 10, 5, self.ndim],
                                           nonlinearity=torch.nn.ReLU)  # A default encoder
            if tvae_decoder is None:
                tvae_decoder = MLP([self.ndim, 5, 10, 10, 15, self.n_features], nonlinearity=torch.nn.ReLU,
                                   initial_batchnorm=False)  # A default decoder
            tvae_encoder = tvae_encoder.to(self.device)
            tvae_decoder = tvae_decoder.to(self.device)
            self.estimator = TVAE(tvae_encoder, tvae_decoder, learning_rate=tvae_learning_rate)
        elif isinstance(log_file, str):
            self._reload(log_file)

    def _save_logs(self, filename):
        """Save logging information of the run.

        :param filename: str.
            Name of output file.
        :return: None.
        """
        logs = dict(
            system=self.system,
            fhandler=self.fhandler,
            save_rate=self.save_rate,
            n_round=self.n_round,
            features=self.features,
            n_features=self.n_features,
            cluster_args=self.cluster_args,
            cluster_object=self._cluster_object,
            states=self.states,
            ndim=self.ndim,
            lagtime=self.lagtime,
            device=self.device,
            batch_size=self.batch_size,
            epochs=self.epochs,
            device_name=self.device_name,
            tvae_num_threads=self.tvae_num_threads,
            estimator=self.estimator,
        )
        root_dir = self.fhandler.root
        path = os.path.join(root_dir, filename)
        with open(path, "wb") as outfile:
            pickle.dump(logs, outfile)

    def _reload(self, log_file):
        """Reset simulation object to state in log_file. Called in constructor.

        For this method to work, trajectories should be found where self.fhandler expects them.

        :param log_file: str.
            Path to log file.
        """
        with open(log_file, "rb") as infile:
            logs = pickle.load(infile)
        self.system = logs['system']
        self.fhandler = logs['fhandler']
        self.save_rate = logs['save_rate']
        self.n_round = logs['n_round']
        self.features = logs['features']
        self.n_features = logs['n_features']
        self.cluster_args = logs['cluster_args']
        self._cluster_object = logs['cluster_object']
        self.states = logs['states']
        self.ndim = logs['ndim']
        self.lagtime = logs['lagtime']
        self.device_name = logs['device_name']
        self.tvae_num_threads = logs['tvae_num_threads']
        self._set_device(self.device_name, self.tvae_num_threads)
        self.estimator = logs['estimator']
        self.save_info = True  # If loading from log file, assume that log files are required
        self.data = None
        self.concat_data = None
        self._cached_trajs = set()
        self._cached_trajs_ordered = []
        self._cluster_object = None
        self._update_data()

    def _set_device(self, device, num_threads):
        """Set the device for VAMPNet training.

        :param device: str.
            Device where training of the VAMPNet will take place. See pytorch documentation for options.
        :param num_threads: int, default = 1.
            Number of threads available for TVAE fitting.
        :return: None.
        """
        if (device == "cuda") and torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")
        torch.set_num_threads(num_threads)

    def _propagate_kinetic_model(self, frames):
        """Transform specified simulation frames using the learned TVAE.

        :param frames: np.ndarray of shape (n_frames, n_features).
            Features to transform.
        :return: np.ndarray of shape (n_frames, ndim).
            Transformed frames.
        """
        model = self.estimator.fetch_model()
        propagated = model.transform(frames)
        return propagated

    def _lagged_dataset(self, data):
        """Convert trajectories to deeptime.util.data.TrajectoryDataset object.

        :param data: list[np.ndarray].
            List of trajectories.
        :return: deeptime.util.data.TrajectoryDataset.
            Lagged dataset object.
        """
        data_float32 = list(map(lambda x: x.astype(np.float32), data))
        return TrajectoryDataset.from_trajectories(self.lagtime, data_float32)

    def _format_data(self, data, batch_size):
        """Format list of trajectories into the correct format to fit a TVAE.

        :param data: list[np.ndarray].
            List of trajectories.
        :param batch_size: int.
            Batch size for TVAE fitting.
        :return: torch.utils.data.dataloader.DataLoader.
            Data loader to train TVAE.
        """
        lagged_data = self._lagged_dataset(data)
        loader_train = DataLoader(lagged_data, batch_size=batch_size, shuffle=True)
        return loader_train

    def _fit_estimator(self):
        """Fit TVAE with list of trajectories currently stored in self.data.

        :return: None.
        """
        loader_train = self._format_data(self.data, self.batch_size)
        self.estimator.fit(loader_train, n_epochs=self.epochs)

    def _update_data(self):
        """Update data with newly saved trajectories.

        :return: None.
        """
        fnames = self.fhandler.list_all_files()
        new_fnames = []
        for fn in fnames:
            if fn not in self._cached_trajs:
                new_fnames.append(fn)
        if self.data is None:
            self.data = self.system.project_cvs(self.features, new_fnames)
        else:
            new_data = self.system.project_cvs(self.features, new_fnames)
            self.data.extend(new_data)
        self._fit_estimator()
        self.concat_data = self._propagate_kinetic_model(np.concatenate(self.data, axis=0))
        self._cached_trajs.update(new_fnames)
        self._cached_trajs_ordered.extend(new_fnames)
