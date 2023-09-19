"""Definition of the classes for REAP-like implementations.
"""
import os

import dill as pickle
import numpy as np
from sklearn.preprocessing import normalize

from AdaptiveSampling import LeastCountsRegSpace
from Agent import AgentReap
from FileHandler import FileHandlerMultiagent


class Reap(LeastCountsRegSpace):
    """This class implements REAP adaptive sampling using RegularSpace clustering.
    """
    def __init__(self,
                 system=None,
                 root="",
                 basename="",
                 save_format='.dcd',
                 save_rate=100,
                 features=None,
                 cv_weights=None,
                 delta=0.05,
                 n_candidates=50,
                 save_info=False,
                 cluster_args=None,
                 log_file=None):
        """Constructor for Reap class.

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
        :param cv_weights: list[float].
            Weights for collective variables.
        :param delta: float in [0, 1].
            Max change in collective variable weights between rounds.
        :param n_candidates: int, default = 50.
            Number of Least Counts candidates to consider in each iteration.
        :param save_info: Bool, default = False.
            Save logging info for each trajectory run.
        :param cluster_args: list[float, int].
            List of parameters for RegularSpace clustering (dmin, max_centers). dmin is the minimum distance admissible
            between two centers. max_centers is the maximum number of clusters that can be created.
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
                                     cluster_args=cluster_args)
        if log_file is None:
            self.n_agents = 1  # Single agent version
            if cv_weights is None:
                cv_weights = [1 / self.n_features for _ in range(self.n_features)]
            self.agent = AgentReap(cv_weights=cv_weights, delta=delta)
            self.n_candidates = n_candidates
        elif isinstance(log_file, str):
            self._reload(log_file)

    def _save_logs(self, filename):
        """Save logging information of the run.

        :param filename: str.
            Name of output file.
        :return: None.
        """
        agent_logs = self.agent.get_log_info()
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
            n_agents=self.n_agents,
            n_candidates=self.n_candidates,
            agent_logs=agent_logs,
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
        self.n_agents = logs['n_agents']
        self.n_candidates = logs['n_candidates']
        self.agent = AgentReap(logs=logs['agent_logs'])
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
        self._update_data_agent()
        self._cached_trajs.update(new_fnames)
        self._cached_trajs_ordered.extend(new_fnames)

    def _select_states(self, n_select):
        """Select states to restart simulations.

        :param n_select: int.
            Number of states to select.
        :return: starting_states_info (list[dict]).
            List of dicts containing information to read a state (fname, frame_idx, and top_file).
        """
        least_counts_cluster_indices, candidate_frames_indices = self._find_candidates(self.n_candidates)
        self.states = self.concat_data[candidate_frames_indices]
        self._train_agent()
        scores = self.agent.score()
        assert (len(scores) == len(self.states))
        selected_states = np.argsort(scores)[-n_select:][::-1]
        starting_states = candidate_frames_indices[selected_states]
        starting_states_info = self.fhandler.find_state_info(starting_states,
                                                             self._cached_trajs_ordered,
                                                             self.system.top_file)

        return starting_states_info

    def _update_data_agent(self):
        """Update the data of the (single) agent.

        :return: None.
        """
        fnames = self.fhandler.list_all_files()
        new_fnames = []
        for fn in fnames:
            if fn not in self._cached_trajs:
                new_fnames.append(fn)
        new_data = self.system.project_cvs(self.features, new_fnames)
        self.agent.set_data(new_data)

    def _train_agent(self):
        """Call agent's train() method.

        :return: None.
        """
        self.agent.set_states(self.states)
        self.agent.train()


class MultiagentReap(Reap):
    """This class implements multiagent REAP adaptive sampling using RegularSpace clustering.
    """
    def __init__(self,
                 system=None,
                 root="",
                 basename="",
                 save_format='.dcd',
                 save_rate=100,
                 features=None,
                 cv_weights=None,
                 delta=0.05,
                 n_agents=1,
                 n_candidates=50,
                 stakes_method="percentage",
                 stakes_kwargs=None,
                 interaction="collaborative",
                 save_info=False,
                 cluster_args=None
                 ):
        """Constructor for multiagent Reap class.

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
        :param cv_weights: list[float].
            Weights for collective variables.
        :param delta: float in [0, 1].
            Max change in collective variable weights between rounds.
        :param n_agents: int, default = 1.
            Number of agents.
        :param n_candidates: int, default = 50.
            Number of Least Counts candidates to consider in each iteration.
        :param stakes_method: str {percentage, equal, logistic}, default = 'percentage'.
            Method used to compute the stakes of the agent.
        :param stakes_kwargs: dict.
            Aguments required to compute stakes.
            In current implementation, this is only needed when using stakes_method = 'logistic', in which case
            stakes_kwargs must be defined as {'k': float} where k is the kappa parameter.
        :param interaction: str {collaborative, noncollaborative, competitive}, default = 'collaborative'.
            Regime to combine rewards from different agents.
        :param save_info: Bool, default = False.
            Save logging info for each trajectory run.
        :param cluster_args: list[float, int].
            List of parameters for RegularSpace clustering (dmin, max_centers). dmin is the minimum distance admissible
            between two centers. max_centers is the maximum number of clusters that can be created.
        """
        Reap.__init__(self, system=system, root=root, basename=basename, save_format=save_format, save_rate=save_rate,
                      features=features, cv_weights=cv_weights, delta=delta, n_candidates=n_candidates,
                      save_info=save_info, cluster_args=cluster_args)
        self.n_agents = n_agents
        # Three options to initialize cv_weights
        if cv_weights is None:  # All weights equal
            cv_weights = [1 / self.n_features for _ in range(self.n_features)]
            self.agent = [AgentReap(cv_weights, delta) for _ in range(self.n_agents)]
        elif np.asarray(cv_weights).shape == (self.n_agents, self.n_features):  # Custom weights for each agent
            self.agent = [AgentReap(cv_weights[i], delta) for i in range(self.n_agents)]
        elif np.asarray(cv_weights).shape == (self.n_features,):  # Custom weights (same for all agents)
            self.agent = [AgentReap(cv_weights, delta) for _ in range(self.n_agents)]

        self.fhandler = FileHandlerMultiagent(root, n_agents, basename, save_format)
        self.stakes = None
        self.stakes_method = stakes_method
        self.stakes_kwargs = stakes_kwargs if stakes_kwargs else {}
        self._cached_agent_trajs = [set() for _ in range(n_agents)]
        self._cached_agent_trajs_ordered = [[] for _ in range(n_agents)]
        self.interaction = interaction

    def _save_logs(self, filename):
        """Save logging information of the run.

        :param filename: str.
            Name of output file.
        :return: None.
        """
        agent_logs = [a.get_log_info() for a in self.agent]
        logs = dict(
            system=self.system,
            fhandler=self.fhandler,
            save_rate=self.save_rate,
            n_round=self.n_round,
            features=self.features,
            n_features=self.n_features,
            states=self.states,
            n_agents=self.n_agents,
            n_candidates=self.n_candidates,
            agent_logs=agent_logs,
            stakes_method=self.stakes_method,
            stakes_kwargs=self.stakes_kwargs,
            interaction=self.interaction,
        )
        root_dir = self.fhandler.root
        path = os.path.join(root_dir, filename)
        with open(path, "wb") as outfile:
            pickle.dump(logs, outfile)

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
        for n in range(self.n_agents):
            self._update_data_agent(n)
        self._cached_trajs.update(new_fnames)
        self._cached_trajs_ordered.extend(new_fnames)

    def _update_data_agent(self, agent_idx):
        """Update the data of the agent indexed by agent_idx.

        :param agent_idx: int.
            Agent index.
        :return: None.
        """
        fnames = self.fhandler.list_agent_files(agent_idx)
        new_fnames = []
        for fn in fnames:
            if fn not in self._cached_agent_trajs[agent_idx]:
                new_fnames.append(fn)
        new_data = self.system.project_cvs(self.features, new_fnames)
        self.agent[agent_idx].set_data(new_data)
        self._cached_agent_trajs[agent_idx].update(new_fnames)
        self._cached_agent_trajs_ordered[agent_idx].extend(new_fnames)

    def _compute_stakes(self, cluster_labels):
        """Compute stakes for agents.

        :param cluster_labels: np.ndarray.
            Labels indicating cluster membership of each simulation frame.
        :return: None.
        """
        num_frames = np.empty((self.n_agents, self.n_candidates))

        for n, agent in enumerate(self.agent):
            data_labels = self._cluster_object.model.transform(agent.data_concat)
            for i, label in enumerate(cluster_labels):
                num_frames[n, i] = len(np.where(data_labels == label)[0])

        self.stakes = normalize(num_frames, norm="l1", axis=0)

        if self.stakes_method != "percentage":
            self._transform_stakes()

    def _transform_stakes(self):
        """Auxiliary function for compute stakes. Only called if using a stakes_method different from 'percentage'.

        :return: None.
        """
        temp = self.stakes

        if self.stakes_method == "equal":
            for i in range(temp.shape[1]):
                temp[:, i][np.where(self.stakes[:, i] != 0)] = 1 / np.count_nonzero(self.stakes[:, i])

        elif self.stakes_method == "logistic":
            k = self.stakes_kwargs["k"]
            x0 = 0.5

            def logistic_fun(x):
                return 1 / (1 + np.exp(-k * (x - x0)))

            for i in range(temp.shape[1]):
                temp[:, i] = logistic_fun(self.stakes[:, i])
                temp[:, i][np.where(self.stakes[:, i] < 1e-18)] = 0  # Evaluate to zero at x < 1e-18
                temp[:, i] /= temp[:, i].sum()

        self.stakes = temp

    def _train_agent(self):
        """ Call train() method in all agents.

        :return: None.
        """
        for n, agent in enumerate(self.agent):
            agent.set_states(self.states)
            agent.set_stakes(self.stakes[n])
            agent.train()

    def _aggregate_scores(self):
        """Combine rewards from all agents.

        :return: np.ndarray of shape (n_candidates,).
            Aggregated score for each candidate structure.
        """
        scores = np.empty((self.n_agents, self.n_candidates))
        for n, agent in enumerate(self.agent):
            scores[n] = agent.score()

        if self.interaction == "collaborative":
            aggregated_scores = scores.sum(axis=0)
        elif self.interaction == "noncollaborative":
            aggregated_scores = scores.max(axis=0)
        elif self.interaction == "competitive":
            aggregated_scores = 2 * scores.max(axis=0) - scores.sum(axis=0)

        return aggregated_scores

    def _select_states(self, n_select):
        """Select states to restart simulations.

        :param n_select: int.
            Number of states to select.
        :return: starting_states_info (list[dict]).
            List of dicts containing information to read a state (fname, frame_idx, top_file, and agent_idx).
        """
        least_counts_cluster_indices, candidate_frames_indices = self._find_candidates(self.n_candidates)
        self.states = self.concat_data[candidate_frames_indices]
        self._compute_stakes(least_counts_cluster_indices)
        assert (self.stakes.shape[1] == len(self.states))
        self._train_agent()
        aggregated_scores = self._aggregate_scores()
        assert (len(aggregated_scores) == len(self.states))
        selected_states = np.argsort(aggregated_scores)[-n_select:][::-1]
        starting_states = candidate_frames_indices[selected_states]
        executors = np.argmax(self.stakes, axis=0)[selected_states]
        starting_states_info = self.fhandler.find_state_info(starting_states,
                                                             self._cached_trajs_ordered,
                                                             self.system.top_file,
                                                             executors)

        return starting_states_info
