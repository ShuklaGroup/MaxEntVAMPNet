import os

import dill as pickle
import numpy as np
from sklearn.preprocessing import normalize

from AdaptiveSampling import LeastCounts, LeastCountsBis, LeastCountsRegSpace
from Agent import AgentReap
from FileHandler import FileHandlerMultiagent


class Reap(LeastCountsRegSpace):
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
                 cluster_args=None):
        LeastCountsRegSpace.__init__(self, system=system, root=root, basename=basename, save_format=save_format,
                                     save_rate=save_rate, features=features, save_info=save_info,
                                     cluster_args=cluster_args)
        self.n_agents = 1  # Single agent version
        if cv_weights is None:
            cv_weights = [1 / self.n_features for _ in range(self.n_features)]
        self.agent = AgentReap(cv_weights, delta)
        self.n_candidates = n_candidates

    def _save_logs(self, filename):
        """
        Unlike save(), this method does not pickle the entire object, but just a dictionary containing info about the
        run (without trajectories).
        """
        agent_logs = self.agent.get_log_info()
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
        )
        root_dir = self.fhandler.root
        path = os.path.join(root_dir, filename)
        with open(path, "wb") as outfile:
            pickle.dump(logs, outfile)

    def _update_data(self):
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
        fnames = self.fhandler.list_all_files()
        new_fnames = []
        for fn in fnames:
            if fn not in self._cached_trajs:
                new_fnames.append(fn)
        new_data = self.system.project_cvs(self.features, new_fnames)
        self.agent.set_data(new_data)

    def _train_agent(self):
        self.agent.set_states(self.states)
        self.agent.train()


class MultiagentReap(Reap):

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
        '''
        Unlike save(), this method does not pickle the entire object, but just a dictionary containing info about the
        run (without trajectories).
        '''
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
        num_frames = np.empty((self.n_agents, self.n_candidates))

        for n, agent in enumerate(self.agent):
            data_labels = self._cluster_object.model.transform(agent.data_concat)  # TODO: Speed up by avoiding using predict
            for i, label in enumerate(cluster_labels):
                num_frames[n, i] = len(np.where(data_labels == label)[0])

        self.stakes = normalize(num_frames, norm="l1", axis=0)

        if self.stakes_method != "percentage":
            self._transform_stakes()

    def _transform_stakes(self):
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
        for n, agent in enumerate(self.agent):
            agent.set_states(self.states)
            agent.set_stakes(self.stakes[n])
            agent.train()

    def _aggregate_scores(self):
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
