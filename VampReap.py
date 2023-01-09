import numpy as np

from Reap import Reap, MultiagentReap
from Agent import AgentVampReap
from deeptime.decomposition import VAMP


# TODO: change in implementation: the adaptive sampling object keeps a single kinetic model.
# Even if multiple agents are used, they all reference the same kinetic model.

class VampReap(Reap):
    def __init__(self,
                 system=None,
                 root="",
                 basename="",
                 save_format=".dcd",
                 save_rate=100,
                 features=None,
                 cv_weights=None,  # Same length as number of VAMP-reduced coordinates we want to keep
                 delta=0.05,
                 n_candidates=50,
                 lagtime=1,
                 propagation_steps=1,
                 save_info=False,
                 cluster_args=None
                 ):
        Reap.__init__(self, system=system, root=root, basename=basename, save_format=save_format, save_rate=save_rate,
                      features=features, cv_weights=cv_weights, delta=delta, n_candidates=n_candidates,
                      save_info=save_info, cluster_args=cluster_args)
        self.lagtime = lagtime
        self.propagation_steps = propagation_steps
        self.n_features = len(features)  # Must distinguish between features and CVs
        if cv_weights is None:
            cv_weights = [1 / self.n_features for _ in range(self.n_features)]
            self.cv_num = self.n_features
        else:
            self.cv_num = len(cv_weights)
        self.estimator = VAMP(lagtime=self.lagtime, dim=self.cv_num, epsilon=1e-18)  # Epsilon is set to a low
        # number to prevent errors, but this may result in poor quality models.
        self.agent = AgentVampReap(cv_weights, delta, self.estimator, propagation_steps=self.propagation_steps)

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
        self.estimator.fit(self.data)
        self.agent.set_estimator(self.estimator)
        self._update_data_agent()
        self._cached_trajs.update(new_fnames)
        self._cached_trajs_ordered.extend(new_fnames)


class MultiagentVampReap(MultiagentReap):

    def __init__(self,
                 system=None,
                 root="",
                 basename="",
                 save_format='.dcd',
                 save_rate=100,
                 features=None,
                 cv_weights=None,  # Same length as number of VAMP-reduced coordinates we want to keep
                 delta=0.05,
                 n_agents=1,
                 n_candidates=50,
                 stakes_method="percentage",
                 stakes_kwargs=None,
                 interaction="collaborative",
                 lagtime=1,
                 propagation_steps=1,
                 save_info=False,
                 cluster_args=None
                 ):
        MultiagentReap.__init__(self, system=system, root=root, basename=basename, save_format=save_format,
                                save_rate=save_rate, features=features, cv_weights=cv_weights, delta=delta,
                                n_agents=n_agents, n_candidates=n_candidates, stakes_method=stakes_method,
                                stakes_kwargs=stakes_kwargs, interaction=interaction, save_info=save_info,
                                cluster_args=cluster_args)
        self.lagtime = lagtime
        self.propagation_steps = propagation_steps

        # Three options to initialize cv_weights
        if cv_weights is None:  # All weights equal
            cv_weights = [1 / self.n_features for _ in range(self.n_features)]
            self.estimator = VAMP(lagtime=self.lagtime, dim=self.n_features, epsilon=1e-18)  # Epsilon is set to a low
            # number to prevent errors, but these may result in poor quality models.
            self.agent = [AgentVampReap(cv_weights, delta, self.estimator)
                          for _ in range(self.n_agents)]
        else:
            cv_weights = np.asarray(cv_weights)
            cv_w_shape = cv_weights.shape
            if len(cv_w_shape) == 2:  # Custom weights for each agent with shape (n_agents, cv_num)
                self.cv_num = cv_w_shape[1]
                self.estimator = VAMP(lagtime=self.lagtime, dim=self.cv_num, epsilon=1e-18)  # Epsilon is set to
                # a low number to prevent errors, but these may result in poor quality models.
                self.agent = [AgentVampReap(cv_weights[i], delta, self.estimator)
                              for i in range(self.n_agents)]
            elif len(cv_w_shape) == 1:  # Custom weights (same for all agents)
                self.cv_num = cv_w_shape[0]
                self.estimator = VAMP(lagtime=self.lagtime, dim=self.cv_num, epsilon=1e-18)  # Epsilon is set to
                # a low number to prevent errors, but these may result in poor quality models.
                self.agent = [AgentVampReap(cv_weights, delta, self.estimator)
                              for _ in range(self.n_agents)]

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
        self.estimator.fit(self.data)
        for n, agent in enumerate(self.agent):
            agent.set_estimator(self.estimator)
            self._update_data_agent(n)
        self._cached_trajs.update(new_fnames)
        self._cached_trajs_ordered.extend(new_fnames)
