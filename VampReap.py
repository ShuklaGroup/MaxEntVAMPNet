"""Definition of the classes for VAMP + REAP implementations.
"""

import numpy as np

from Reap import Reap, MultiagentReap
from Agent import AgentVampReap
from deeptime.decomposition import VAMP


class VampReap(Reap):
    """This class implements VAMP + REAP adaptive sampling using RegularSpace clustering.
    """
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
        """Constructor for VampReap class.

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
        :param lagtime: int, default = 1.
            Lag time expressed as number of frames.
        :param propagation_steps: int, default = 1.
            This option allows to apply the Koopman operator propagation_steps times to the input conformation.
        :param save_info: Bool, default = False.
            Save logging info for each trajectory run.
        :param cluster_args: list[float, int].
            List of parameters for RegularSpace clustering (dmin, max_centers). dmin is the minimum distance admissible
            between two centers. max_centers is the maximum number of clusters that can be created.
        """
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
        """Update data with newly saved trajectories. This method also updates the estimator.

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
        self.estimator.fit(self.data)
        self.agent.set_estimator(self.estimator)
        self._update_data_agent()
        self._cached_trajs.update(new_fnames)
        self._cached_trajs_ordered.extend(new_fnames)


class MultiagentVampReap(MultiagentReap):
    """This class implements VAMP + multiagent REAP adaptive sampling using RegularSpace clustering.
    """
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
        """Constructor for MultiagentVampReap class.

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
        :param lagtime: int, default = 1.
            Lag time expressed as number of frames.
        :param propagation_steps: int, default = 1.
            This option allows to apply the Koopman operator propagation_steps times to the input conformation.
        :param save_info: Bool, default = False.
            Save logging info for each trajectory run.
        :param cluster_args: list[float, int].
            List of parameters for RegularSpace clustering (dmin, max_centers). dmin is the minimum distance admissible
            between two centers. max_centers is the maximum number of clusters that can be created.
        """
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
        """Update data with newly saved trajectories. This method also updates the estimator.

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
        self.estimator.fit(self.data)
        for n, agent in enumerate(self.agent):
            agent.set_estimator(self.estimator)
            self._update_data_agent(n)
        self._cached_trajs.update(new_fnames)
        self._cached_trajs_ordered.extend(new_fnames)
