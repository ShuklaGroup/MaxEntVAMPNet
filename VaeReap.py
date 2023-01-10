"""Definition of the classes for TVAE + REAP implementations.
"""

import numpy as np
import torch
from deeptime.decomposition.deep import TVAEEncoder, TVAE
from deeptime.util.torch import MLP

from Agent import AgentVaeReap
from VampNetReap import VampNetReap, MultiagentVampNetReap


class VaeReap(VampNetReap):
    """This class implements REAP adaptive sampling using RegularSpace clustering in combination with TVAEs
    from the deeptime package.
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
                 device="cuda",
                 tvae_encoder=None,
                 tvae_decoder=None,
                 tvae_learning_rate=1e-4,
                 tvae_batch_size=64,
                 tvae_epochs=100,
                 tvae_num_threads=1,
                 save_info=False,
                 cluster_args=None
                 ):
        """Constructor for VaeReap class.

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
        :param save_info: Bool, default = False.
            Save logging info for each trajectory run.
        :param cluster_args: list[float, int].
            List of parameters for RegularSpace clustering (dmin, max_centers). dmin is the minimum distance admissible
            between two centers. max_centers is the maximum number of clusters that can be created.
        """
        VampNetReap.__init__(self, system=system, root=root, basename=basename, save_format=save_format,
                             save_rate=save_rate, features=features, cv_weights=cv_weights, delta=delta,
                             n_candidates=n_candidates, save_info=save_info, cluster_args=cluster_args)
        self.lagtime = lagtime
        self.n_features = len(features)
        if cv_weights is None:
            self.cv_num = self.n_features
            cv_weights = [1 / self.n_features for _ in range(self.n_features)]
        else:
            self.cv_num = len(cv_weights)
        self.batch_size = tvae_batch_size
        self.epochs = tvae_epochs
        self._set_device(device, tvae_num_threads)
        # Set estimator
        if tvae_encoder is None:
            tvae_encoder = TVAEEncoder([self.n_features, 15, 10, 10, 5, self.cv_num],
                                       nonlinearity=torch.nn.ReLU)  # A default encoder
        if tvae_decoder is None:
            tvae_decoder = MLP([self.cv_num, 5, 10, 10, 15, self.n_features], nonlinearity=torch.nn.ReLU,
                               initial_batchnorm=False)  # A default decoder
        tvae_encoder = tvae_encoder.to(self.device)
        tvae_decoder = tvae_decoder.to(self.device)
        self.estimator = TVAE(tvae_encoder, tvae_decoder, learning_rate=tvae_learning_rate)
        self.agent = AgentVaeReap(cv_weights, delta, self.estimator)


class MultiagentVaeReap(MultiagentVampNetReap):
    """This class implements MA REAP adaptive sampling using RegularSpace clustering in combination with TVAEs
    from the deeptime package.
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
                 n_agents=1,
                 n_candidates=50,
                 stakes_method="percentage",
                 stakes_kwargs=None,
                 interaction="collaborative",
                 lagtime=1,
                 # propagation_steps=10, --> Does not apply
                 device="cuda",
                 tvae_encoder=None,
                 tvae_decoder=None,
                 tvae_learning_rate=1e-4,
                 tvae_batch_size=64,
                 tvae_epochs=100,
                 tvae_num_threads=1,
                 save_info=False,
                 cluster_args=None
                 ):
        """Constructor for MultiagentVaeReap.

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
        :param save_info: Bool, default = False.
            Save logging info for each trajectory run.
        :param cluster_args: list[float, int].
            List of parameters for RegularSpace clustering (dmin, max_centers). dmin is the minimum distance admissible
            between two centers. max_centers is the maximum number of clusters that can be created.
        """
        MultiagentVampNetReap.__init__(self, system=system, root=root, basename=basename, save_format=save_format,
                                       save_rate=save_rate, features=features, cv_weights=cv_weights, delta=delta,
                                       n_agents=n_agents, n_candidates=n_candidates, stakes_method=stakes_method,
                                       stakes_kwargs=stakes_kwargs, interaction=interaction, save_info=save_info,
                                       cluster_args=cluster_args)
        self.lagtime = lagtime
        self.n_features = len(features)  # Must distinguish between features and CVs
        self.batch_size = tvae_batch_size
        self.epochs = tvae_epochs
        self._set_device(device, tvae_num_threads)

        # Three options to initialize cv_weights
        if cv_weights is None:  # All weights equal
            cv_weights = np.asarray([1 / self.n_features for _ in range(self.n_features)])
            cv_w_shape = cv_weights.shape
            self.cv_num = self.n_features
        else:
            cv_weights = np.asarray(cv_weights)
            cv_w_shape = cv_weights.shape
            if len(cv_w_shape) == 2:  # Custom weights for each agent with shape (n_agents, cv_num)
                self.cv_num = cv_w_shape[1]
            elif len(cv_w_shape) == 1:  # Custom weights (same for all agents)
                self.cv_num = cv_w_shape[0]
        # Set estimator
        if tvae_encoder is None:
            tvae_encoder = TVAEEncoder([self.n_features, 100, 100, self.cv_num],
                                       nonlinearity=torch.nn.ReLU)  # A default encoder
        if tvae_decoder is None:
            tvae_decoder = MLP([self.cv_num, 100, 100, self.n_features], nonlinearity=torch.nn.ReLU,
                               initial_batchnorm=False)  # A default decoder
        tvae_encoder = tvae_encoder.to(self.device)
        tvae_decoder = tvae_decoder.to(self.device)
        self.estimator = TVAE(tvae_encoder, tvae_decoder, learning_rate=tvae_learning_rate)
        # Initialize agents
        if len(cv_w_shape) == 1:
            self.agent = [AgentVaeReap(cv_weights, delta, self.estimator)
                          for _ in range(self.n_agents)]
        elif len(cv_w_shape) == 2:
            self.agent = [AgentVaeReap(cv_weights[i], delta, self.estimator)
                          for i in range(self.n_agents)]
