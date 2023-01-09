import numpy as np
import torch
from deeptime.decomposition.deep import TVAEEncoder, TVAE
from deeptime.util.torch import MLP

from Agent import AgentVaeReap
from VampNetReap import VampNetReap, MultiagentVampNetReap


class VaeReap(VampNetReap):
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
