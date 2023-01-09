import numpy as np
from Reap import Reap, MultiagentReap
from Agent import AgentVampNetReap

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from deeptime.util.torch import MLP
from deeptime.decomposition.deep import VAMPNet
from deeptime.util.data import TrajectoryDataset


class VampNetReap(Reap):
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
                 vnet_lobe=None,
                 vnet_learning_rate=1e-4,
                 vnet_batch_size=64,
                 vnet_epochs=100,
                 vnet_num_threads=1,
                 save_info=False,
                 cluster_args=None
                 ):
        Reap.__init__(self, system=system, root=root, basename=basename, save_format=save_format, save_rate=save_rate,
                      features=features, cv_weights=cv_weights, delta=delta, n_candidates=n_candidates,
                      save_info=save_info, cluster_args=cluster_args)
        self.lagtime = lagtime
        self.n_features = len(features)  # Must distinguish between features and CVs
        if cv_weights is None:
            self.cv_num = self.n_features
            cv_weights = [1 / self.n_features for _ in range(self.n_features)]
        else:
            self.cv_num = len(cv_weights)
        self.batch_size = vnet_batch_size
        self.epochs = vnet_epochs
        self._set_device(device, vnet_num_threads)
        # Set estimator
        if vnet_lobe is None:
            vnet_lobe = MLP(units=[self.n_features, 15, 10, 10, 5, self.cv_num],
                            nonlinearity=nn.ReLU)  # A default model
        vnet_lobe = vnet_lobe.to(self.device)
        self.estimator = VAMPNet(lobe=vnet_lobe, learning_rate=vnet_learning_rate, device=self.device)
        self.agent = AgentVampNetReap(cv_weights, delta, self.estimator)

    def _set_device(self, device, num_threads):
        if (device == "cuda") and torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")
        torch.set_num_threads(num_threads)

    def _lagged_dataset(self, data):
        data_float32 = list(map(lambda x: x.astype(np.float32), data))
        return TrajectoryDataset.from_trajectories(self.lagtime, data_float32)

    def _format_data(self, data, batch_size):
        lagged_data = self._lagged_dataset(data)
        loader_train = DataLoader(lagged_data, batch_size=batch_size, shuffle=True)
        return loader_train

    def _format_data_val(self, data, batch_size, val_percent=0.3):  # Include validation split to check if model works
        lagged_data = self._lagged_dataset(data)
        n_val = int(len(lagged_data) * val_percent)
        train_data, val_data = random_split(lagged_data, [len(lagged_data) - n_val, n_val])
        loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        loader_val = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        return loader_train, loader_val

    def _fit_estimator(self):
        loader_train = self._format_data(self.data, self.batch_size)
        self.estimator.fit(loader_train, n_epochs=self.epochs)

    def fit_estimator_validation(self):  # Check if model is good
        loader_train, loader_val = self._format_data_val(self.data, self.batch_size)
        self.estimator.fit(loader_train, n_epochs=self.epochs, validation_loader=loader_val)

    def _update_data(self):
        Reap._update_data(self)
        self._fit_estimator()
        self.agent.set_estimator(self.estimator)


class MultiagentVampNetReap(MultiagentReap):
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
                 vnet_lobe=None,
                 vnet_learning_rate=1e-4,
                 vnet_batch_size=64,
                 vnet_epochs=100,
                 vnet_num_threads=1,
                 save_info=False,
                 cluster_args=None
                 ):
        MultiagentReap.__init__(self, system=system, root=root, basename=basename, save_format=save_format,
                                save_rate=save_rate, features=features, cv_weights=cv_weights, delta=delta,
                                n_agents=n_agents, n_candidates=n_candidates, stakes_method=stakes_method,
                                stakes_kwargs=stakes_kwargs, interaction=interaction, save_info=save_info,
                                cluster_args=cluster_args)

        self.lagtime = lagtime
        self.n_features = len(features)  # Must distinguish between features and CVs
        self.batch_size = vnet_batch_size
        self.epochs = vnet_epochs
        self._set_device(device, vnet_num_threads)

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
        if vnet_lobe is None:
            vnet_lobe = MLP(units=[self.n_features, 15, 10, 10, 5, self.cv_num],
                            nonlinearity=nn.ReLU)  # A default model
        vnet_lobe = vnet_lobe.to(self.device)
        self.estimator = VAMPNet(lobe=vnet_lobe, learning_rate=vnet_learning_rate, device=self.device)
        # Initialize agents
        if len(cv_w_shape) == 1:
            self.agent = [AgentVampNetReap(cv_weights, delta, self.estimator)
                          for _ in range(self.n_agents)]
        elif len(cv_w_shape) == 2:
            self.agent = [AgentVampNetReap(cv_weights[i], delta, self.estimator)
                          for i in range(self.n_agents)]

    def _set_device(self, device, num_threads):
        if (device == "cuda") and torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")
        torch.set_num_threads(num_threads)

    def _build_model(self, lobe, learning_rate):
        if lobe is None:
            lobe = MLP(units=[self.n_features, 15, 10, 10, 5, self.cv_num],
                       nonlinearity=nn.ReLU)  # A default model
        lobe = lobe.to(self.device)
        self.estimator = VAMPNet(lobe=lobe, learning_rate=learning_rate, device=self.device)

    def _lagged_dataset(self, data):
        data_float32 = list(map(lambda x: x.astype(np.float32), data))
        return TrajectoryDataset.from_trajectories(self.lagtime, data_float32)

    def _format_data(self, data, batch_size, val_percent=0.3):
        lagged_data = self._lagged_dataset(data)
        loader_train = DataLoader(lagged_data, batch_size=batch_size, shuffle=True)
        return loader_train

    def _format_data_val(self, data, batch_size, val_percent=0.3):  # Include validation split to check if model works
        lagged_data = self._lagged_dataset(data)
        n_val = int(len(lagged_data) * val_percent)
        train_data, val_data = random_split(lagged_data, [len(lagged_data) - n_val, n_val])
        loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        loader_val = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        return loader_train, loader_val

    def _fit_estimator(self):
        loader_train = self._format_data(self.data, self.batch_size)
        self.estimator.fit(loader_train, n_epochs=self.epochs)

    def fit_estimator_validation(self):  # Check if model is good
        loader_train, loader_val = self._format_data_val(self.data, self.batch_size)
        self.estimator.fit(loader_train, n_epochs=self.epochs, validation_loader=loader_val)

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
        self._fit_estimator()
        for n, agent in enumerate(self.agent):
            agent.set_estimator(self.estimator)
            self._update_data_agent(n)
        self._cached_trajs.update(new_fnames)
        self._cached_trajs_ordered.extend(new_fnames)
