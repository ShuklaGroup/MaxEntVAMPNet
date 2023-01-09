import os
import dill as pickle

import numpy as np
from AdaptiveSampling import LeastCountsRegSpace
from Agent import EntropyBasedAgent

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from deeptime.util.torch import MLP
from deeptime.decomposition.deep import VAMPNet
from deeptime.util.data import TrajectoryDataset

from utils import filter_periodic_jumps


class EntropyBasedSampling(LeastCountsRegSpace):
    def __init__(self,
                 system=None,
                 root="",
                 basename="",
                 save_format='.dcd',
                 save_rate=100,
                 features=None,
                 n_candidates=None,
                 save_info=False,
                 cluster_args=None,
                 lagtime=1,
                 device="cuda",
                 vnet_lobe=None,
                 vnet_learning_rate=1e-4,
                 vnet_batch_size=2048,
                 vnet_epochs=100,
                 vnet_num_threads=1,
                 vnet_output_states=None,
                 ):
        LeastCountsRegSpace.__init__(self, system=system, root=root, basename=basename, save_format=save_format,
                                     save_rate=save_rate, features=features, save_info=save_info,
                                     cluster_args=cluster_args)
        self.n_candidates = n_candidates
        self.lagtime = lagtime
        self.n_features = len(features)
        self.n_agents = 1  # Single agent version
        self.batch_size = vnet_batch_size
        self.epochs = vnet_epochs
        self._set_device(device, vnet_num_threads)
        self.output_states = vnet_output_states if vnet_output_states is not None else self.n_features
        # Set estimator
        if vnet_lobe is None:
            vnet_lobe = MLP(units=[self.n_features, 16, 32, 64, 128, 256, 128, 64, 32, 16, self.output_states],
                            nonlinearity=nn.ReLU,
                            output_nonlinearity=nn.Softmax)  # A default model
        vnet_lobe = vnet_lobe.to(self.device)
        self.estimator = VAMPNet(lobe=vnet_lobe, learning_rate=vnet_learning_rate, device=self.device, epsilon=1e-12)
        self.agent = EntropyBasedAgent(self.estimator)

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
        if self.n_candidates is not None:
            least_counts_cluster_indices, candidate_frames_indices = self._find_candidates(self.n_candidates)
            self.states = self.concat_data[candidate_frames_indices]
        else:
            self.states = self.concat_data
            candidate_frames_indices = np.arange(len(self.concat_data), dtype=int)
        self._train_agent()
        scores = self.agent.score()
        assert (len(scores) == len(self.states))
        selected_states = np.argsort(scores)[-n_select:][::-1]
        starting_states = candidate_frames_indices[selected_states]
        starting_states_info = self.fhandler.find_state_info(starting_states,
                                                             self._cached_trajs_ordered,
                                                             self.system.top_file)
        return starting_states_info

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
        self._fit_estimator()
        self.agent.set_estimator(self.estimator)

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

    def _fit_estimator(self):
        loader_train = self._format_data(self.data, self.batch_size)
        self.estimator.fit(loader_train, n_epochs=self.epochs)

    def fit_estimator_validation(self):  # Check if model is good
        loader_train, loader_val = self._format_data_val(self.data, self.batch_size)
        self.estimator.fit(loader_train, n_epochs=self.epochs, validation_loader=loader_val)



