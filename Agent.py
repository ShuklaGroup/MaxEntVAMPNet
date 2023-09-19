"""
Definition of the Agent object for REAP-like implementations.
The agent possess data and parameters that allows it to score conformations or trajectories.
"""
import sys
from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize
from scipy.stats import entropy


class Agent(ABC):
    """Abstract class for Agent object.
    """
    def __init__(self, arg1):
        pass

    @abstractmethod
    def set_data(self):
        """Assign data to agent.
        :return:
        """
        pass

    @abstractmethod
    def train(self):
        """Learn the parameters that the agent will use to score a structure.
        :return:
        """
        pass

    @abstractmethod
    def score(self):
        """Score a structure.
        :return:
        """
        pass


class AgentReap(Agent):
    """Agent class for REAP or MA REAP simulations.
    """

    def __init__(self, cv_weights=None, delta=None, data=None, logs=None):
        """Constructor for AgentReap.

        :param cv_weights: list[float].
            Weights for collective variables.
        :param delta: float in [0, 1].
            Max change in collective variable weights between rounds.
        :param data: list[np.ndarray].
            Initial trajectory data for agent. Not necessary.
        """
        if logs is None:
            assert (np.allclose(sum(cv_weights), 1))  # CV weights must add up to 1
            assert (0 <= delta <= 1)  # delta must be between 0 and 1
            self.cv_weights = cv_weights
            self.delta = delta
            self.cv_num = len(cv_weights)
            self.data = data
            self.data_concat = None
            self.means = None
            self.stdev = None
            if self.data is not None:
                self.data_concat = np.concatenate(data, axis=0)
                self.means = self.data_concat.mean(axis=0)
                self.stdev = self.data_concat.std(axis=0)
            self.states = None
            self.stakes = None
        elif isinstance(logs, dict):
            self._reload(logs)

    def get_log_info(self):
        """Returns a dictionary with certain object attributes for logging purposes.

        :return: dict.
            Logs.
        """
        logs = dict(
            cv_weights=self.cv_weights,
            delta=self.delta,
            cv_num=self.cv_num,
            means=self.means,
            stdev=self.stdev,
            states=self.states,
            stakes=self.stakes,
            scores=self.score()
        )

        return logs

    def _reload(self, logs):
        self.cv_weights = logs['cv_weights']
        self.delta = logs['delta']
        self.cv_num = logs['cv_num']
        self.means = logs['means']
        self.stdev = logs['stdev']
        self.states = logs['states']
        self.stakes = logs['stakes']

    def set_data(self, data):
        """Set data (trajectories) for the agent.

        :param data: list[np.ndarray].
            Trajectory data.
        :return: None.
        """
        if self.data is None:
            self.data = data
        else:
            self.data.extend(data)
        self.data_concat = np.concatenate(self.data, axis=0)
        self.means = self.data_concat.mean(axis=0)
        self.stdev = self.data_concat.std(axis=0)

    def set_states(self, states):
        """Set states that agent can select to restart simulations.

        :param states: np.ndarray of shape (n_states, ndim).
            Usually, central frames of clusters.
        :return: None.
        """
        self.states = states

    def set_stakes(self, stakes):
        """Set agent stakes. Necessary only for multiagent runs.

        :param stakes: np.ndarray.
        :return: None.
        """
        self.stakes = stakes

    def score(self, cv_weights=None):
        """Score takes the role of the reward function. In the case of REAP, this is based on the standardized Euclidean
        distance.

        :return: np.ndarray of shape (n_states,).
            Scores for each state.
        """
        if self.stakes is None:
            stakes = np.ones(len(self.states))
        else:
            stakes = self.stakes

        if cv_weights is None:
            cv_weights = self.cv_weights

        epsilon = sys.float_info.epsilon
        dist = np.abs(self.states - self.means[:self.cv_num])
        distances = (cv_weights * dist / (self.stdev[:self.cv_num] + epsilon)).sum(axis=1)

        return stakes * distances

    def train(self):
        """Get new CV weights by maximizing REAP score.

        :return: None.
        """
        weights_prev = self.cv_weights
        delta = self.delta
        # Create constraints
        constraints = [
            # Inequality constraints (fun(x, *args) >= 0)
            # This constraint makes the weights change by delta (at most)
            {
                'type': 'ineq',
                'fun': lambda weights, weights_prev, delta: delta - np.abs((weights_prev - weights)),
                'jac': lambda weights, weights_prev, delta: np.diagflat(np.sign(weights_prev - weights)),
                'args': (weights_prev, delta),
            },
            # This constraint makes the weights be always positive
            {
                'type': 'ineq',
                'fun': lambda weights: weights,
                'jac': lambda weights: np.eye(weights.shape[0]),
            },
            # Equality constraints (fun(x, *args) = 0)
            # This constraint makes sure the weights add up to one
            {
                'type': 'eq',
                'fun': lambda weights: weights.sum() - 1,
                'jac': lambda weights: np.ones(weights.shape[0]),
            }]

        def minimize_helper(x):
            return -self.score(cv_weights=x).sum()

        results = minimize(minimize_helper, weights_prev, method='SLSQP', constraints=constraints)

        self.cv_weights = results.x


class AgentVampReap(AgentReap):
    """Agent class for VAMP + MA REAP simulations.
    """

    def __init__(self, cv_weights, delta, estimator, data=None, propagation_steps=1):
        """Constructor for AgentVampReap.

        :param cv_weights: list[float].
            Weights for collective variables.
        :param delta: float in [0, 1].
            Max change in collective variable weights between rounds.
        :param estimator: deeptime.decomposition._vamp.VAMP.
            Estimator object for dimensionality reduction.
        :param data: list[np.ndarray].
            Initial trajectory data for agent. Not necessary.
        :param propagation_steps: int, default = 1.
            This option allows to apply the Koopman operator propagation_steps times to the input conformation.
        """
        # Note that here the term CV is (inaccurately) used to refer to VAMP-reduced coordinates.
        # cv_weights must have length equal to the number of coordinates we want to keep, not to the number of total
        # features
        AgentReap.__init__(self, cv_weights, delta, data)
        self.estimator = estimator
        self.propagation_steps = propagation_steps
        if self.data is not None:
            self.transformed_data = self.estimator.transform(self.data_concat)

    def get_log_info(self):
        """Returns a dictionary with certain object attributes for logging purposes.

        :return: dict.
            Logs.
        """
        logs = AgentReap.get_log_info(self)
        logs |= dict(
            estimator=self.estimator,
            propagation_steps=self.propagation_steps,
        )

        return logs

    def set_data(self, data):
        """Set data (trajectories) for the agent.

        :param data: list[np.ndarray].
            Trajectory data.
        :return: None.
        """
        AgentReap.set_data(self, data)
        self.transformed_data = self.estimator.transform(self.data_concat)
        self.means = self.transformed_data.mean(axis=0)
        self.stdev = self.transformed_data.std(axis=0)

    def set_estimator(self, estimator):
        """Set the estimator used by the agent. Needed to update the estimator once it has been fit to new data.

        :param estimator: deeptime.decomposition._vamp.VAMP.
            Estimator object for dimensionality reduction.
        :return: None.
        """
        self.estimator = estimator

    def set_states(self, states):
        """Set states that agent can select to restart simulations.

        :param states: np.ndarray of shape (n_states, ndim).
            Usually, central frames of clusters.
        :return: None.
        """
        self.states = self._propagate_kinetic_model(states)

    def _propagate_kinetic_model(self, frames):
        """Propagate the candidate frames using the kinetic model.

        :param frames: np.ndarray of shape (n_frames, n_features).
            Frames to transform applying Koopman operator.
        :return: np.ndarray of shape (n_frames, ndim).
            Transformed frames.
        """
        model = self.estimator.fetch_model()
        inst_obs = model.transform(frames)
        propagated = inst_obs @ (model.operator ** self.propagation_steps)[:self.cv_num, :self.cv_num]
        # propagated = model._instantaneous_whitening_backwards(propagated) --> The score is calculated in CV space
        return propagated


class AgentVampNetReap(AgentVampReap):
    """Agent class for VAMPNet + MA REAP simulations.

    Note that the self.estimator attribute will be a deeptime.util.torch.MLP object for this class.
    """
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


class AgentVaeReap(AgentVampNetReap):
    """Identical implementation to AgentVampNetReap with a different name.

    Note that the self.estimator attribute will be a deeptime.decomposition.deep._tae.TVAE object for this class.
    """
    pass


class EntropyBasedAgent(Agent):
    """Agent class for VAMPNet + MaxEnt simulations.
    """

    def __init__(self, estimator=None, logs=None, data=None):
        """Constructor for EntropyBasedAgent.

        :param estimator: deeptime.util.torch.MLP.
            VAMPNet estimator. Note that the output non-linearity must be softmax or similar.
        :param data: list[np.ndarray].
            Initial trajectory data for agent. Not necessary.
        """
        if logs is None:
            self.estimator = estimator
            self.data = data
            self.data_concat = None
            self.transformed_data = None
            if self.data is not None:
                self.data_concat = np.concatenate(data, axis=0)
                self.transformed_data = estimator.model.transform(self.data_concat)
            self.states = self.transformed_data
        elif isinstance(logs, dict):
            self._reload(logs)

    def get_log_info(self):
        """Returns a dictionary with certain object attributes for logging purposes.

        :return: dict.
            Logs.
        """
        logs = dict(
            estimator=self.estimator,
            scores=self.score(),
        )

        return logs

    def _reload(self, logs):
        self.estimator = logs['estimator']

    def set_data(self, data):
        """Set data (trajectories) for the agent.

        :param data: list[np.ndarray].
            Trajectory data.
        :return: None.
        """
        if self.data is None:
            self.data = data
        else:
            self.data.extend(data)
        self.data_concat = np.concatenate(self.data, axis=0)
        self.transformed_data = self.estimator.model.transform(self.data_concat)

    def set_estimator(self, estimator):
        """Set the estimator used by the agent. Needed to update the estimator once it has been fit to new data.

        :param estimator: deeptime.util.torch.MLP.
            Estimator object for soft discretization.
        :return: None.
        """
        self.estimator = estimator

    def set_states(self, states):
        """Set states that agent can select to restart simulations.

        :param states: np.ndarray of shape (n_states, ndim).
            Since MaxEnt does not require discretization, any conformation can be considered a state.
        :return: None.
        """
        self.states = self.estimator.transform(states)

    def score(self):
        """Compute score for MaxEnt (Shannon entropy).

        :return: np.ndarray of shape (n_states,).
            Shannon entropy of the conformations considered as states, potentially all conformations.
        """
        return entropy(self.states, axis=1)

    def train(self):
        pass
