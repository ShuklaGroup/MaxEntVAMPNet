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

    def __init__(self, arg1):
        pass

    @abstractmethod
    def set_data(self):
        """
        Assign data to agent.
        :return:
        """
        pass

    @abstractmethod
    def train(self):
        """
        This function learns the parameters that the agent will use to score a structure.
        :return:
        """
        pass

    @abstractmethod
    def score(self):
        """
        This function scores a structure.
        :return:
        """
        pass


class AgentReap(Agent):

    def __init__(self, cv_weights, delta, data=None):
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

    def get_log_info(self):
        '''
        Returns a dictionary with certain object attributes for logging purposes.
        :return:
        '''
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

    def set_data(self, data):
        if self.data is None:
            self.data = data
        else:
            self.data.extend(data)
        self.data_concat = np.concatenate(self.data, axis=0)
        self.means = self.data_concat.mean(axis=0)
        self.stdev = self.data_concat.std(axis=0)

    def set_states(self, states):
        """
        :param states: Usually, central frames of clusters.
        :return:
        """
        self.states = states

    def set_stakes(self, stakes):
        """
        Necessary only for multiagent runs.
        :param stakes:
        :return:
        """
        self.stakes = stakes

    def score(self, cv_weights=None):
        """
        Score takes the role of the reward function. In the case of REAP, this is based on the standardized Euclidean
        distance.
        :return:
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

    def train(self):  # TODO: constraint the maximum reward an agent can give
        """
        Get new CV weights by maximizing score.
        :return:
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

    def __init__(self, cv_weights, delta, estimator, data=None, propagation_steps=10):
        # Note that here the term CV is (inaccurately) used to refer to VAMP-reduced coordinates.
        # cv_weights must have length equal to the number of coordinates we want to keep, not to the number of total
        # features
        AgentReap.__init__(self, cv_weights, delta, data)
        self.estimator = estimator
        self.propagation_steps = propagation_steps
        if self.data is not None:
            self.transformed_data = self.estimator.transform(self.data_concat)

    def get_log_info(self):
        '''
        Returns a dictionary with certain object attributes for logging purposes.
        :return: logs (dict).
        '''
        logs = AgentReap.get_log_info(self)
        logs |= dict(
            estimator=self.estimator,
            propagation_steps=self.propagation_steps,
        )

        return logs

    def set_data(self, data):
        AgentReap.set_data(self, data)
        self.transformed_data = self.estimator.transform(self.data_concat)
        self.means = self.transformed_data.mean(axis=0)
        self.stdev = self.transformed_data.std(axis=0)

    def set_estimator(self, estimator):
        self.estimator = estimator

    def set_states(self, states):
        self.states = self._propagate_kinetic_model(states)

    def _propagate_kinetic_model(self, frames):
        """
        Propagate the candidate frames using the kinetic model.
        :return:
        """
        model = self.estimator.fetch_model()
        inst_obs = model.transform(frames)
        propagated = inst_obs @ (model.operator ** self.propagation_steps)[:self.cv_num, :self.cv_num]
        # propagated = model._instantaneous_whitening_backwards(propagated) --> The score is calculated in CV space
        return propagated


class AgentVampNetReap(AgentVampReap):

    def _propagate_kinetic_model(self, frames):
        """
        Propagate the candidate frames using the kinetic model.
        :return:
        """
        model = self.estimator.fetch_model()
        propagated = model.transform(frames)
        return propagated


class AgentVaeReap(AgentVampNetReap):
    # It's the same implementation as the base class, I just wanted to have a class with a different name
    pass


class EntropyBasedAgent(Agent):

    def __init__(self, estimator, data=None):
        self.estimator = estimator
        self.data = data
        self.data_concat = None
        self.transformed_data = None
        if self.data is not None:
            self.data_concat = np.concatenate(data, axis=0)
            self.transformed_data = estimator.model.transform(self.data_concat)
        self.states = self.transformed_data

    def get_log_info(self):
        """
        Returns a dictionary with certain object attributes for logging purposes.
        :return:
        """
        logs = dict(
            estimator=self.estimator,
            scores=self.score(),
        )

        return logs

    def set_data(self, data):
        if self.data is None:
            self.data = data
        else:
            self.data.extend(data)
        self.data_concat = np.concatenate(self.data, axis=0)
        self.transformed_data = self.estimator.model.transform(self.data_concat)

    def set_estimator(self, estimator):
        self.estimator = estimator

    def set_states(self, states):
        self.states = self.estimator.transform(states)

    def score(self):
        return entropy(self.states, axis=1)

    def train(self):
        pass
