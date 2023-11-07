import numpy as np

from algos.logistic_bandit_algo import LogisticBandit
from utils.optimization import fit_online_logistic_estimate, fit_online_logistic_estimate_bar
from utils.utils import sigmoid, dsigmoid, weighted_norm, gaussian_sample_ellipsoid

"""
Class for the ECOLog algorithm.
Additional Attributes
---------------------
l2_reg: float
    ell-two regularization of maximum-likelihood problem (lambda)
v_tilde_matrix: np.array(dim x dim)
    matrix tilde{V}_t from the paper
v_tilde_inv_matrix: np.array(dim x dim)
    inverse of matrix tilde{V}_t from the paper
theta : np.array(dim)
    online estimator
conf_radius : float
    confidence set radius
cum_loss : float
    cumulative loss between theta and theta_bar
ctr : int
    counter
"""


class EcoLog(LogisticBandit):
    def __init__(self, theta_norm_ub, context_norm_ub, dim, failure_level, num_arms):
        super().__init__(theta_norm_ub, context_norm_ub, dim, failure_level)
        self.name = 'adaECOLog'
        self.num_arms = num_arms
        self.l2reg = 2
        self.vtilde_matrix = [self.l2reg * np.eye(self.dim) for _ in range(self.num_arms)]
        self.vtilde_matrix_inv = [(1 / self.l2reg) * np.eye(self.dim) for _ in range(self.num_arms)]
        self.theta = [np.zeros((self.dim,)) for _ in range(self.num_arms)]
        self.conf_radius = [0 for _ in range(self.num_arms)]
        self.cum_loss = [0 for _ in range(self.num_arms)]
        self.ctr = [1 for _ in range(self.num_arms)]

    def reset(self):
        self.vtilde_matrix = [self.l2reg * np.eye(self.dim) for _ in range(self.num_arms)]
        self.vtilde_matrix_inv = [(1 / self.l2reg) * np.eye(self.dim) for _ in range(self.num_arms)]
        self.theta = [np.zeros((self.dim,)) for _ in range(self.num_arms)]
        self.conf_radius = [0 for _ in range(self.num_arms)]
        self.cum_loss = [0 for _ in range(self.num_arms)]
        self.ctr = [1 for _ in range(self.num_arms)]

    def learn(self, arm, context, reward):
        # compute new estimate theta
        self.theta[arm] = np.real_if_close(fit_online_logistic_estimate(context=context,
                                                                   reward=reward,
                                                                   current_estimate=self.theta[arm],
                                                                   vtilde_matrix=self.vtilde_matrix[arm],
                                                                   vtilde_inv_matrix=self.vtilde_matrix_inv[arm],
                                                                   constraint_set_radius=self.param_norm_ub,
                                                                   diameter=self.param_norm_ub,
                                                                   precision=1/self.ctr[arm]))
        # compute theta_bar (needed for data-dependent conf. width)
        theta_bar = np.real_if_close(fit_online_logistic_estimate_bar(context=context,
                                                                      current_estimate=self.theta[arm],
                                                                      vtilde_matrix=self.vtilde_matrix[arm],
                                                                      vtilde_inv_matrix=self.vtilde_matrix_inv[arm],
                                                                      constraint_set_radius=self.param_norm_ub,
                                                                      diameter=self.param_norm_ub,
                                                                      precision=1/self.ctr[arm]))
        disc_norm = np.clip(weighted_norm(self.theta[arm]-theta_bar, self.vtilde_matrix[arm]), 0, np.inf)

        # update matrices
        sensitivity = dsigmoid(np.dot(self.theta[arm], context))
        self.vtilde_matrix[arm] += sensitivity * np.outer(context, context)
        self.vtilde_matrix_inv[arm] += - sensitivity * np.dot(self.vtilde_matrix_inv[arm],
                                                         np.dot(np.outer(context, context), self.vtilde_matrix_inv[arm])) / (
                                          1 + sensitivity * np.dot(context, np.dot(self.vtilde_matrix_inv[arm], context)))

        # sensitivity check
        sensitivity_bar = dsigmoid(np.dot(theta_bar, context))
        if sensitivity_bar / sensitivity > 2:
            msg = f"\033[95m Oops. ECOLog has a problem: the data-dependent condition was not met. This is rare; try increasing the regularization (self.l2reg) \033[95m"
            raise ValueError(msg)

        # update sum of losses
        coeff_theta = sigmoid(np.dot(self.theta[arm], context))
        loss_theta = -reward * np.log(coeff_theta) - (1-reward) * np.log(1-coeff_theta)
        coeff_bar = sigmoid(np.dot(theta_bar, context))
        loss_theta_bar = -reward * np.log(coeff_bar) - (1-reward) * np.log(1-coeff_bar)
        self.cum_loss[arm] += 2*(1+self.param_norm_ub)*(loss_theta_bar - loss_theta) - 0.5*disc_norm

    def pull(self, context):
        # bonus-based version (strictly equivalent to param-based for this algo) of OL2M
        arm_and_values = list(zip(range(self.num_arms), [self.compute_optimistic_reward(i, context) for i in range(self.num_arms)]))
        # update ctr
        arm = max(arm_and_values, key=lambda x: x[1])[0]
        self.update_ucb_bonus(arm)
        self.ctr[arm] += 1
        return arm

    def update(self, arm, context):
        # bonus-based version (strictly equivalent to param-based for this algo) of OL2M
        self.update_ucb_bonus(arm)
        self.ctr[arm] += 1

    def update_ucb_bonus(self, arm):
        """
        Updates the ucb bonus function (a more precise version of Thm3 in ECOLog paper, refined for the no-warm up alg)
        """
        gamma = np.sqrt(self.l2reg) / 2 + 2 * np.log(
            2 * np.sqrt(1 + self.ctr[arm] / (4 * self.l2reg)) / self.failure_level) / np.sqrt(self.l2reg)
        res_square = 2*self.l2reg*self.param_norm_ub**2 + (1+self.param_norm_ub)**2*gamma + self.cum_loss[arm]
        self.conf_radius[arm] = np.sqrt(res_square)

    def compute_optimistic_reward(self, arm, context):
        """
        Returns prediction + exploration_bonus for arm.
        """
        norm = weighted_norm(self.theta[arm], self.vtilde_matrix_inv[arm])
        pred_reward = sigmoid(np.sum(self.theta[arm] * context))
        # seems to work better after deviding
        bonus = self.conf_radius[arm] * norm / 5
        return pred_reward + bonus