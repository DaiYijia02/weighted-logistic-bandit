import numpy as np

from algos.logistic_bandit_algo import LogisticBandit
from utils.optimization import fit_online_logistic_estimate, fit_online_logistic_estimate_bar, fit_online_logistic_estimate_ranking, fit_online_logistic_estimate_bar_ranking
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
top_k: int
"""


class EcoLogRanking(LogisticBandit):
    def __init__(self, theta_norm_ub, context_norm_ub, dim, failure_level, num_arms, top_k=None, rank_weight=None):
        super().__init__(theta_norm_ub, context_norm_ub, dim, failure_level)
        self.name = 'adaECOLogRanking'
        self.num_arms = num_arms
        self.top_k = top_k
        self.rank_weight = rank_weight
        self.l2reg = 3
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

    def learn(self, arms, context, rewards):
        # compute new estimate theta
        for i in range(len(arms)):
            arm = arms[i]
            # self.theta[arm] = np.real_if_close(fit_online_logistic_estimate(context=context,
            #                                                            reward=rewards[arm],
            #                                                            current_estimate=self.theta[arm],
            #                                                            vtilde_matrix=self.vtilde_matrix[arm],
            #                                                            vtilde_inv_matrix=self.vtilde_matrix_inv[arm],
            #                                                            constraint_set_radius=self.param_norm_ub,
            #                                                            diameter=self.param_norm_ub,
            #                                                            precision=1/self.ctr[arm]))
            # # compute theta_bar (needed for data-dependent conf. width)
            # theta_bar = np.real_if_close(fit_online_logistic_estimate_bar(context=context,
            #                                                               current_estimate=self.theta[arm],
            #                                                               vtilde_matrix=self.vtilde_matrix[arm],
            #                                                               vtilde_inv_matrix=self.vtilde_matrix_inv[arm],
            #                                                               constraint_set_radius=self.param_norm_ub,
            #                                                               diameter=self.param_norm_ub,
            #                                                               precision=1/self.ctr[arm]))
            self.theta[arm] = np.real_if_close(fit_online_logistic_estimate_ranking(context=context,
                                                                       reward=rewards[arm],
                                                                       ranking_weight=self.rank_weight[i],
                                                                       current_estimate=self.theta[arm],
                                                                       vtilde_matrix=self.vtilde_matrix[arm],
                                                                       vtilde_inv_matrix=self.vtilde_matrix_inv[arm],
                                                                       constraint_set_radius=self.param_norm_ub,
                                                                       diameter=self.param_norm_ub,
                                                                       precision=1/self.ctr[arm]))
            # compute theta_bar (needed for data-dependent conf. width)
            theta_bar = np.real_if_close(fit_online_logistic_estimate_bar_ranking(context=context,
                                                                          current_estimate=self.theta[arm],
                                                                          ranking_weight=self.rank_weight[i],
                                                                          vtilde_matrix=self.vtilde_matrix[arm],
                                                                          vtilde_inv_matrix=self.vtilde_matrix_inv[arm],
                                                                          constraint_set_radius=self.param_norm_ub,
                                                                          diameter=self.param_norm_ub,
                                                                          precision=1/self.ctr[arm]))
            disc_norm = np.clip(weighted_norm(self.theta[arm]-theta_bar, self.vtilde_matrix[arm]), 0, np.inf)

            # # version 1: without any rank_weight
            # # update matrices
            sensitivity = dsigmoid(np.dot(self.theta[arm], context))
            # self.vtilde_matrix[arm] += sensitivity * np.outer(context, context)
            # self.vtilde_matrix_inv[arm] += - sensitivity * np.dot(self.vtilde_matrix_inv[arm],
            #                                                  np.dot(np.outer(context, context), self.vtilde_matrix_inv[arm])) / (
            #                                   1 + sensitivity * np.dot(context, np.dot(self.vtilde_matrix_inv[arm], context))) # Sherman–Morrison
            # # sensitivity check
            # sensitivity_bar = dsigmoid(np.dot(theta_bar, context))
            # if sensitivity_bar / sensitivity > 2:
            #     msg = f"\033[95m Oops. ECOLog has a problem: the data-dependent condition was not met. This is rare; try increasing the regularization (self.l2reg) \033[95m"
            #     raise ValueError(msg)
            
            # # update sum of losses
            # coeff_theta = sigmoid(np.dot(self.theta[arm], context))
            # loss_theta = -rewards[arm] * np.log(coeff_theta) - (1-rewards[arm]) * np.log(1-coeff_theta)
            # coeff_bar = sigmoid(np.dot(theta_bar, context))
            # loss_theta_bar = -rewards[arm] * np.log(coeff_bar) - (1-rewards[arm]) * np.log(1-coeff_bar)
            # self.cum_loss[arm] += 2*(1+self.param_norm_ub)*(loss_theta_bar - loss_theta) - 0.5*disc_norm

            # version 2: with rank_weight
            # update matrices
            # sensitivity = self.rank_weight[i]**2 * dsigmoid(np.dot(self.theta[arm], context))
            
            # trials! for the regularizer with rank_weight
            rank_boost = 0
            # rank_boost = (1 - self.rank_weight[i]) * self.l2reg
            # rank_boost = (1 - self.rank_weight[i]) / np.exp(-np.dot(self.theta[arm], context)) * self.l2reg / 10
            self.vtilde_matrix[arm] += sensitivity * np.outer(context, context) + rank_boost * np.eye(self.dim)
            # first add the original iteration
            self.vtilde_matrix_inv[arm] += - sensitivity * np.dot(self.vtilde_matrix_inv[arm],
                                                             np.dot(np.outer(context, context), self.vtilde_matrix_inv[arm])) / (
                                              1 + sensitivity * np.dot(context, np.dot(self.vtilde_matrix_inv[arm], context))) # Sherman–Morrison
            # then add the second iteration term
            self.vtilde_matrix_inv[arm] += - rank_boost * np.dot(self.vtilde_matrix_inv[arm], self.vtilde_matrix_inv[arm]) / (
                                              1 + rank_boost * np.dot(np.ones(self.dim), np.dot(self.vtilde_matrix_inv[arm], np.ones(self.dim)))) # Sherman–Morrison
            
            # sensitivity check
            sensitivity_bar = self.rank_weight[i]**2 * dsigmoid(np.dot(theta_bar, context))
            if sensitivity_bar / sensitivity > 2:
                msg = f"\033[95m Oops. ECOLog has a problem: the data-dependent condition was not met. This is rare; try increasing the regularization (self.l2reg) \033[95m"
                raise ValueError(msg)

            # update sum of losses
            coeff_theta = np.exp(-np.dot(self.theta[arm], context))
            loss_theta = - np.log(1 + coeff_theta) + rewards[arm] * np.log(self.rank_weight[i]) + (1-rewards[arm]) * np.log(1+coeff_theta-self.rank_weight[i])
            coeff_bar = np.exp(-np.dot(theta_bar, context))
            loss_theta_bar = - np.log(1 + coeff_bar) + rewards[arm] * np.log(self.rank_weight[i]) + (1-rewards[arm]) * np.log(1+coeff_bar-self.rank_weight[i])
            self.cum_loss[arm] += 2*(1+self.param_norm_ub)*(loss_theta_bar - loss_theta) - 0.5*disc_norm

    def pull(self, context):
        # bonus-based version (strictly equivalent to param-based for this algo) of OL2M
        arms_and_values = list(zip(range(self.num_arms), [self.compute_optimistic_reward(i, context) for i in range(self.num_arms)]))
        arms_and_values.sort(key=lambda x: x[1], reverse = True)
        arms = []
        num_pulls = self.top_k if self.top_k!=None else self.num_arms
        for i in range(num_pulls):
            arm = arms_and_values[i][0]
            arms.append(arm)
        return arms

    def update(self, ranking):
        # bonus-based version (strictly equivalent to param-based for this algo) of OL2M
        for arm in ranking:
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
        bonus = self.conf_radius[arm] * norm / 18
        return pred_reward + bonus

    def compute_bonus(self, arm):
        norm = weighted_norm(self.theta[arm], self.vtilde_matrix_inv[arm])
        return self.conf_radius[arm] * norm / 18

    def compute_pred(self, arm, context, ranking):
        pred_reward = sigmoid(np.sum(self.theta[arm] * context))
        # pred_reward = sigmoid(np.sum(self.theta[arm] * context)) / self.rank_weight[ranking]
        return pred_reward