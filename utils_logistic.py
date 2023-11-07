import numpy as np
import scipy.stats
from ipywidgets import Button, HBox
import itertools
from plot_ellipse import plot_ellipse
from algos.ecolog import EcoLog
from algos.ecolog_ranking import EcoLogRanking

class LogisticContextualBandit():
    def __init__(self, K=3, rng=None, d=2, mus=None):
        self.rng = np.random if rng is None else rng
        self.K = K
        self.d = d
        self.ctr = 0
         
        # Generate d dim vectors
        if mus != None:
            self.mus = mus
        else: 
            self.mus = [self.rng.randn(d) for _ in range(K)]
        
        # Observation noise is clipped Gaussian
        # self.sigma = 1
        # self.noise = lambda: np.clip(self.rng.randn(),-10,10)
    
    def pull(self, arm, context):
        def sigmoid(x):
            return 1/(1+np.exp(-x))
        expect = sigmoid(np.dot(self.mus[arm], context))
        reward = int(np.random.uniform(0, 1) < expect)
        optimal = max([sigmoid(np.dot(self.mus[i], context)) for i in range(self.K)])
        regret = optimal-expect
        return reward, regret, expect
    
    def get_cyclic_context(self):
        # cyclic state of context, change only by one dimension at once
        context = [np.array([3, 0]), np.array([0, 3])]
        if self.ctr == 0:
            self.ctr = 1
            return context[0]
        else:
            self.ctr = 0
            return context[1]

    def get_random_context(self):
        return self.rng.randn(self.d)


class LogisticContextualBanditRanking():
    def __init__(self, K=3, rng=None, d=2, top_k=None, mus=None, rank_weight=False):
        self.rng = np.random if rng is None else rng
        self.K = K
        self.d = d
        self.top_k = top_k
        self.ctr = 0
         
        # Generate d dim vectors
        if mus != None:
            self.mus = mus
        else: 
            self.mus = [self.rng.randn(d) for _ in range(K)]
        
        # Observation noise is clipped Gaussian
        # self.sigma = 1
        # self.noise = lambda: np.clip(self.rng.randn(),-10,10)

        # Ranking score is known
        if top_k != None:
            self.rank_weight = [int(i < top_k) for i in range(K)]
        elif rank_weight==True:
            self.rank_weight = [1/(i+1) for i in range(K)]
        else:
            self.rank_weight = [1/2 for i in range(K)]
    
    def pull(self, ranking, context):
        def sigmoid(x):
            return 1/(1+np.exp(-x))
        total_reward = [0 for _ in range(self.K)]
        total_expect = 0
        ground_truth = [0 for _ in range(self.K)]
        if len(ranking) != self.K:
            for j in range(self.K):
                expect = sigmoid(np.dot(self.mus[j], context))
                ground_truth[j] = expect
        for i in range(len(ranking)):
            arm = ranking[i]
            expect = sigmoid(np.dot(self.mus[arm], context))
            ground_truth[arm] = expect
            reward = int(np.random.uniform(0, 1) < self.rank_weight[i] * expect)
            total_expect += self.rank_weight[i] * expect
            total_reward[arm] = self.rank_weight[i] * reward
        opt_ground_truth = [sigmoid(np.dot(self.mus[i], context)) for i in range(self.K)]
        opt_ground_truth.sort(reverse = True)
        optimal = np.sum([a*b for a,b in zip(opt_ground_truth,self.rank_weight)])
        regret = optimal-total_expect
        return total_reward, regret, ground_truth
    
    def get_cyclic_context(self):
        # cyclic state of context, change only by one dimension at once
        context = [np.array([1, 0]), np.array([0, 1])]
        if self.ctr == 0:
            self.ctr = 1
            return context[0]
        else:
            self.ctr = 0
            return context[1]

    def get_random_context(self):
        return self.rng.randn(self.d)


class LogisticRewardHistory():

    def __init__(self, K, rng, d):
        self.history = np.empty([0,2+d])
        self.actions = np.arange(K)
        self.d = d
        self.per_action_history = {action: [] for action in self.actions}
        self.ci = {action: (None, None) for action in self.actions} # mean, Sigma
        self.T = 0
        self.policy = EcoLog(theta_norm_ub=np.sqrt(10*d),
                      context_norm_ub=np.sqrt(10*d),
                      dim=d,
                      failure_level=0.05,
                      num_arms=K)
        # self.policy = EcoLogRanking(theta_norm_ub=np.sqrt(10*d),
        #               context_norm_ub=np.sqrt(10*d),
        #               dim=d,
        #               failure_level=0.05,
        #               num_arms=K,
        #               top_k=None,
        #               rank_weight=[1/2 for i in range(K)])
    
    def record(self, action, context, reward):
        self.history = np.append(self.history, [np.hstack([action, context, reward])], axis=0)
        self.per_action_history[action].append(np.hstack([context, reward]))
        self.T += 1
    
    def compute_ci_for_graph(self, arm, context, reward):
        self.policy.update(arm, context)
        self.policy.learn(arm, context, reward)
        # self.policy.update([arm])
        # self.policy.learn([arm], context, [reward])
        N = len(self.per_action_history[arm])
        if N >= 1:
            mean = self.policy.theta[arm]
        else:
            mean = None
        if N >= 1*self.d:
            # use the inverse matrix in ecolog
            inv = self.policy.vtilde_matrix_inv[arm]*self.policy.conf_radius[arm]
        else:
            inv = None
        self.ci[arm] = (mean, inv)

    def alg_pull(self, context):
        arm = self.policy.pull(context)
        return arm

    def alg_learn(self, arm, context, reward):
        self.policy.learn(arm, context, reward)
        self.policy.update(arm, context)


class LogisticRewardHistoryRanking():
    def __init__(self, K, rng, d, top_k, rank_weight):
        self.history = np.empty([0,K*2+d])
        self.actions = np.arange(K)
        self.d = d
        self.top_k = top_k
        self.rank_weight = rank_weight
        self.ci = {action: (None, None) for action in self.actions} # mean, Sigma
        self.T = 0
        self.policy = EcoLogRanking(theta_norm_ub=np.sqrt(10*d),
                      context_norm_ub=np.sqrt(10*d),
                      dim=d,
                      failure_level=0.05,
                      num_arms=K,
                      top_k=top_k,
                      rank_weight=rank_weight)
    
    def record(self, action, context, reward):
        action = np.asarray(action).flatten()
        context = context.flatten()
        self.history = np.append(self.history, np.reshape(np.concatenate((action, context, reward)),(1,-1)), axis=0)
        self.T += 1
    
    def compute_ci_for_graph(self, arms, context, reward):
        self.policy.update(arms)
        self.policy.learn(arms, context, reward)
        N = len(self.per_action_history[arm])
        if N >= 1:
            mean = self.policy.theta[arm]
        else:
            mean = None
        if N >= 1*self.d:
            # use the inverse matrix in ecolog
            inv = self.policy.vtilde_matrix_inv[arm]*self.policy.conf_radius[arm]
        else:
            inv = None
        self.ci[arm] = (mean, inv)

    def alg_pull(self, context):
        ranking = self.policy.pull(context)
        return ranking

    def alg_learn(self, ranking, context, reward):
        self.policy.learn(ranking, context, reward)
        self.policy.update(ranking)

class OfflineLearning():
    def __init__(self, K, rng, d, top_k, rank_weight, history):
        self.actions = np.arange(K)
        self.K = K
        self.d = d
        self.top_k = top_k
        self.rank_weight = rank_weight
        self.offline_history = history
        self.history = np.empty([0,K*2+d])
        self.ci = {action: (None, None) for action in self.actions} # mean, Sigma
        self.T = 0
        self.policy = EcoLogRanking(theta_norm_ub=np.sqrt(10*d),
                      context_norm_ub=np.sqrt(10*d),
                      dim=d,
                      failure_level=0.05,
                      num_arms=K,
                      top_k=top_k,
                      rank_weight=rank_weight)
    
    def record(self, action, context, reward):
        action = np.asarray(action).flatten()
        self.history = np.append(self.history, np.reshape(np.concatenate((action, context, reward)),(1,-1)), axis=0)
        self.T += 1

    def offline_pull(self, timestep):
        ranking = self.offline_history[timestep][:self.K]
        context = self.offline_history[timestep][self.K:self.K+self.d]
        ranking = tuple(ranking.astype(int))
        return ranking, context

    def alg_learn(self, ranking, context, reward):
        self.policy.learn(ranking, context, reward)
        self.policy.update(ranking)

class AutoExperiment():
    def __init__(self, mab, hist, horizon, warm_up, cyclic=False):
        self.mab = mab
        self.hist = hist
        self.horizon = horizon
        self.warm_up = warm_up
        self.cyclic = cyclic
    
    def run(self):
        reward_hist = []
        regret_hist = []
        for arm in range(self.mab.K):
            for _ in range(self.warm_up):
                if self.cyclic:
                    context = self.mab.get_cyclic_context()
                else:
                    context = self.mab.get_random_context()
                reward, regret, ground_truth = self.mab.pull(arm, context)
                self.hist.alg_learn(arm, context, reward)
                reward_hist.append(reward)
                regret_hist.append(regret)
        for _ in range(self.horizon):
            if self.cyclic:
                context = self.mab.get_cyclic_context()
            else:
                context = self.mab.get_random_context()
            arm = self.hist.alg_pull(context)
            reward, regret, ground_truth = self.mab.pull(arm, context)
            self.hist.alg_learn(arm, context, reward)
            reward_hist.append(reward)
            regret_hist.append(regret)
        return reward_hist, regret_hist


class AutoExperimentRanking():
    def __init__(self, mab, hist, horizon, warm_up, cyclic=False):
        self.mab = mab
        self.hist = hist
        self.horizon = horizon
        self.warm_up = warm_up
        self.cyclic = cyclic

    def estimate_with_bonus(self):
        estimate_hists = [[] for _ in range(self.mab.K)]
        bonus_hists = [[] for _ in range(self.mab.K)]
        ground_truth_hists = [[] for _ in range(self.mab.K)]
        reward_hist = []
        regret_hist = []
        if self.mab.top_k == None:
            rankings = list(itertools.permutations([i for i in range(self.mab.K)]))
            for ranking in rankings:
                for _ in range(self.warm_up):
                    if self.cyclic:
                        context = self.mab.get_cyclic_context()
                    else:
                        context = self.mab.get_random_context()
                    reward, regret, ground_truth = self.mab.pull(ranking, context)
                    self.hist.alg_learn(ranking, context, reward)
                    self.hist.record(ranking, context, reward)
                    reward_hist.append(reward)
                    regret_hist.append(regret)
                    for i in range(self.mab.K):
                        estimate_hists[ranking[i]].append((i, self.hist.policy.compute_pred(ranking[i], context, i))) # ranking position starts from 0
                        bonus_hists[ranking[i]].append((i, self.hist.policy.compute_bonus(ranking[i])))
                        ground_truth_hists[ranking[i]].append((i, ground_truth[ranking[i]]))
        else:
            for arm in range(self.mab.K):
                for _ in range(self.warm_up):
                    if self.cyclic:
                        context = self.mab.get_cyclic_context()
                    else:
                        context = self.mab.get_random_context()
                    ranking = [arm] # for semi-bandit
                    reward, regret, ground_truth = self.mab.pull(ranking, context)
                    reward_hist.append(reward)
                    regret_hist.append(regret)
                    self.hist.alg_learn(ranking, context, reward)
                    self.hist.record(ranking, context, reward)
                    estimate_hists[arm].append((0, self.hist.policy.compute_pred(arm, context))) # ranking position starts from 0
                    bonus_hists[arm].append((0, self.hist.policy.compute_bonus(arm)))
                    ground_truth_hists[arm].append((0, ground_truth[arm]))
                    no_pull = set([i for i in range(self.mab.K)]) - set([arm])
                    no_pull = list(no_pull)
                    for i in range(len(no_pull)):
                        estimate_hists[no_pull[i]].append((-1, self.hist.policy.compute_pred(no_pull[i], context))) # ranking position starts from 0
                        bonus_hists[no_pull[i]].append((-1, self.hist.policy.compute_bonus(no_pull[i])))
                        ground_truth_hists[no_pull[i]].append((i, ground_truth[no_pull[i]]))
        for _ in range(self.horizon):
            if self.cyclic:
                context = self.mab.get_cyclic_context()
            else:
                context = self.mab.get_random_context()
            ranking = self.hist.alg_pull(context)
            reward, regret, ground_truth = self.mab.pull(ranking, context)
            self.hist.alg_learn(ranking, context, reward)
            self.hist.record(ranking, context, reward)
            reward_hist.append(reward)
            regret_hist.append(regret)
            for i in range(len(ranking)):
                estimate_hists[ranking[i]].append((i, self.hist.policy.compute_pred(ranking[i], context, i))) # ranking position starts from 0
                bonus_hists[ranking[i]].append((i, self.hist.policy.compute_bonus(ranking[i])))
                ground_truth_hists[ranking[i]].append((i, ground_truth[ranking[i]]))
            no_pull = set([i for i in range(self.mab.K)]) - set(ranking)
            no_pull = list(no_pull)
            for i in range(len(no_pull)):
                estimate_hists[no_pull[i]].append((-1, self.hist.policy.compute_pred(no_pull[i], context, i))) # ranking position starts from 0
                bonus_hists[no_pull[i]].append((-1, self.hist.policy.compute_bonus(no_pull[i]))) 
                ground_truth_hists[no_pull[i]].append((i, ground_truth[no_pull[i]]))
        return estimate_hists, bonus_hists, ground_truth_hists, reward_hist, regret_hist       
    
    def run(self):
        reward_hist = []
        regret_hist = []
        if self.mab.top_k == None:
            rankings = list(itertools.permutations([i for i in range(self.mab.K)]))
            for ranking in rankings:
                for _ in range(self.warm_up):
                    if self.cyclic:
                        context = self.mab.get_cyclic_context()
                    else:
                        context = self.mab.get_random_context()
                    reward, regret, ground_truth = self.mab.pull(ranking, context)
                    self.hist.alg_learn(ranking, context, reward)
                    self.hist.record(ranking, context, reward)
                    reward_hist.append(reward)
                    regret_hist.append(regret)
        else:
            for arm in range(self.mab.K):
                for _ in range(self.warm_up):
                    if self.cyclic:
                        context = self.mab.get_cyclic_context()
                    else:
                        context = self.mab.get_random_context()
                    ranking = [arm] # for semi-bandit
                    reward, regret, ground_truth = self.mab.pull(ranking, context)
                    self.hist.alg_learn(ranking, context, reward)
                    self.hist.record(ranking, context, reward)
                    reward_hist.append(reward)
                    regret_hist.append(regret)
        for _ in range(self.horizon):
            if self.cyclic:
                context = self.mab.get_cyclic_context()
            else:
                context = self.mab.get_random_context()
            ranking = self.hist.alg_pull(context)
            reward, regret, ground_truth = self.mab.pull(ranking, context)
            self.hist.alg_learn(ranking, context, reward)
            self.hist.record(ranking, context, reward)
            reward_hist.append(reward)
            regret_hist.append(regret)
        return reward_hist, regret_hist

class AutoExperimentRankingOffline():
    # current version only support cyclic context, for consistency
    def __init__(self, mab, hist, cyclic=True):
        self.mab = mab
        self.hist = hist
        self.cyclic = cyclic

    def estimate_with_bonus(self):
        estimate_hists = [[] for _ in range(self.mab.K)]
        bonus_hists = [[] for _ in range(self.mab.K)]
        ground_truth_hists = [[] for _ in range(self.mab.K)]
        reward_hist = []
        regret_hist = []
        offline_hist = self.hist.offline_history
        for t in range(len(offline_hist)):
            ranking, context = self.hist.offline_pull(t)
            reward, regret, ground_truth = self.mab.pull(ranking, context)
            self.hist.alg_learn(ranking, context, reward)
            reward_hist.append(reward)
            regret_hist.append(regret)
            for i in range(len(ranking)):
                estimate_hists[ranking[i]].append((i, self.hist.policy.compute_pred(ranking[i], context, i))) # ranking position starts from 0
                bonus_hists[ranking[i]].append((i, self.hist.policy.compute_bonus(ranking[i])))
                ground_truth_hists[ranking[i]].append((i, ground_truth[ranking[i]]))
            no_pull = set([i for i in range(self.mab.K)]) - set(ranking)
            no_pull = list(no_pull)
            for i in range(len(no_pull)):
                estimate_hists[no_pull[i]].append((-1, self.hist.policy.compute_pred(no_pull[i], context, i))) # ranking position starts from 0
                bonus_hists[no_pull[i]].append((-1, self.hist.policy.compute_bonus(no_pull[i]))) 
                ground_truth_hists[no_pull[i]].append((i, ground_truth[no_pull[i]]))
        return estimate_hists, bonus_hists, ground_truth_hists, reward_hist, regret_hist

class AverageAutoExperiment():
    def __init__(self, K, top_k, horizon, warm_up, mab_class, hist_class, runner_class, seeds = 10):
        self.K = K
        self.top_k = top_k
        self.horizon = horizon
        self.warm_up = warm_up
        self.mab_class = mab_class
        self.hist_class = hist_class
        self.runner_class = runner_class
        self.seeds = seeds
    
    def run(self):
        reward_stack = []
        regret_stack = []
        for _ in range(self.seeds):
            try:
                mab = self.mab_class(K=self.K, top_k=self.top_k)
                hist = self.hist_class(mab.K, mab.rng, mab.d, mab.top_k, mab.rank_weight)
                runner = self.runner_class(mab, hist, self.horizon, self.warm_up)
                reward, regret = runner.run()
                reward_stack.append(reward)
                regret_stack.append(regret)
            except Exception as e:
                print(f'caught {type(e)}: e')
                continue
        reward_stack = np.array(reward_stack)
        regret_stack = np.array(regret_stack)
        reward_stack.mean(axis=1)
        regret_stack.mean(axis=1)
        return reward_stack, regret_stack

def update_plot(ax, hist, K, context, colors):
    ax.clear()
    
    plotted_arms = 0
    xmin, xmax = (-5,5)
    ymin, ymax = (-5,5)
    for arm in range(K):
        # current context
        ax.scatter(context[0], context[1], color='black', marker='x')
        
        # observed contexts
        if len(hist.per_action_history[arm]) > 0:
            contexts = np.array(hist.per_action_history[arm])[:,:-1]
            ax.scatter(contexts[:,0], contexts[:,1], color=colors[arm], marker='x', alpha=0.5)
            xmin = min(xmin, np.min(contexts[:,0]))
            xmax = max(xmax, np.max(contexts[:,0]))
            ymin = min(ymin, np.min(contexts[:,1]))
            ymax = max(ymax, np.max(contexts[:,1]))
        
        # estimated parameters
        mean, inv = hist.ci[arm]
        # mean, s, u = hist.ci[arm]
        if mean is not None:
            # mean
            ax.scatter(mean[0], mean[1], color=colors[arm], label=arm)
            ax.plot([0, mean[0]], [0, mean[1]], color=colors[arm])
            xmin = min(xmin, mean[0])
            xmax = max(xmax, mean[0])
            ymin = min(ymin, mean[1])
            ymax = max(ymax, mean[1])
        
            # confidence ellipse
            if inv is not None:
            # if s is not None:
                plot_ellipse(ax, cov=inv, x_cent=mean[0], y_cent=mean[1], plot_kwargs={'alpha':0}, fill=True,
                fill_kwargs={'color':colors[arm],'alpha':0.1})
                # plot_ellipse(ax, cov=u.T@np.diag(1/s)@u, x_cent=mean[0], y_cent=mean[1], plot_kwargs={'alpha':0}, fill=True,
                # fill_kwargs={'color':colors[arm],'alpha':0.1})
            plotted_arms += 1

    ax.set_title("contexts and estimated parameters")
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])
    if plotted_arms>0: ax.legend(title='arm')
    ax.grid()


class InteractivePlot():
    def __init__(self, mab, hist, axs):
        self.mab = mab
        self.hist = hist
        arm_buttons = [Button(description=str(arm)) for arm in np.arange(mab.K)]
        reveal_button = Button(description='Reveal')
        policy_bottons = [Button(description='EcoLog'), Button(description='Optimal')]
        self.combined = HBox([items for items in arm_buttons] + [reveal_button] + policy_bottons)
        
        self.ax = axs[0]
        self.ax2 = axs[1]
        self.colors = ['r', 'm', 'b', 'c', 'g', 'y']
        
        for n in range(mab.K):
            arm_buttons[n].on_click(self.upon_clicked)
        reveal_button.on_click(self.upon_reveal)
        for b in policy_bottons:
            b.on_click(self.upon_policy)
        
        # self.context = self.mab.get_cyclic_context()
        self.context = self.mab.get_random_context()
        update_plot(self.ax, self.hist, self.mab.K, self.context, self.colors) 
        
    def upon_clicked(self, btn):
        arm = int(btn.description)
        reward, regret, ground_truth = self.mab.pull(arm, self.context)
        self.hist.record(arm, self.context, reward)
        self.hist.compute_ci_for_graph(arm, self.context, reward)
        # self.context = self.mab.get_cyclic_context()
        self.context = self.mab.get_random_context()
        update_plot(self.ax, self.hist, self.mab.K, self.context, self.colors) 
    
    def upon_reveal(self, b):
        xs = [mu[0] for mu in self.mab.mus]
        ys = [mu[1] for mu in self.mab.mus]
        self.ax.scatter(xs, ys, marker="*", c=self.colors[0:self.mab.K])
        
    def upon_policy(self, b):
        if b.description == 'Optimal':
            plot_policy(self.ax2, self.hist, self.colors, self.context, mab=self.mab)
        else:
            plot_policy(self.ax2, self.hist, self.colors, self.context, alg=(b.description == 'EcoLog'))

def plot_policy(ax, hist, colors, context, alg=False, mab=None):
    x = np.linspace(-10,10,100+1)
    y = np.linspace(-10,10,100+1)
    zz = get_policy(x, y, hist, alg=alg, mab=mab)
    ax.clear()
    ax.contourf(x, y, zz, colors=colors, levels=[-0.5,0.5,1.5,2.5], alpha=0.5)
    ax.scatter(context[0], context[1], color='black', marker='x')
    ax.set_title('policy')
    
def get_policy(xs, ys, hist, alg=False, mab=None):
    zz = np.zeros([len(xs), len(ys)])
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            context = np.array([x,y])
            zz[i,j] = get_argmax(context, hist, alg=alg, mab=mab)
    return zz.T


def get_argmax(context, hist, alg=False, mab=None):
    ests = []
    for arm in hist.actions:
        if mab is None:
            mean, inv = hist.ci[arm]
            # mean, s, u = hist.ci[arm]
            if mean is not None and not alg:
                ests.append(np.dot(mean, context))
            elif alg and inv is not None:
                est = hist.policy.compute_optimistic_reward(arm, context)
                ests.append(est)
            else:
                ests.append(-np.inf)
        else:
            ests.append(np.dot(mab.mus[arm], context))
    if np.max(ests) == np.inf:
        return None
    else:
        return np.argmax(ests)
