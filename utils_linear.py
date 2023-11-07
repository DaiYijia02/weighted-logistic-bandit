import numpy as np
import scipy.stats
from ipywidgets import Button, HBox
import itertools
from plot_ellipse import plot_ellipse


class LinearContextualBandit():
    def __init__(self, K=3, rng=None, d=2):
        self.rng = np.random if rng is None else rng
        self.K = K
        self.d = d
         
        # Generate d dim vectors
        self.mus = [np.sqrt(10)*self.rng.randn(d) for _ in range(K)]
        # Observation noise is clipped Gaussian
        self.sigma = 1
        self.noise = lambda: np.clip(self.rng.randn(),-10,10)
    
    def pull(self, arm, context):
        return np.dot(self.mus[arm], context) + self.noise()
    
    def get_context(self):
        return np.sqrt(10)*self.rng.randn(self.d)

class WeightedLinearContextualBandit():
    def __init__(self, K=3, rng=None, d=2):
        self.rng = np.random if rng is None else rng
        self.K = K
        self.d = d
         
        # Generate d dim vectors
        self.mus = [np.sqrt(10)*self.rng.randn(d) for _ in range(K)]
        # Observation noise is clipped Gaussian
        self.sigma = 1
        self.noise = lambda: np.clip(self.rng.randn(),-10,10)
        self.weight = 1/2
    
    def pull(self, arm, context):
        return self.weight * np.dot(self.mus[arm], context) + self.noise()
    
    def get_context(self):
        return np.sqrt(10)*self.rng.randn(self.d)


class LinearRewardHistory():

    def __init__(self, actions, d):
        self.history = np.empty([0,2+d])
        self.actions = actions
        self.d = d
        self.per_action_history = {action: [] for action in actions}
        self.ci = {action: (None, None, None) for action in actions} # mean, Sigma
        self.T = 0
    
    def record(self, action, context, reward):
        self.history = np.append(self.history, [np.hstack([action, context, reward])], axis=0)
        self.per_action_history[action].append(np.hstack([context, reward]))
        self.T += 1

    def compute_ci(self):
        for action in self.actions:
            N = len(self.per_action_history[action])
            if N >= 1:
                matrix = np.array(self.per_action_history[action])
                A = matrix[:,:-1]
                b = matrix[:,-1]
                mean, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            else:
                mean = None
            if N >= 1*self.d:
                _,s,v = np.linalg.svd(A)
            else:
                s, v = None, None  
            self.ci[action] = (mean, s, v)
            
    def get_unexplored_actions(self, N=0):
        ret = [a for a in self.actions if len(self.per_action_history[a])<(N+1)*d]
        return ret

    def get_smallest_N(self):
        min_N = min([len(self.per_action_history[a]) for a in self.actions])
        ret = [a for a in self.actions if len(self.per_action_history[a]) <= min_N][0]
        return ret
         
    def get_means(self, context=None):
        self.compute_ci()
        if context is None:
            ret = np.array([self.ci[a][0] for a in self.actions])
        else:
            ret = np.array([np.dot(self.ci[a][0], context) for a in self.actions])
        return ret

    def get_highest_mean(self, context):
        self.compute_ci()
        max_r = max([np.dot(self.ci[a][0], context) for a in self.actions])
        ret = [a for a in self.actions if np.dot(self.ci[a][0], context) >= max_r][0]
        return ret
    
    def get_ucb(self, context):
        self.compute_ci()
        ucb_list = [np.dot(self.ci[a][0], context) + np.linalg.norm(context @ np.diag(1/self.ci[a][1]) @ self.ci[a][2] @ context) for a in self.actions]
        ucb = max(ucb_list)
        ret = [a for a in self.actions if ucb_list[a] >= ucb][0]
        return ret

class WeightedLinearRewardHistory():

    def __init__(self, actions, d):
        self.history = np.empty([0,2+d])
        self.actions = actions
        self.d = d
        self.per_action_history = {action: [] for action in actions}
        self.ci = {action: (None, None, None) for action in actions} # mean, Sigma
        self.T = 0
        self.weight = 1/2
    
    def record(self, action, context, reward):
        self.history = np.append(self.history, [np.hstack([action, context, reward])], axis=0)
        self.per_action_history[action].append(np.hstack([context, reward]))
        self.T += 1

    def compute_ci(self):
        for action in self.actions:
            N = len(self.per_action_history[action])
            if N >= 1:
                matrix = np.array(self.per_action_history[action])
                A = matrix[:,:-1]
                b = matrix[:,-1]
                mean, _, _, _ = np.linalg.lstsq(self.weight * A, b, rcond=None)
            else:
                mean = None
            if N >= 1*self.d:
                _,s,v = np.linalg.svd(A)
            else:
                s, v = None, None  
            self.ci[action] = (mean, s, v)
            
    def get_unexplored_actions(self, N=0):
        ret = [a for a in self.actions if len(self.per_action_history[a])<(N+1)*d]
        return ret

    def get_smallest_N(self):
        min_N = min([len(self.per_action_history[a]) for a in self.actions])
        ret = [a for a in self.actions if len(self.per_action_history[a]) <= min_N][0]
        return ret
         
    def get_means(self, context=None):
        self.compute_ci()
        if context is None:
            ret = np.array([self.ci[a][0] for a in self.actions])
        else:
            ret = np.array([np.dot(self.ci[a][0], context) for a in self.actions])
        return ret

    def get_highest_mean(self, context):
        self.compute_ci()
        max_r = max([np.dot(self.ci[a][0], context) for a in self.actions])
        ret = [a for a in self.actions if np.dot(self.ci[a][0], context) >= max_r][0]
        return ret
    
    def get_ucb(self, context):
        self.compute_ci()
        ucb_list = [np.dot(self.ci[a][0], context) + np.linalg.norm(context @ np.diag(1/self.ci[a][1]) @ self.ci[a][2] @ context) for a in self.actions]
        ucb = max(ucb_list)
        ret = [a for a in self.actions if ucb_list[a] >= ucb][0]
        return ret


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
        mean, s, u = hist.ci[arm]
        if mean is not None:
            # mean
            ax.scatter(mean[0], mean[1], color=colors[arm], label=arm)
            ax.plot([0, mean[0]], [0, mean[1]], color=colors[arm])
            xmin = min(xmin, mean[0])
            xmax = max(xmax, mean[0])
            ymin = min(ymin, mean[1])
            ymax = max(ymax, mean[1])
        
            # confidence ellipse
            if s is not None:
                plot_ellipse(ax, cov=u.T@np.diag(1/s**2)@u, x_cent=mean[0], y_cent=mean[1], plot_kwargs={'alpha':0}, fill=True,
                fill_kwargs={'color':colors[arm],'alpha':0.1})
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
        policy_bottons = [Button(description='ArgMax'), Button(description='LinUCB'), Button(description='Optimal')]
        self.combined = HBox([items for items in arm_buttons] + [reveal_button] + policy_bottons)
        
        self.ax = axs[0]
        self.ax2 = axs[1]
        self.colors = ['r', 'm', 'b', 'c', 'g', 'y']
        
        for n in range(mab.K):
            arm_buttons[n].on_click(self.upon_clicked)
        reveal_button.on_click(self.upon_reveal)
        for b in policy_bottons:
            b.on_click(self.upon_policy)
        
        self.context = self.mab.get_context()
        update_plot(self.ax, self.hist, self.mab.K, self.context, self.colors) 
        
    def upon_clicked(self, btn):
        arm = int(btn.description)
        reward = self.mab.pull(arm, self.context)
        self.hist.record(arm, self.context, reward)
        self.hist.compute_ci()
        self.context = self.mab.get_context()
        update_plot(self.ax, self.hist, self.mab.K, self.context, self.colors) 
    
    def upon_reveal(self, b):
        xs = [mu[0] for mu in self.mab.mus]
        ys = [mu[1] for mu in self.mab.mus]
        self.ax.scatter(xs, ys, marker="*", c=self.colors[0:self.mab.K])
        
    def upon_policy(self, b):
        if b.description == 'Optimal':
            plot_policy(self.ax2, self.hist, self.colors, self.context, mab=self.mab)
        else:
            plot_policy(self.ax2, self.hist, self.colors, self.context, ucb=(b.description == 'LinUCB'))

def plot_policy(ax, hist, colors, context, ucb=False, mab=None):
    x = np.linspace(-10,10,100+1)
    y = np.linspace(-10,10,100+1)
    zz = get_policy(x, y, hist, ucb=ucb, mab=mab)
    ax.clear()
    ax.contourf(x, y, zz, colors=colors, levels=[-0.5,0.5,1.5,2.5], alpha=0.5)
    ax.scatter(context[0], context[1], color='black', marker='x')
    ax.set_title('policy')
    
def get_policy(xs, ys, hist, ucb=False, mab=None):
    zz = np.zeros([len(xs), len(ys)])
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            context = np.array([x,y])
            zz[i,j] = get_argmax(context, hist, ucb=ucb, mab=mab)
    return zz.T


def get_argmax(context, hist, ucb=False, mab=None):
    ests = []
    for arm in hist.actions:
        if mab is None:
            mean, s, u = hist.ci[arm]
            if mean is not None and not ucb:
                ests.append(np.dot(mean, context))
            elif ucb and s is not None:
                pass
                est = np.dot(mean, context) + np.linalg.norm(np.diag(1/s) @ u @ context)
                ests.append(est)
            else:
                ests.append(-np.inf)
        else:
            ests.append(np.dot(mab.mus[arm], context))
    if np.max(ests) == np.inf:
        return None
    else:
        return np.argmax(ests)

        
def greedy_policy(hist):
    unexplored = hist.get_unexplored_actions()
    if len(unexplored) > 0:
        return np.random.choice(unexplored)
    return hist.get_highest_mean()