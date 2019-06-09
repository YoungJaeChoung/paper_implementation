# http://research.sualab.com/introduction/practice/2019/04/01/bayesian-optimization-overview-2.html

from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import numpy as np


def target(x):
    return np.exp(-(x-3)**2) + np.exp(-(3*x-2)**2) + 1/(x**2+1)


def plot_exp_func():
    path_fig = "/Users/youngjae/Documents/analysis/git/paper_implementation/hyperParameterOptimization/figures"
    x = np.arange(-10, 10, 0.01)
    y = target(x)
    plt.plot(x, y, 'o-', markersize=0.5)
    plt.savefig(path_fig + '/' + 'exp_func' + '.png')
    plt.close()

# plot_exp_func()


if __name__ == "__main__":
    """ BayesianOptimization
    
    Params
    ------
    -. f: target function
    -. pbounds (dict): bounds of input 
            i.e. {'x': (-2, 6)}                   
    -. random_state (int): 
        > default: None
    -. verbose (int): 
        > default: None
    
    Returns
    -------
    -. object 
    
    """
    bayes_optimizer = BayesianOptimization(target, {'x': (-10, 10)},
                                           random_state=0)

    """ maximize func of BayesianOptimization object
    
    Params 
    ------
    -. init_points (float): the number of initial random search points  
        > default: 5
    -. n_iter (int): the number of iteration 
        > default: 25
    -. acq: the name of 'acq'uisition function
        > i.e. 'ei': Expected Improvement 
        > default: 'ucb'
    -. kappa: If UCB is to be used, a constant kappa is needed.
        > default: 2.576
    -. xi: the parameter of 'exploration'. 0.01 is recommended to explore
        > default: 0.0
    -. **gp_params
    
    Returns 
    -------
    
    """
    bayes_optimizer.maximize(init_points=2, n_iter=30, acq='ei', xi=0.01)
    print(bayes_optimizer.max)  # maximum value obtained using EI (expected improvement)

    # real maximum value
    x = np.arange(-10, 10, 0.0001)
    y = target(x)
    np.max(y)


