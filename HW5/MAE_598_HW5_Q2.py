import matplotlib.pyplot as plt 
import numpy as np
from bayes_opt import BayesianOptimization


def func1(x,y):
    return -((4-2.1*x**2+x**4/3)*x**2+x*y+(-4+4*y**2)*y**2)

pbounds = {'x': (-3, 3), 'y': (-2, 2)}

optimizer = BayesianOptimization(
    f=func1,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

optimizer.maximize(
    init_points=10,
    n_iter=90,
)

print(optimizer.max)