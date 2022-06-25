# support vector machine

import numpy as np
import cvxopt
from mlfromscratch.utils import Plot


vcxopt.solvers.options['show_progress'] = False

class SupportVectorMachine(object):

    def __init__(self, C=1, kernel=rbf_kernel, power=1, gamma=None, coef=4): -> None:
        self.C = C
        self.kernel = kernel
        self.power = power
        self.gamma = gamma
        self.coef = coef
        self.lagr_multpliers = None
        self.support_vector_labels = None
        self.intercept = None
    
    def linear_kernel(**kwargs):
        def f(x1,x2):
            return np.inner(x1, x2)
        return f

    pass