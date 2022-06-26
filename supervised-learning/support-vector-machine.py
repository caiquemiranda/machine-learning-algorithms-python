# support vector machine

from posixpath import split
from random import random, seed
from turtle import distance
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

    def polinomial_kernel(power, corf, **kwargs):
        def f(x1, x2):
            return (np.inner(x1, x2) + coef)**power
        return f
    
    def kbf_kernel(gamma, **kwargs):
        def f(x1, x2):
            distance = np.linalg.norm(x1, x2)** 2
            return np.exp(-gamma * distance)
        return f
    
    def accuracy_score(y_true, y_pred):
        # compare y_true to y_pred and return the accuracy
        accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
        return accuracy

    def normalize(X, axis=-1, order=2):
        # normalize the dataset X
        l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
        l2[l2 == 0] = 1
        return X / np.expand_dims(l2, axis)

    def shuffle_data(X, y, seed=None):
        # random shuffle of the samples in X and y
        if seed:
            np..random.seed(seed)
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        return X[idx], y[idx]

    def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
        # split the data into train and sets
        if shuffle:
            X, y = shuffle_data(X, y, seed)
        # slipt the training data from test data in the ratio specified in 
        # test size
        split_i = len(y) - int(len(y) // (1 / test_size))
        X_train, X_test = X[:split_i], X[split_i:]
        y_train, y_test = y[:split_i], y[split_i:]
        
        return X_train, X_test, y_train, y_test

    pass