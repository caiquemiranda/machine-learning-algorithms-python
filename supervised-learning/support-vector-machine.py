# support vector machine

import numpy as np
import cvxopt
import matplotlib.pyplot as plt

# Desabilitar saída do progresso do solucionador cvxopt
cvxopt.solvers.options['show_progress'] = False

class SupportVectorMachine(object):

    def __init__(self, C=1, kernel='rbf', power=4, gamma=None, coef=0):
        self.C = C
        self.kernel = kernel
        self.power = power
        self.gamma = gamma
        self.coef = coef
        self.lagr_multipliers = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.intercept = None
    
    @staticmethod
    def linear_kernel(**kwargs):
        def f(x1, x2):
            return np.inner(x1, x2)
        return f

    @staticmethod
    def polynomial_kernel(power, coef, **kwargs):
        def f(x1, x2):
            return (np.inner(x1, x2) + coef)**power
        return f
    
    @staticmethod
    def rbf_kernel(gamma, **kwargs):
        def f(x1, x2):
            distance = np.linalg.norm(x1 - x2) ** 2
            return np.exp(-gamma * distance)
        return f

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Selecionar kernel apropriado
        if self.kernel == 'linear':
            self.kernel_func = self.linear_kernel()
        elif self.kernel == 'polynomial':
            self.kernel_func = self.polynomial_kernel(power=self.power, coef=self.coef)
        elif self.kernel == 'rbf':
            if self.gamma is None:
                self.gamma = 1 / n_features
            self.kernel_func = self.rbf_kernel(gamma=self.gamma)
            
        # Matriz de kernel
        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self.kernel_func(X[i], X[j])
                
        # Resolvendo usando programação quadrática (QP)
        y = np.array(y, dtype=np.float64)
        
        # Formato para cvxopt: min 0.5 x^T P x + q^T x, Gx <= h, Ax = b
        P = cvxopt.matrix(np.outer(y, y) * kernel_matrix, tc='d')
        q = cvxopt.matrix(-np.ones(n_samples), tc='d')
        
        # -alpha_i <= 0
        G_std = cvxopt.matrix(-np.eye(n_samples), tc='d')
        h_std = cvxopt.matrix(np.zeros(n_samples), tc='d')
        
        # alpha_i <= C
        G_slack = cvxopt.matrix(np.eye(n_samples), tc='d')
        h_slack = cvxopt.matrix(np.ones(n_samples) * self.C, tc='d')
        
        G = cvxopt.matrix(np.vstack((G_std, G_slack)), tc='d')
        h = cvxopt.matrix(np.vstack((h_std, h_slack)), tc='d')
        
        A = cvxopt.matrix(y.reshape(1, -1), tc='d')
        b = cvxopt.matrix(np.zeros(1), tc='d')
        
        # Resolver problema de otimização
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        
        # Lagrange multipliers
        a = np.array(solution['x']).reshape(-1)
        
        # Extrair support vectors (SVs)
        # SVs têm lagrange multipliers > 0
        sv_threshold = 1e-5
        sv_indices = a > sv_threshold
        
        self.lagr_multipliers = a[sv_indices]
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]
        
        # Calcular intercept
        self.intercept = 0
        for i in range(len(self.lagr_multipliers)):
            self.intercept += self.support_vector_labels[i]
            self.intercept -= np.sum(self.lagr_multipliers * self.support_vector_labels * 
                              np.apply_along_axis(lambda x: self.kernel_func(self.support_vectors[i], x), 1, self.support_vectors))
        self.intercept /= len(self.lagr_multipliers)
        
        return self
    
    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)
        
        for i in range(n_samples):
            prediction = 0
            for alpha, sv_y, sv in zip(self.lagr_multipliers, self.support_vector_labels, self.support_vectors):
                prediction += alpha * sv_y * self.kernel_func(X[i], sv)
            y_pred[i] = prediction
            
        return np.sign(y_pred + self.intercept)
    
    @staticmethod
    def accuracy_score(y_true, y_pred):
        # compare y_true to y_pred and return the accuracy
        accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
        return accuracy

    @staticmethod
    def normalize(X, axis=-1, order=2):
        # normalize the dataset X
        l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
        l2[l2 == 0] = 1
        return X / np.expand_dims(l2, axis)

    @staticmethod
    def shuffle_data(X, y, seed=None):
        # random shuffle of the samples in X and y
        if seed:
            np.random.seed(seed)
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        return X[idx], y[idx]

    @staticmethod
    def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
        # split the data into train and test sets
        if shuffle:
            X, y = SupportVectorMachine.shuffle_data(X, y, seed)
        # split the training data from test data in the ratio specified in 
        # test size
        split_i = len(y) - int(len(y) // (1 / test_size))
        X_train, X_test = X[:split_i], X[split_i:]
        y_train, y_test = y[:split_i], y[split_i:]
        
        return X_train, X_test, y_train, y_test