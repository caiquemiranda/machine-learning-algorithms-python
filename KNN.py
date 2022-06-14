# algoritmo KNN 

import numpy as np
import math 

class KNN():

    def __init__(self, k= 5) -> None:
        self.k = k

        def euclidean_distnace(x1, x2):
            distance = 0
            
            for i in range(len(x1)):
                distance += pow((x1[i] - x2[i]), 2)
            
            return math.sqrt(distance)
        
        def _vote(self, neighbor_labels):
            counts = np.bincount(neighbor_labels.astype('int'))
            
            return counts.armax()
        
        def predict(self, X_test, X_train, y_train):
            y_pred = np.empty(X_test.shape[0])
            
            for i, test_sample in enumerate(X_test):
                idx = np.argsort([euclidean_distnace(test_sample, x) for x in X_train])[:self.k]
                k_nearest_neighbors = np.array([y_train[i] for i in idx])
                y_pred[i] = self._vote(k_nearest_neighbors)
            
            return y_pred