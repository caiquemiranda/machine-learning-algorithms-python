# Algoritmo KNN (K-Nearest Neighbors)

import numpy as np
import math
from collections import Counter

class KNN:
    """
    Implementação do algoritmo K-Nearest Neighbors para classificação e regressão.
    """
    
    def __init__(self, k=5, weights='uniform', algorithm='brute', p=2):
        """
        Inicializa o classificador KNN.
        
        Parâmetros:
        -----------
        k : int
            Número de vizinhos a considerar.
        weights : str
            Tipo de peso a usar: 'uniform' ou 'distance'.
        algorithm : str
            Algoritmo para encontrar vizinhos: 'brute' ou 'kd_tree'.
        p : int
            Parâmetro da distância de Minkowski. p=1 para distância Manhattan,
            p=2 para distância Euclidiana.
        """
        self.k = k
        self.weights = weights
        self.algorithm = algorithm
        self.p = p
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        """
        Treina o classificador KNN.
        
        Parâmetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Amostras de treinamento.
        y : array-like, shape (n_samples,)
            Classes ou valores das amostras de treinamento.
        """
        self.X_train = X
        self.y_train = y
        return self
    
    def minkowski_distance(self, x1, x2):
        """
        Calcula a distância de Minkowski entre dois vetores.
        
        Parâmetros:
        -----------
        x1, x2 : arrays
            Vetores de atributos.
            
        Retorna:
        --------
        distance : float
            Distância de Minkowski.
        """
        if self.p == 1:
            # Manhattan distance
            return np.sum(np.abs(x1 - x2))
        elif self.p == 2:
            # Euclidean distance
            return np.sqrt(np.sum((x1 - x2) ** 2))
        else:
            # Minkowski distance
            return np.power(np.sum(np.power(np.abs(x1 - x2), self.p)), 1/self.p)
    
    def _get_neighbors(self, x):
        """
        Encontra os k vizinhos mais próximos de x.
        
        Parâmetros:
        -----------
        x : array
            Vetor de atributos da amostra.
            
        Retorna:
        --------
        idx : array
            Índices dos k vizinhos mais próximos.
        distances : array
            Distâncias aos k vizinhos mais próximos.
        """
        if self.algorithm == 'brute':
            # Calcula distâncias para todas as amostras
            distances = np.array([self.minkowski_distance(x, x_train) for x_train in self.X_train])
            
            # Pega os índices dos k vizinhos mais próximos
            idx = np.argsort(distances)[:self.k]
            
            return idx, distances[idx]
        
        elif self.algorithm == 'kd_tree':
            # Implementação básica de KD-Tree (apenas para demonstração)
            # Em uma implementação real, usaria a biblioteca scikit-learn
            # ou outra implementação eficiente de KD-Tree
            print("Aviso: A implementação KD-Tree é básica e pode não ser eficiente.")
            distances = np.array([self.minkowski_distance(x, x_train) for x_train in self.X_train])
            idx = np.argsort(distances)[:self.k]
            return idx, distances[idx]
        
        else:
            raise ValueError("Algoritmo não suportado. Use 'brute' ou 'kd_tree'.")
    
    def _vote(self, neighbor_labels, distances):
        """
        Realiza a votação para determinar a classe ou valor da amostra.
        
        Parâmetros:
        -----------
        neighbor_labels : array
            Classes ou valores dos vizinhos mais próximos.
        distances : array
            Distâncias aos vizinhos mais próximos.
            
        Retorna:
        --------
        result : object
            Classe ou valor predito.
        """
        if self.weights == 'uniform':
            # Votação uniforme - cada vizinho tem peso igual
            if len(np.unique(neighbor_labels)) == 1:
                # Se todos os vizinhos têm a mesma classe/valor
                return neighbor_labels[0]
            
            # Caso contrário, conta as ocorrências
            if isinstance(neighbor_labels[0], (int, np.integer, str)):
                # Para classificação, pega a classe mais frequente
                counts = Counter(neighbor_labels)
                return counts.most_common(1)[0][0]
            else:
                # Para regressão, calcula a média
                return np.mean(neighbor_labels)
                
        elif self.weights == 'distance':
            # Votação ponderada pela distância - vizinhos mais próximos têm mais peso
            # Evitar divisão por zero
            distances = np.array(distances)
            distances = np.maximum(distances, np.finfo(float).eps)
            weights = 1.0 / distances
            
            if isinstance(neighbor_labels[0], (int, np.integer, str)):
                # Para classificação, soma os pesos por classe
                weighted_votes = {}
                for label, weight in zip(neighbor_labels, weights):
                    if label in weighted_votes:
                        weighted_votes[label] += weight
                    else:
                        weighted_votes[label] = weight
                return max(weighted_votes.items(), key=lambda x: x[1])[0]
            else:
                # Para regressão, calcula a média ponderada
                return np.sum(weights * neighbor_labels) / np.sum(weights)
                
        else:
            raise ValueError("Peso não suportado. Use 'uniform' ou 'distance'.")
    
    def predict(self, X):
        """
        Prediz as classes ou valores para as amostras em X.
        
        Parâmetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Amostras para predição.
            
        Retorna:
        --------
        y_pred : array, shape (n_samples,)
            Classes ou valores preditos.
        """
        y_pred = np.empty(X.shape[0], dtype=object)
        
        for i, x in enumerate(X):
            # Encontra os k vizinhos mais próximos
            idx, distances = self._get_neighbors(x)
            
            # Pega as classes ou valores dos vizinhos
            neighbor_labels = self.y_train[idx]
            
            # Realiza a votação
            y_pred[i] = self._vote(neighbor_labels, distances)
            
        return y_pred
    
    @staticmethod
    def accuracy_score(y_true, y_pred):
        """
        Calcula a acurácia da predição para classificação.
        
        Parâmetros:
        -----------
        y_true : array-like
            Classes verdadeiras.
        y_pred : array-like
            Classes preditas.
            
        Retorna:
        --------
        accuracy : float
            Acurácia da predição (0.0 a 1.0).
        """
        return np.sum(y_true == y_pred) / len(y_true)
    
    @staticmethod
    def r2_score(y_true, y_pred):
        """
        Calcula o coeficiente de determinação R² para regressão.
        
        Parâmetros:
        -----------
        y_true : array-like
            Valores verdadeiros.
        y_pred : array-like
            Valores preditos.
            
        Retorna:
        --------
        r2 : float
            Coeficiente de determinação.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        mean_y = np.mean(y_true)
        ss_total = np.sum((y_true - mean_y) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        
        if ss_total == 0:
            return 1  # Modelo perfeito se todos os valores são iguais
        
        return 1 - (ss_residual / ss_total)
    
    @staticmethod
    def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
        """
        Divide os dados em conjuntos de treino e teste.
        
        Parâmetros:
        -----------
        X : array-like
            Atributos.
        y : array-like
            Classes ou valores.
        test_size : float
            Proporção do conjunto de teste.
        shuffle : bool
            Se True, embaralha os dados antes da divisão.
        random_state : int
            Semente para reprodutibilidade.
            
        Retorna:
        --------
        X_train, X_test, y_train, y_test : arrays
            Conjuntos de treino e teste.
        """
        if random_state:
            np.random.seed(random_state)
            
        if shuffle:
            indices = np.random.permutation(len(y))
            X, y = X[indices], y[indices]
            
        split_idx = int(len(y) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test