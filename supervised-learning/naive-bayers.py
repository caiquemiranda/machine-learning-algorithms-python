# Naive Bayes

import numpy as np
from collections import defaultdict

class NaiveBayes:
    """
    Implementação do algoritmo Naive Bayes para classificação.
    Suporta tanto distribuição Gaussiana (para atributos contínuos) quanto
    distribuição de frequência (para atributos categóricos).
    """
    
    def __init__(self, distribution="gaussian"):
        """
        Inicializa o classificador Naive Bayes
        
        Parâmetros:
        -----------
        distribution : str
            O tipo de distribuição a ser usada para os atributos.
            "gaussian" para atributos contínuos, "categorical" para atributos categóricos.
        """
        self.distribution = distribution
        self.classes = None
        self.parameters = {}
        self.priors = {}
        self.n_features = None
        
    def _calculate_prior(self, y):
        """
        Calcula as probabilidades a priori de cada classe
        """
        _, counts = np.unique(y, return_counts=True)
        self.priors = counts / len(y)
        
    def _calculate_likelihood_gaussian(self, X, y):
        """
        Calcula os parâmetros da distribuição Gaussiana (média e variância)
        para atributos contínuos.
        """
        self.classes = np.unique(y)
        self.n_features = X.shape[1]
        
        # Para cada classe, calcular média e variância de cada atributo
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.parameters[c] = {
                'mean': np.mean(X_c, axis=0),
                'var': np.var(X_c, axis=0) + 1e-10  # Adicionamos uma pequena constante para evitar divisão por zero
            }
    
    def _calculate_likelihood_categorical(self, X, y):
        """
        Calcula as probabilidades para atributos categóricos
        """
        self.classes = np.unique(y)
        self.n_features = X.shape[1]
        
        # Para cada classe e cada atributo, contar ocorrências de cada valor
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.parameters[c] = []
            
            # Para cada atributo
            for j in range(self.n_features):
                # Contar ocorrências de cada valor
                values, counts = np.unique(X_c[:, j], return_counts=True)
                # Calcular probabilidades (com suavização de Laplace)
                probs = defaultdict(lambda: 1)  # Contagem inicial de 1 para todos os valores (suavização)
                for val, count in zip(values, counts):
                    probs[val] = count + 1  # Adiciona 1 para suavização
                    
                # Normalizar probabilidades
                total = sum(probs.values())
                for val in probs:
                    probs[val] /= total
                    
                self.parameters[c].append(probs)
                
    def fit(self, X, y):
        """
        Treina o classificador Naive Bayes
        
        Parâmetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Amostras de treinamento
        y : array-like, shape (n_samples,)
            Classes das amostras de treinamento
        """
        self._calculate_prior(y)
        
        if self.distribution == "gaussian":
            self._calculate_likelihood_gaussian(X, y)
        elif self.distribution == "categorical":
            self._calculate_likelihood_categorical(X, y)
        else:
            raise ValueError("Distribuição não suportada. Use 'gaussian' ou 'categorical'.")
            
        return self
    
    def _calculate_posterior_gaussian(self, x):
        """
        Calcula a probabilidade posterior para cada classe usando
        distribuição Gaussiana para atributos contínuos.
        """
        posteriors = []
        
        # Calcular probabilidade para cada classe
        for i, c in enumerate(self.classes):
            prior = np.log(self.priors[i])
            class_conditional = 0
            
            # Para cada atributo, calcular probabilidade usando distribuição Gaussiana
            for j in range(self.n_features):
                mean = self.parameters[c]['mean'][j]
                var = self.parameters[c]['var'][j]
                
                # Log da função de densidade de probabilidade Gaussiana
                class_conditional += -0.5 * np.log(2 * np.pi * var)
                class_conditional += -0.5 * ((x[j] - mean) ** 2) / var
                
            posterior = prior + class_conditional
            posteriors.append(posterior)
            
        return posteriors
    
    def _calculate_posterior_categorical(self, x):
        """
        Calcula a probabilidade posterior para cada classe usando
        frequências para atributos categóricos.
        """
        posteriors = []
        
        # Calcular probabilidade para cada classe
        for i, c in enumerate(self.classes):
            prior = np.log(self.priors[i])
            class_conditional = 0
            
            # Para cada atributo, buscar probabilidade pré-calculada
            for j in range(self.n_features):
                class_conditional += np.log(self.parameters[c][j][x[j]])
                
            posterior = prior + class_conditional
            posteriors.append(posterior)
            
        return posteriors
    
    def predict(self, X):
        """
        Prediz as classes para as amostras em X
        
        Parâmetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Amostras para predição
            
        Retorna:
        --------
        y_pred : array, shape (n_samples,)
            Classes preditas
        """
        y_pred = []
        
        for x in X:
            if self.distribution == "gaussian":
                posteriors = self._calculate_posterior_gaussian(x)
            else:
                posteriors = self._calculate_posterior_categorical(x)
                
            # Classe com maior probabilidade posterior
            y_pred.append(self.classes[np.argmax(posteriors)])
            
        return np.array(y_pred)
    
    @staticmethod
    def accuracy_score(y_true, y_pred):
        """
        Calcula a acurácia da predição
        
        Parâmetros:
        -----------
        y_true : array-like
            Classes verdadeiras
        y_pred : array-like
            Classes preditas
            
        Retorna:
        --------
        accuracy : float
            Acurácia da predição (0.0 a 1.0)
        """
        return np.sum(y_true == y_pred) / len(y_true)
    
    @staticmethod
    def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
        """
        Divide os dados em conjuntos de treino e teste
        
        Parâmetros:
        -----------
        X : array-like
            Atributos
        y : array-like
            Classes
        test_size : float
            Proporção do conjunto de teste
        shuffle : bool
            Se True, embaralha os dados antes da divisão
        random_state : int
            Semente para reprodutibilidade
            
        Retorna:
        --------
        X_train, X_test, y_train, y_test : arrays
            Conjuntos de treino e teste
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
