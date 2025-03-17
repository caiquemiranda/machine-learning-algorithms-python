# Regressão Linear

import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    """
    Implementação do algoritmo de Regressão Linear.
    Suporta regressão linear simples e múltipla,
    com opção de regularização L2 (Ridge).
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, alpha=0, 
                 fit_intercept=True, method='normal_equation'):
        """
        Inicializa o modelo de Regressão Linear.
        
        Parâmetros:
        -----------
        learning_rate : float
            Taxa de aprendizado para o método de gradiente descendente.
        n_iterations : int
            Número de iterações para o método de gradiente descendente.
        alpha : float
            Parâmetro de regularização L2 (Ridge).
        fit_intercept : bool
            Se True, adiciona um termo de interceptação.
        method : str
            Método para encontrar os parâmetros: 'normal_equation' ou 'gradient_descent'.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.method = method
        self.weights = None
        self.intercept = None
        self.cost_history = []
        
    def _add_intercept(self, X):
        """
        Adiciona uma coluna de 1's para o intercepto.
        """
        return np.c_[np.ones((X.shape[0], 1)), X]
    
    def _normal_equation(self, X, y):
        """
        Usa a equação normal para encontrar os pesos ótimos.
        w = (X^T X + alpha*I)^-1 X^T y
        """
        n_samples, n_features = X.shape
        
        # Adicionar regularização L2 (Ridge)
        I = np.identity(n_features)
        I[0, 0] = 0  # Não regulariza o intercepto
        
        # Calcular pesos usando a equação normal
        XtX = X.T.dot(X)
        XtX_reg = XtX + self.alpha * I
        
        try:
            # Tentativa de inversão direta
            XtX_inv = np.linalg.inv(XtX_reg)
            self.weights = XtX_inv.dot(X.T).dot(y)
        except np.linalg.LinAlgError:
            # Se a matriz for singular, usa pseudo-inversa
            print("Aviso: A matriz X^T X é singular. Usando a pseudo-inversa.")
            XtX_inv = np.linalg.pinv(XtX_reg)
            self.weights = XtX_inv.dot(X.T).dot(y)
    
    def _gradient_descent(self, X, y):
        """
        Usa o gradiente descendente para encontrar os pesos ótimos.
        """
        n_samples, n_features = X.shape
        
        # Inicializar pesos
        self.weights = np.zeros(n_features)
        
        # Gradiente descendente
        for i in range(self.n_iterations):
            # Predições com os pesos atuais
            y_pred = X.dot(self.weights)
            
            # Erro
            error = y_pred - y
            
            # Gradiente = X^T * erro / n_amostras + alpha * w / n_amostras
            # Não regularizamos o intercepto (primeira coluna)
            gradient = X.T.dot(error) / n_samples
            gradient[1:] += (self.alpha * self.weights[1:]) / n_samples
            
            # Atualização dos pesos
            self.weights -= self.learning_rate * gradient
            
            # Cálculo do custo (erro quadrático médio + termo de regularização)
            cost = np.mean(error ** 2) / 2
            if self.alpha > 0:
                cost += (self.alpha * np.sum(self.weights[1:] ** 2)) / (2 * n_samples)
            
            self.cost_history.append(cost)
    
    def fit(self, X, y):
        """
        Treina o modelo de Regressão Linear.
        
        Parâmetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Atributos das amostras de treinamento.
        y : array-like, shape (n_samples,)
            Valores alvo das amostras de treinamento.
            
        Retorna:
        --------
        self : objeto
            Retorna o próprio objeto.
        """
        # Converter para arrays numpy
        X = np.array(X)
        y = np.array(y)
        
        # Adicionar intercepto se necessário
        if self.fit_intercept:
            X = self._add_intercept(X)
        
        # Escolher método para encontrar os pesos
        if self.method == 'normal_equation':
            self._normal_equation(X, y)
        elif self.method == 'gradient_descent':
            self._gradient_descent(X, y)
        else:
            raise ValueError("Método não suportado. Use 'normal_equation' ou 'gradient_descent'.")
        
        # Extrair intercepto se foi adicionado
        if self.fit_intercept:
            self.intercept = self.weights[0]
            self.weights = self.weights[1:]
        
        return self
    
    def predict(self, X):
        """
        Prediz os valores para as amostras em X.
        
        Parâmetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Amostras para predição.
            
        Retorna:
        --------
        y_pred : array, shape (n_samples,)
            Valores preditos.
        """
        X = np.array(X)
        
        if self.fit_intercept:
            return self.intercept + X.dot(self.weights)
        else:
            return X.dot(self.weights)
    
    def score(self, X, y):
        """
        Calcula o coeficiente de determinação R² do modelo.
        
        Parâmetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Amostras de teste.
        y : array-like, shape (n_samples,)
            Valores verdadeiros.
            
        Retorna:
        --------
        r2 : float
            Coeficiente de determinação.
        """
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        
        return 1 - (u / v)
    
    def plot_regression(self, X, y, feature_idx=0, ax=None):
        """
        Plota a linha de regressão para um atributo específico.
        
        Parâmetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Amostras.
        y : array-like, shape (n_samples,)
            Valores verdadeiros.
        feature_idx : int
            Índice do atributo a ser plotado.
        ax : matplotlib.axes.Axes
            Eixo para plotagem.
            
        Retorna:
        --------
        ax : matplotlib.axes.Axes
            Eixo com a plotagem.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        X = np.array(X)
        y = np.array(y)
        
        # Plotar pontos
        ax.scatter(X[:, feature_idx], y, c='b', marker='o', alpha=0.5)
        
        # Plotar linha de regressão
        x_range = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), 100)
        if X.shape[1] == 1:
            # Regressão simples
            y_pred = self.predict(x_range.reshape(-1, 1))
        else:
            # Regressão múltipla: usa a média de outros atributos
            X_mean = X.mean(axis=0)
            X_plot = np.tile(X_mean, (100, 1))
            X_plot[:, feature_idx] = x_range
            y_pred = self.predict(X_plot)
        
        ax.plot(x_range, y_pred, c='r', linestyle='-')
        
        # Adicionar rótulos
        ax.set_xlabel(f'Atributo {feature_idx}')
        ax.set_ylabel('Valor alvo')
        ax.set_title('Regressão Linear')
        
        return ax
    
    def plot_cost_history(self, ax=None):
        """
        Plota o histórico de custo durante o treinamento.
        
        Parâmetros:
        -----------
        ax : matplotlib.axes.Axes
            Eixo para plotagem.
            
        Retorna:
        --------
        ax : matplotlib.axes.Axes
            Eixo com a plotagem.
        """
        if self.method != 'gradient_descent' or not self.cost_history:
            raise ValueError("Histórico de custo disponível apenas para o método de gradiente descendente.")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(range(len(self.cost_history)), self.cost_history)
        ax.set_xlabel('Iteração')
        ax.set_ylabel('Custo')
        ax.set_title('Histórico de Custo')
        
        return ax
    
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """
        Calcula o erro quadrático médio.
        
        Parâmetros:
        -----------
        y_true : array-like
            Valores verdadeiros.
        y_pred : array-like
            Valores preditos.
            
        Retorna:
        --------
        mse : float
            Erro quadrático médio.
        """
        return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)
    
    @staticmethod
    def r2_score(y_true, y_pred):
        """
        Calcula o coeficiente de determinação R².
        
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
        
        u = ((y_true - y_pred) ** 2).sum()
        v = ((y_true - y_true.mean()) ** 2).sum()
        
        return 1 - (u / v)
    
    @staticmethod
    def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
        """
        Divide os dados em conjuntos de treino e teste.
        
        Parâmetros:
        -----------
        X : array-like
            Atributos.
        y : array-like
            Valores alvo.
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