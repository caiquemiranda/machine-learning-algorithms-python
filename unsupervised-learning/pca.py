# Principal Component Analysis (PCA)

import numpy as np
import matplotlib.pyplot as plt

class PCA:
    """
    Implementação do algoritmo de Análise de Componentes Principais (PCA)
    para redução de dimensionalidade.
    """
    
    def __init__(self, n_components=None, whiten=False):
        """
        Inicializa o algoritmo PCA.
        
        Parâmetros:
        -----------
        n_components : int ou None
            Número de componentes principais a manter. Se None, mantém todas.
        whiten : bool
            Se True, os componentes terão variância unitária.
        """
        self.n_components = n_components
        self.whiten = whiten
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.n_samples_ = None
        self.n_features_ = None
        
    def fit(self, X):
        """
        Ajusta o PCA aos dados.
        
        Parâmetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Amostras de treinamento.
            
        Retorna:
        --------
        self : objeto
            Retorna o próprio objeto.
        """
        X = np.array(X)
        self.n_samples_, self.n_features_ = X.shape
        
        # Centralizar os dados: subtrair a média
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Calcular a matriz de covariância
        # Usar método SVD em vez da matriz de covariância diretamente
        # para maior estabilidade numérica
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Autovalores são os quadrados dos valores singulares divididos por (n-1)
        self.explained_variance_ = (S ** 2) / (self.n_samples_ - 1)
        
        # Proporção da variância explicada
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(self.explained_variance_)
        
        # Autovetores (componentes principais)
        self.components_ = Vt
        
        # Valores singulares
        self.singular_values_ = S
        
        return self
    
    def transform(self, X):
        """
        Aplica a redução de dimensionalidade aos dados.
        
        Parâmetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Amostras a transformar.
            
        Retorna:
        --------
        X_new : array, shape (n_samples, n_components)
            Amostras transformadas.
        """
        X = np.array(X)
        
        # Verificar se o PCA foi ajustado
        if self.components_ is None:
            raise ValueError("O PCA precisa ser ajustado antes de transformar dados. Use fit() primeiro.")
        
        # Centralizar os dados: subtrair a média
        X_centered = X - self.mean_
        
        # Determinar o número de componentes
        n_components = self.n_components
        if n_components is None:
            n_components = self.n_features_
        
        # Projetar os dados nos autovetores (componentes principais)
        X_transformed = np.dot(X_centered, self.components_[:n_components].T)
        
        # Branqueamento (whitening): dividir por sqrt(λ) para obter variância unitária
        if self.whiten:
            X_transformed /= np.sqrt(self.explained_variance_[:n_components] + 1e-10)
        
        return X_transformed
    
    def fit_transform(self, X):
        """
        Ajusta o PCA aos dados e aplica a redução de dimensionalidade.
        
        Parâmetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Amostras de treinamento.
            
        Retorna:
        --------
        X_new : array, shape (n_samples, n_components)
            Amostras transformadas.
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        """
        Transforma os dados de volta para o espaço original.
        Projeta os dados transformados de volta no espaço original.
        
        Parâmetros:
        -----------
        X : array-like, shape (n_samples, n_components)
            Amostras transformadas.
            
        Retorna:
        --------
        X_original : array, shape (n_samples, n_features)
            Amostras reconstruídas.
        """
        X = np.array(X)
        
        # Verificar se o PCA foi ajustado
        if self.components_ is None:
            raise ValueError("O PCA precisa ser ajustado antes de transformar dados. Use fit() primeiro.")
        
        # Determinar o número de componentes
        n_components = self.n_components
        if n_components is None:
            n_components = self.n_features_
        
        # Reverter o branqueamento se aplicado
        if self.whiten:
            X = X * np.sqrt(self.explained_variance_[:n_components] + 1e-10)
        
        # Projetar de volta para o espaço original
        X_original = np.dot(X, self.components_[:n_components])
        
        # Adicionar a média de volta
        X_original = X_original + self.mean_
        
        return X_original
    
    def get_covariance(self):
        """
        Calcula a matriz de covariância a partir dos componentes.
        
        Retorna:
        --------
        cov : array, shape (n_features, n_features)
            Matriz de covariância.
        """
        # Verificar se o PCA foi ajustado
        if self.components_ is None:
            raise ValueError("O PCA precisa ser ajustado antes de calcular a covariância. Use fit() primeiro.")
        
        # Calcular matriz de covariância
        # cov = V * Λ * V^T
        return np.dot(
            self.components_.T * self.explained_variance_,
            self.components_
        )
    
    def plot_explained_variance(self, ax=None):
        """
        Plota a variância explicada acumulada por componente.
        
        Parâmetros:
        -----------
        ax : matplotlib.axes.Axes
            Eixo para plotagem.
            
        Retorna:
        --------
        ax : matplotlib.axes.Axes
            Eixo com a plotagem.
        """
        # Verificar se o PCA foi ajustado
        if self.components_ is None:
            raise ValueError("O PCA precisa ser ajustado antes de plotar. Use fit() primeiro.")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calcular variância explicada acumulada
        cumulative_variance_ratio = np.cumsum(self.explained_variance_ratio_)
        
        # Plotar variância explicada por componente
        ax.bar(range(1, len(self.explained_variance_ratio_) + 1), 
               self.explained_variance_ratio_, alpha=0.5, color='b',
               label='Variância explicada individual')
        
        # Plotar variância explicada acumulada
        ax.step(range(1, len(cumulative_variance_ratio) + 1), 
                cumulative_variance_ratio, where='mid', color='r',
                label='Variância explicada acumulada')
        
        # Linha de referência em 95% da variância explicada
        ax.axhline(y=0.95, color='k', linestyle='--', alpha=0.5,
                  label='95% da variância explicada')
        
        # Adicionar rótulos
        ax.set_xlabel('Número de componentes')
        ax.set_ylabel('Proporção da variância explicada')
        ax.set_title('Variância explicada por componente principal')
        ax.set_xticks(range(1, len(self.explained_variance_ratio_) + 1))
        ax.set_ylim([0, 1.05])
        ax.legend(loc='best')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return ax
    
    def plot_components(self, X, y=None, ax=None, components=[0, 1]):
        """
        Plota os dados transformados nos dois primeiros componentes principais.
        
        Parâmetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Amostras.
        y : array-like, shape (n_samples,)
            Rótulos das amostras para coloração. Se None, usa uma única cor.
        ax : matplotlib.axes.Axes
            Eixo para plotagem.
        components : list
            Índices dos componentes a plotar [x, y].
            
        Retorna:
        --------
        ax : matplotlib.axes.Axes
            Eixo com a plotagem.
        """
        # Transformar os dados
        X_transformed = self.transform(X)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Verificar se os componentes solicitados existem
        if max(components) >= X_transformed.shape[1]:
            raise ValueError(f"Componentes solicitados {components} não disponíveis. Máximo: {X_transformed.shape[1]-1}")
        
        # Plotar os dados
        comp1, comp2 = components
        if y is not None:
            scatter = ax.scatter(X_transformed[:, comp1], X_transformed[:, comp2], 
                                c=y, alpha=0.6, edgecolors='k', s=50)
            
            # Adicionar legenda se y tiver valores discretos
            if len(np.unique(y)) < 10:
                legend1 = ax.legend(*scatter.legend_elements(),
                                   loc="upper right", title="Classes")
                ax.add_artist(legend1)
        else:
            ax.scatter(X_transformed[:, comp1], X_transformed[:, comp2], 
                      alpha=0.6, edgecolors='k', s=50)
        
        # Adicionar rótulos
        ax.set_xlabel(f'Componente Principal {comp1+1} ({self.explained_variance_ratio_[comp1]:.2%})')
        ax.set_ylabel(f'Componente Principal {comp2+1} ({self.explained_variance_ratio_[comp2]:.2%})')
        ax.set_title('Projeção nos Componentes Principais')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return ax
    
    @staticmethod
    def get_optimal_components(explained_variance_ratio, threshold=0.95):
        """
        Encontra o número ideal de componentes baseado na variância explicada.
        
        Parâmetros:
        -----------
        explained_variance_ratio : array
            Proporção da variância explicada por componente.
        threshold : float
            Threshold de variância explicada (0-1).
            
        Retorna:
        --------
        n_components : int
            Número de componentes que explicam pelo menos threshold da variância.
        """
        cumulative_variance = np.cumsum(explained_variance_ratio)
        n_components = np.argmax(cumulative_variance >= threshold) + 1
        return n_components 