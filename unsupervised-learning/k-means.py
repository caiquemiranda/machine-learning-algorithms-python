# K-Means Clustering

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class KMeans:
    """
    Implementação do algoritmo K-Means para agrupamento.
    """
    
    def __init__(self, n_clusters=3, max_iterations=100, random_state=None, 
                 init='random', tol=1e-4):
        """
        Inicializa o algoritmo K-Means.
        
        Parâmetros:
        -----------
        n_clusters : int
            Número de clusters para agrupar os dados.
        max_iterations : int
            Número máximo de iterações.
        random_state : int ou None
            Semente para reprodutibilidade.
        init : str
            Método de inicialização: 'random' ou 'k-means++'.
        tol : float
            Tolerância de convergência.
        """
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.init = init
        self.tol = tol
        self.centroids = None
        self.labels = None
        self.inertia_ = None
        
        if random_state:
            np.random.seed(random_state)
    
    def _initialize_centroids(self, X):
        """
        Inicializa os centroides usando o método especificado.
        
        Parâmetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Amostras de treinamento.
            
        Retorna:
        --------
        centroids : array, shape (n_clusters, n_features)
            Centroides iniciais.
        """
        n_samples, n_features = X.shape
        
        if self.init == 'random':
            # Escolhe n_clusters amostras aleatórias como centroides iniciais
            idx = np.random.choice(n_samples, self.n_clusters, replace=False)
            centroids = X[idx]
            
        elif self.init == 'k-means++':
            # Implementação de K-Means++
            centroids = np.zeros((self.n_clusters, n_features))
            
            # Escolhe o primeiro centroide aleatoriamente
            idx = np.random.randint(0, n_samples)
            centroids[0] = X[idx]
            
            # Escolhe os próximos centroides com base na distância
            for i in range(1, self.n_clusters):
                # Calcula a distância de cada ponto ao centroide mais próximo
                distances = np.array([
                    np.min([np.linalg.norm(x - c) ** 2 for c in centroids[:i]])
                    for x in X
                ])
                
                # Normaliza para criar uma distribuição de probabilidade
                distances /= np.sum(distances)
                
                # Escolhe o próximo centroide com probabilidade proporcional à distância
                idx = np.random.choice(n_samples, p=distances)
                centroids[i] = X[idx]
                
        else:
            raise ValueError("Método de inicialização não suportado. Use 'random' ou 'k-means++'.")
            
        return centroids
    
    def _assign_clusters(self, X, centroids):
        """
        Atribui cada amostra ao cluster mais próximo.
        
        Parâmetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Amostras de treinamento.
        centroids : array, shape (n_clusters, n_features)
            Posições dos centroides atuais.
            
        Retorna:
        --------
        labels : array, shape (n_samples,)
            Rótulos dos clusters para cada amostra.
        """
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        
        # Para cada amostra, encontrar o centroide mais próximo
        for i in range(n_samples):
            # Calcular distâncias a todos os centroides
            distances = np.linalg.norm(X[i] - centroids, axis=1)
            # Atribuir ao cluster com menor distância
            labels[i] = np.argmin(distances)
            
        return labels
    
    def _update_centroids(self, X, labels):
        """
        Atualiza a posição dos centroides com base nas amostras atribuídas.
        
        Parâmetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Amostras de treinamento.
        labels : array, shape (n_samples,)
            Rótulos dos clusters para cada amostra.
            
        Retorna:
        --------
        centroids : array, shape (n_clusters, n_features)
            Novas posições dos centroides.
        """
        n_features = X.shape[1]
        centroids = np.zeros((self.n_clusters, n_features))
        
        # Para cada cluster, calcular a média das amostras atribuídas
        for k in range(self.n_clusters):
            cluster_samples = X[labels == k]
            if len(cluster_samples) > 0:
                centroids[k] = np.mean(cluster_samples, axis=0)
            else:
                # Se um cluster ficar vazio, reposicionar aleatoriamente
                centroids[k] = X[np.random.randint(0, X.shape[0])]
                
        return centroids
    
    def _calculate_inertia(self, X, labels, centroids):
        """
        Calcula a inércia (soma das distâncias quadradas de cada amostra ao seu centroide).
        
        Parâmetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Amostras de treinamento.
        labels : array, shape (n_samples,)
            Rótulos dos clusters para cada amostra.
        centroids : array, shape (n_clusters, n_features)
            Posições dos centroides.
            
        Retorna:
        --------
        inertia : float
            Valor da inércia.
        """
        inertia = 0.0
        
        for i in range(X.shape[0]):
            cluster_idx = labels[i]
            inertia += np.linalg.norm(X[i] - centroids[cluster_idx]) ** 2
            
        return inertia
    
    def fit(self, X):
        """
        Executa o algoritmo K-Means nos dados.
        
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
        
        # Inicializar centroides
        self.centroids = self._initialize_centroids(X)
        
        prev_centroids = np.zeros_like(self.centroids)
        
        # Iterações do algoritmo
        for _ in range(self.max_iterations):
            # Atribuir clusters
            self.labels = self._assign_clusters(X, self.centroids)
            
            # Salvar centroides anteriores para verificar convergência
            prev_centroids = self.centroids.copy()
            
            # Atualizar centroides
            self.centroids = self._update_centroids(X, self.labels)
            
            # Verificar convergência
            centroid_shift = np.linalg.norm(self.centroids - prev_centroids)
            if centroid_shift < self.tol:
                break
        
        # Calcular inércia final
        self.inertia_ = self._calculate_inertia(X, self.labels, self.centroids)
        
        return self
    
    def predict(self, X):
        """
        Prediz o cluster mais próximo para cada amostra em X.
        
        Parâmetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Novas amostras.
            
        Retorna:
        --------
        labels : array, shape (n_samples,)
            Rótulos dos clusters para as novas amostras.
        """
        X = np.array(X)
        return self._assign_clusters(X, self.centroids)
    
    def fit_predict(self, X):
        """
        Executa o fit e prediz o cluster para cada amostra em X.
        
        Parâmetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Amostras de treinamento.
            
        Retorna:
        --------
        labels : array, shape (n_samples,)
            Rótulos dos clusters para as amostras.
        """
        self.fit(X)
        return self.labels
    
    def plot_clusters(self, X, ax=None):
        """
        Plota os clusters e centroides.
        
        Parâmetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Amostras.
        ax : matplotlib.axes.Axes
            Eixo para plotagem.
            
        Retorna:
        --------
        ax : matplotlib.axes.Axes
            Eixo com a plotagem.
        """
        if X.shape[1] != 2:
            raise ValueError("A plotagem só é possível para dados 2D.")
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Criar mapa de cores
        cmap = ListedColormap(['#FF5733', '#33FF57', '#3357FF', '#F3FF33', '#FF33F3',
                              '#33FFF3', '#F333FF', '#FF3383', '#33FF83', '#8333FF'])
        
        # Limitar ao número de clusters
        cmap = ListedColormap([cmap(i) for i in range(min(self.n_clusters, 10))])
        
        # Plotar pontos coloridos por cluster
        scatter = ax.scatter(X[:, 0], X[:, 1], c=self.labels, cmap=cmap, 
                            marker='o', s=50, alpha=0.6)
        
        # Plotar centroides
        ax.scatter(self.centroids[:, 0], self.centroids[:, 1],
                   c=range(self.n_clusters), cmap=cmap,
                   marker='x', s=200, linewidths=3)
        
        # Adicionar rótulos
        ax.set_xlabel('Atributo 1')
        ax.set_ylabel('Atributo 2')
        ax.set_title('Clusters K-Means')
        
        # Adicionar legenda
        legend1 = ax.legend(*scatter.legend_elements(),
                            loc="upper right", title="Clusters")
        ax.add_artist(legend1)
        
        return ax
    
    @staticmethod
    def silhouette_score(X, labels):
        """
        Calcula o coeficiente silhouette para avaliação do agrupamento.
        
        Parâmetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Amostras.
        labels : array, shape (n_samples,)
            Rótulos dos clusters para as amostras.
            
        Retorna:
        --------
        silhouette : float
            Coeficiente silhouette (-1 a 1).
        """
        n_samples = X.shape[0]
        n_clusters = np.max(labels) + 1
        
        # Para cada amostra
        silhouettes = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Cluster da amostra atual
            cluster_i = labels[i]
            
            # Calcular a média da distância da amostra i para outras amostras no mesmo cluster (coesão)
            # Se o cluster tem apenas uma amostra, a coesão é 0
            same_cluster_indices = np.where(labels == cluster_i)[0]
            if len(same_cluster_indices) > 1:
                a_i = np.mean([np.linalg.norm(X[i] - X[j]) for j in same_cluster_indices if j != i])
            else:
                a_i = 0
            
            # Calcular a média da distância da amostra i para amostras em outros clusters (separação)
            b_i = float('inf')
            for k in range(n_clusters):
                if k != cluster_i:
                    other_cluster_indices = np.where(labels == k)[0]
                    if len(other_cluster_indices) > 0:
                        b_ik = np.mean([np.linalg.norm(X[i] - X[j]) for j in other_cluster_indices])
                        b_i = min(b_i, b_ik)
            
            # Calcular o coeficiente silhouette para a amostra i
            if a_i == 0 and b_i == float('inf'):
                silhouettes[i] = 0
            elif b_i == float('inf'):
                silhouettes[i] = 0
            else:
                silhouettes[i] = (b_i - a_i) / max(a_i, b_i)
        
        # Retornar a média dos coeficientes silhouette
        return np.mean(silhouettes)
    
    @staticmethod
    def elbow_method(X, k_range=range(1, 11), random_state=None):
        """
        Executa o método do cotovelo para encontrar o número ideal de clusters.
        
        Parâmetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Amostras.
        k_range : range
            Intervalo do número de clusters a testar.
        random_state : int ou None
            Semente para reprodutibilidade.
            
        Retorna:
        --------
        fig : matplotlib.figure.Figure
            Figura com o gráfico do método do cotovelo.
        """
        inertias = []
        
        for k in k_range:
            model = KMeans(n_clusters=k, random_state=random_state)
            model.fit(X)
            inertias.append(model.inertia_)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_range, inertias, 'o-', color='#3333FF')
        ax.set_xlabel('Número de clusters (k)')
        ax.set_ylabel('Inércia')
        ax.set_title('Método do Cotovelo para K-Means')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig 