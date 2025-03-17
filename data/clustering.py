"""
Gerador de conjuntos de dados sintéticos para clustering e redução de dimensionalidade.
Este módulo contém funções para gerar diferentes tipos de conjuntos de dados
para testar algoritmos de agrupamento e redução de dimensionalidade.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles, make_moons

def gerar_clusters(n_amostras=300, n_clusters=3, n_atributos=2, 
                   sep=1.0, random_state=42):
    """
    Gera clusters para agrupamento.
    
    Parâmetros:
    -----------
    n_amostras : int
        Número de amostras a gerar.
    n_clusters : int
        Número de clusters.
    n_atributos : int
        Número de atributos (características).
    sep : float
        Separação entre clusters (maior = mais separados).
    random_state : int
        Semente para reprodutibilidade.
        
    Retorna:
    --------
    X : array, shape (n_amostras, n_atributos)
        Dados gerados.
    y : array, shape (n_amostras,)
        Rótulos dos clusters para avaliação.
    """
    X, y = make_blobs(
        n_samples=n_amostras,
        n_features=n_atributos,
        centers=n_clusters,
        cluster_std=sep,
        random_state=random_state
    )
    
    return X, y

def gerar_clusters_nao_lineares(tipo='moons', n_amostras=300, noise=0.05, 
                              random_state=42):
    """
    Gera clusters não lineares para agrupamento.
    
    Parâmetros:
    -----------
    tipo : str
        'moons' para clusters em forma de meia-lua, 
        'circles' para clusters em forma de círculos concêntricos.
    n_amostras : int
        Número de amostras a gerar.
    noise : float
        Quantidade de ruído a adicionar.
    random_state : int
        Semente para reprodutibilidade.
        
    Retorna:
    --------
    X : array, shape (n_amostras, 2)
        Dados gerados.
    y : array, shape (n_amostras,)
        Rótulos dos clusters para avaliação.
    """
    if tipo == 'moons':
        X, y = make_moons(n_samples=n_amostras, noise=noise, random_state=random_state)
    elif tipo == 'circles':
        X, y = make_circles(n_samples=n_amostras, noise=noise, factor=0.5, random_state=random_state)
    else:
        raise ValueError("Tipo deve ser 'moons' ou 'circles'.")
    
    return X, y

def gerar_dados_alta_dimensao(n_amostras=300, n_atributos=10, n_informative=3,
                            n_clusters=3, random_state=42):
    """
    Gera dados de alta dimensão com apenas alguns atributos informativos,
    útil para testar algoritmos de redução de dimensionalidade.
    
    Parâmetros:
    -----------
    n_amostras : int
        Número de amostras a gerar.
    n_atributos : int
        Número total de atributos.
    n_informative : int
        Número de atributos que realmente contêm informação.
    n_clusters : int
        Número de clusters.
    random_state : int
        Semente para reprodutibilidade.
        
    Retorna:
    --------
    X : array, shape (n_amostras, n_atributos)
        Dados gerados.
    y : array, shape (n_amostras,)
        Rótulos dos clusters para avaliação.
    """
    # Gerar dados apenas para os atributos informativos
    X_informative, y = make_blobs(
        n_samples=n_amostras,
        n_features=n_informative,
        centers=n_clusters,
        random_state=random_state
    )
    
    # Adicionar atributos ruidosos
    np.random.seed(random_state)
    X_noise = np.random.randn(n_amostras, n_atributos - n_informative)
    
    # Combinar atributos informativos e ruidosos
    X = np.hstack((X_informative, X_noise))
    
    return X, y

def visualizar_clusters(X, y=None, titulo=None, ax=None):
    """
    Visualiza clusters em um gráfico 2D.
    
    Parâmetros:
    -----------
    X : array
        Dados a serem visualizados.
    y : array, opcional
        Rótulos dos clusters, se disponíveis.
    titulo : str
        Título do gráfico.
    ax : matplotlib.axes.Axes
        Eixo para plotagem. Se None, cria um novo.
        
    Retorna:
    --------
    ax : matplotlib.axes.Axes
        Eixo com a plotagem.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    
    # Garantir que estamos visualizando apenas 2D
    if X.shape[1] > 2:
        print(f"Aviso: Visualizando apenas as duas primeiras dimensões de {X.shape[1]} dimensões.")
    
    # Criar mapa de cores para até 10 clusters
    cores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    if y is not None:
        # Se temos rótulos, usar cores diferentes para cada cluster
        for i in np.unique(y):
            ax.scatter(X[y == i, 0], X[y == i, 1], 
                      color=cores[int(i) % len(cores)], 
                      label=f'Cluster {i}', 
                      alpha=0.7)
        ax.legend()
    else:
        # Sem rótulos, plotar todos os pontos com a mesma cor
        ax.scatter(X[:, 0], X[:, 1], alpha=0.7, color='blue')
    
    if titulo:
        ax.set_title(titulo)
    ax.set_xlabel('Atributo 1')
    ax.set_ylabel('Atributo 2')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    return ax

# Exemplo de uso
if __name__ == "__main__":
    plt.figure(figsize=(20, 15))
    
    # Clusters padrão
    X, y = gerar_clusters(n_amostras=300, n_clusters=4, random_state=42)
    plt.subplot(2, 2, 1)
    visualizar_clusters(X, y, "Clusters Padrão")
    
    # Clusters não lineares - moons
    X, y = gerar_clusters_nao_lineares('moons', n_amostras=300, random_state=42)
    plt.subplot(2, 2, 2)
    visualizar_clusters(X, y, "Clusters em Forma de Lua")
    
    # Clusters não lineares - circles
    X, y = gerar_clusters_nao_lineares('circles', n_amostras=300, random_state=42)
    plt.subplot(2, 2, 3)
    visualizar_clusters(X, y, "Clusters em Forma de Círculo")
    
    # Dados de alta dimensão reduzidos para visualização
    X, y = gerar_dados_alta_dimensao(n_amostras=300, n_atributos=50, n_informative=2, random_state=42)
    plt.subplot(2, 2, 4)
    visualizar_clusters(X[:, :2], y, "Dados de Alta Dimensão (Primeiras 2 Dimensões)")
    
    plt.tight_layout()
    plt.savefig("datasets_clustering.png", dpi=100)
    plt.show() 