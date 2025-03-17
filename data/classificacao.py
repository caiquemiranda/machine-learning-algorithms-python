"""
Gerador de conjuntos de dados sintéticos para classificação.
Este módulo contém funções para gerar diferentes tipos de conjuntos de dados
para testar algoritmos de classificação.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, make_circles

def gerar_classificacao_linear(n_amostras=100, n_atributos=2, n_classes=2, 
                               random_state=42, noise=0.1):
    """
    Gera um conjunto de dados linearmente separável.
    
    Parâmetros:
    -----------
    n_amostras : int
        Número de amostras a gerar.
    n_atributos : int
        Número de atributos (características).
    n_classes : int
        Número de classes (2 ou 3).
    random_state : int
        Semente para reprodutibilidade.
    noise : float
        Quantidade de ruído a adicionar.
        
    Retorna:
    --------
    X : array, shape (n_amostras, n_atributos)
        Dados gerados.
    y : array, shape (n_amostras,)
        Rótulos das classes (0, 1, ...).
    """
    X, y = make_classification(
        n_samples=n_amostras,
        n_features=n_atributos,
        n_classes=n_classes,
        n_redundant=0,
        n_informative=n_atributos,
        random_state=random_state,
        n_clusters_per_class=1,
        class_sep=1.0,
        flip_y=noise
    )
    
    return X, y

def gerar_classificacao_nao_linear(tipo='moons', n_amostras=100, random_state=42, noise=0.1):
    """
    Gera conjuntos de dados não linearmente separáveis: 'moons' ou 'circles'.
    
    Parâmetros:
    -----------
    tipo : str
        'moons' para o conjunto de dados em forma de lua, 
        'circles' para o conjunto de dados em forma de círculos concêntricos.
    n_amostras : int
        Número de amostras a gerar.
    random_state : int
        Semente para reprodutibilidade.
    noise : float
        Quantidade de ruído a adicionar.
        
    Retorna:
    --------
    X : array, shape (n_amostras, 2)
        Dados gerados.
    y : array, shape (n_amostras,)
        Rótulos das classes (0, 1).
    """
    if tipo == 'moons':
        X, y = make_moons(n_samples=n_amostras, noise=noise, random_state=random_state)
    elif tipo == 'circles':
        X, y = make_circles(n_samples=n_amostras, noise=noise, random_state=random_state, factor=0.5)
    else:
        raise ValueError("Tipo deve ser 'moons' ou 'circles'.")
    
    return X, y

def gerar_xor(n_amostras=100, random_state=42, noise=0.1):
    """
    Gera um conjunto de dados que simula o problema XOR.
    
    Parâmetros:
    -----------
    n_amostras : int
        Número de amostras a gerar.
    random_state : int
        Semente para reprodutibilidade.
    noise : float
        Quantidade de ruído a adicionar.
        
    Retorna:
    --------
    X : array, shape (n_amostras, 2)
        Dados gerados.
    y : array, shape (n_amostras,)
        Rótulos das classes (0, 1).
    """
    np.random.seed(random_state)
    
    X = np.random.rand(n_amostras, 2) * 2 - 1  # Valores entre -1 e 1
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
    
    # Adicionar ruído
    if noise > 0:
        n_flip = int(noise * n_amostras)
        flip_idx = np.random.choice(n_amostras, n_flip, replace=False)
        y[flip_idx] = 1 - y[flip_idx]
    
    return X, y

def gerar_multiclasse(n_amostras=300, random_state=42):
    """
    Gera um conjunto de dados com múltiplas classes.
    
    Parâmetros:
    -----------
    n_amostras : int
        Número de amostras a gerar.
    random_state : int
        Semente para reprodutibilidade.
        
    Retorna:
    --------
    X : array, shape (n_amostras, 2)
        Dados gerados.
    y : array, shape (n_amostras,)
        Rótulos das classes (0, 1, 2).
    """
    np.random.seed(random_state)
    
    n_class = n_amostras // 3
    
    # Gerar três clusters
    X0 = np.random.randn(n_class, 2) * 0.5 + np.array([0, 0])
    X1 = np.random.randn(n_class, 2) * 0.5 + np.array([2, 2])
    X2 = np.random.randn(n_class, 2) * 0.5 + np.array([0, 3])
    
    X = np.vstack([X0, X1, X2])
    y = np.hstack([np.zeros(n_class), np.ones(n_class), np.ones(n_class) * 2])
    
    # Embaralhar os dados
    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]
    
    return X, y

def visualizar(X, y, titulo=None, ax=None):
    """
    Visualiza um conjunto de dados de classificação 2D.
    
    Parâmetros:
    -----------
    X : array, shape (n_amostras, 2)
        Dados a serem visualizados.
    y : array, shape (n_amostras,)
        Rótulos das classes.
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
    
    # Criar mapa de cores para até 10 classes
    cores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Plotar cada classe
    for i in np.unique(y):
        ax.scatter(X[y == i, 0], X[y == i, 1], 
                  color=cores[int(i) % len(cores)], 
                  label=f'Classe {i}', 
                  alpha=0.7)
    
    if titulo:
        ax.set_title(titulo)
    ax.set_xlabel('Atributo 1')
    ax.set_ylabel('Atributo 2')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    return ax

# Exemplo de uso
if __name__ == "__main__":
    plt.figure(figsize=(20, 15))
    
    # Dados linearmente separáveis
    X, y = gerar_classificacao_linear(n_amostras=100, random_state=42)
    plt.subplot(2, 2, 1)
    visualizar(X, y, "Dados Linearmente Separáveis")
    
    # Dados em forma de lua
    X, y = gerar_classificacao_nao_linear('moons', n_amostras=100, random_state=42)
    plt.subplot(2, 2, 2)
    visualizar(X, y, "Dados em Forma de Lua")
    
    # Dados XOR
    X, y = gerar_xor(n_amostras=100, random_state=42)
    plt.subplot(2, 2, 3)
    visualizar(X, y, "Dados XOR")
    
    # Dados multiclasse
    X, y = gerar_multiclasse(n_amostras=300, random_state=42)
    plt.subplot(2, 2, 4)
    visualizar(X, y, "Dados Multiclasse")
    
    plt.tight_layout()
    plt.savefig("datasets_classificacao.png", dpi=100)
    plt.show() 