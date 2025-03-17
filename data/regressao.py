"""
Gerador de conjuntos de dados sintéticos para regressão.
Este módulo contém funções para gerar diferentes tipos de conjuntos de dados
para testar algoritmos de regressão.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

def gerar_regressao_linear(n_amostras=100, n_atributos=1, 
                          random_state=42, noise=10.0, bias=0.0):
    """
    Gera um conjunto de dados para regressão linear.
    
    Parâmetros:
    -----------
    n_amostras : int
        Número de amostras a gerar.
    n_atributos : int
        Número de atributos (características).
    random_state : int
        Semente para reprodutibilidade.
    noise : float
        Quantidade de ruído a adicionar.
    bias : float
        Termo de viés a adicionar.
        
    Retorna:
    --------
    X : array, shape (n_amostras, n_atributos)
        Dados gerados.
    y : array, shape (n_amostras,)
        Valores alvo.
    """
    X, y = make_regression(
        n_samples=n_amostras,
        n_features=n_atributos,
        n_informative=n_atributos,
        noise=noise,
        bias=bias,
        random_state=random_state
    )
    
    return X, y

def gerar_regressao_polinomial(n_amostras=100, grau=2, random_state=42, noise=10.0):
    """
    Gera um conjunto de dados para regressão polinomial.
    
    Parâmetros:
    -----------
    n_amostras : int
        Número de amostras a gerar.
    grau : int
        Grau do polinômio.
    random_state : int
        Semente para reprodutibilidade.
    noise : float
        Quantidade de ruído a adicionar.
        
    Retorna:
    --------
    X : array, shape (n_amostras, 1)
        Dados gerados.
    y : array, shape (n_amostras,)
        Valores alvo.
    """
    np.random.seed(random_state)
    
    X = np.sort(np.random.rand(n_amostras, 1) * 10 - 5, axis=0)  # Valores entre -5 e 5
    
    # Gerar coeficientes aleatórios para o polinômio
    coefs = np.random.randn(grau + 1)
    
    # Calcular y usando o polinômio
    y = np.zeros(n_amostras)
    for i in range(grau + 1):
        y += coefs[i] * np.power(X[:, 0], i)
    
    # Adicionar ruído
    y += np.random.randn(n_amostras) * noise
    
    return X, y

def gerar_regressao_senoidal(n_amostras=100, random_state=42, noise=0.5):
    """
    Gera um conjunto de dados para regressão com padrão senoidal.
    
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
    X : array, shape (n_amostras, 1)
        Dados gerados.
    y : array, shape (n_amostras,)
        Valores alvo.
    """
    np.random.seed(random_state)
    
    X = np.sort(np.random.rand(n_amostras, 1) * 10, axis=0)  # Valores entre 0 e 10
    y = np.sin(X[:, 0]) + np.random.randn(n_amostras) * noise
    
    return X, y

def gerar_regressao_multipla(n_amostras=100, n_atributos=3, 
                            random_state=42, noise=10.0):
    """
    Gera um conjunto de dados para regressão múltipla.
    
    Parâmetros:
    -----------
    n_amostras : int
        Número de amostras a gerar.
    n_atributos : int
        Número de atributos (características).
    random_state : int
        Semente para reprodutibilidade.
    noise : float
        Quantidade de ruído a adicionar.
        
    Retorna:
    --------
    X : array, shape (n_amostras, n_atributos)
        Dados gerados.
    y : array, shape (n_amostras,)
        Valores alvo.
    """
    return gerar_regressao_linear(
        n_amostras=n_amostras, 
        n_atributos=n_atributos, 
        random_state=random_state, 
        noise=noise
    )

def visualizar_regressao(X, y, tipo='simples', titulo=None, ax=None):
    """
    Visualiza um conjunto de dados de regressão.
    
    Parâmetros:
    -----------
    X : array
        Dados a serem visualizados.
    y : array
        Valores alvo.
    tipo : str
        'simples' para regressão simples (1D), 'multipla' para regressão múltipla.
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
    
    if tipo == 'simples' and X.shape[1] == 1:
        ax.scatter(X[:, 0], y, alpha=0.7, color='blue')
        ax.set_xlabel('X')
        ax.set_ylabel('y')
    elif tipo == 'multipla':
        # Para regressão múltipla, visualizar os primeiros dois atributos
        sc = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
        plt.colorbar(sc, ax=ax, label='y')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
    
    if titulo:
        ax.set_title(titulo)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    return ax

# Exemplo de uso
if __name__ == "__main__":
    plt.figure(figsize=(20, 15))
    
    # Regressão linear simples
    X, y = gerar_regressao_linear(n_amostras=100, n_atributos=1, random_state=42)
    plt.subplot(2, 2, 1)
    visualizar_regressao(X, y, 'simples', "Regressão Linear Simples")
    
    # Regressão polinomial
    X, y = gerar_regressao_polinomial(n_amostras=100, grau=3, random_state=42)
    plt.subplot(2, 2, 2)
    visualizar_regressao(X, y, 'simples', "Regressão Polinomial (Grau 3)")
    
    # Regressão senoidal
    X, y = gerar_regressao_senoidal(n_amostras=100, random_state=42)
    plt.subplot(2, 2, 3)
    visualizar_regressao(X, y, 'simples', "Regressão Senoidal")
    
    # Regressão múltipla
    X, y = gerar_regressao_multipla(n_amostras=100, n_atributos=2, random_state=42)
    plt.subplot(2, 2, 4)
    visualizar_regressao(X, y, 'multipla', "Regressão Múltipla")
    
    plt.tight_layout()
    plt.savefig("datasets_regressao.png", dpi=100)
    plt.show() 