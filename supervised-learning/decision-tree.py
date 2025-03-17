# Árvore de Decisão

import numpy as np
from collections import Counter

class Node:
    """
    Nó para a Árvore de Decisão.
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Índice do atributo para divisão
        self.threshold = threshold  # Valor do atributo para divisão
        self.left = left            # Nó filho esquerdo (valores <= threshold)
        self.right = right          # Nó filho direito (valores > threshold)
        self.value = value          # Valor para nós folha (classe ou valor médio)
        
    def is_leaf(self):
        """Verifica se o nó é uma folha."""
        return self.value is not None

class DecisionTree:
    """
    Implementação de Árvore de Decisão para classificação e regressão.
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_impurity_decrease=0.0, 
                 criterion='gini', task='classification', random_state=None):
        """
        Inicializa a Árvore de Decisão.
        
        Parâmetros:
        -----------
        max_depth : int ou None
            Profundidade máxima da árvore. Se None, a árvore cresce até as folhas serem puras.
        min_samples_split : int
            Número mínimo de amostras necessárias para dividir um nó.
        min_impurity_decrease : float
            Redução mínima de impureza necessária para dividir um nó.
        criterion : str
            Função para medir a qualidade da divisão:
            'gini' ou 'entropy' para classificação, 'mse' para regressão.
        task : str
            Tipo de tarefa: 'classification' ou 'regression'.
        random_state : int ou None
            Semente para reprodutibilidade.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.criterion = criterion
        self.task = task
        self.random_state = random_state
        self.root = None
        
        if random_state:
            np.random.seed(random_state)
    
    def _entropy(self, y):
        """
        Calcula a entropia de um conjunto de rótulos.
        
        Parâmetros:
        -----------
        y : array-like
            Lista de rótulos.
            
        Retorna:
        --------
        entropy : float
            Valor da entropia.
        """
        _, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
        return entropy
    
    def _gini(self, y):
        """
        Calcula o índice de impureza Gini de um conjunto de rótulos.
        
        Parâmetros:
        -----------
        y : array-like
            Lista de rótulos.
            
        Retorna:
        --------
        gini : float
            Valor do índice Gini.
        """
        _, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        gini = 1 - np.sum(proportions ** 2)
        return gini
    
    def _mse(self, y):
        """
        Calcula o erro quadrático médio de um conjunto de valores.
        
        Parâmetros:
        -----------
        y : array-like
            Lista de valores.
            
        Retorna:
        --------
        mse : float
            Valor do erro quadrático médio.
        """
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)
    
    def _calculate_impurity(self, y):
        """
        Calcula a impureza de um conjunto de valores/rótulos com base no critério escolhido.
        """
        if self.task == 'classification':
            if self.criterion == 'gini':
                return self._gini(y)
            else:  # criterion == 'entropy'
                return self._entropy(y)
        else:  # task == 'regression'
            return self._mse(y)
    
    def _information_gain(self, y, y_left, y_right):
        """
        Calcula o ganho de informação obtido ao dividir um conjunto.
        
        Parâmetros:
        -----------
        y : array-like
            Conjunto original.
        y_left : array-like
            Subconjunto esquerdo.
        y_right : array-like
            Subconjunto direito.
            
        Retorna:
        --------
        info_gain : float
            Ganho de informação.
        """
        parent_impurity = self._calculate_impurity(y)
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        # Impureza ponderada dos nós filhos
        child_impurity = (n_left / n) * self._calculate_impurity(y_left) + (n_right / n) * self._calculate_impurity(y_right)
        
        # Ganho de informação
        return parent_impurity - child_impurity
    
    def _best_split(self, X, y):
        """
        Encontra a melhor divisão para o conjunto de dados.
        
        Parâmetros:
        -----------
        X : array-like
            Atributos.
        y : array-like
            Rótulos ou valores.
            
        Retorna:
        --------
        best_feature : int
            Índice do melhor atributo para divisão.
        best_threshold : float
            Melhor valor de threshold para divisão.
        best_gain : float
            Ganho de informação obtido com a melhor divisão.
        """
        n_samples, n_features = X.shape
        
        # Se não há amostras suficientes para divisão
        if n_samples < self.min_samples_split:
            return None, None, 0
        
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        # Para cada atributo
        for feature_idx in range(n_features):
            # Valores únicos do atributo
            thresholds = np.unique(X[:, feature_idx])
            
            # Para cada valor possível de threshold
            for threshold in thresholds:
                # Índices das amostras que vão para esquerda e direita
                left_idx = X[:, feature_idx] <= threshold
                right_idx = ~left_idx
                
                # Se algum dos lados não tem amostras, pular
                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                    continue
                
                # Calcular ganho de informação
                gain = self._information_gain(y, y[left_idx], y[right_idx])
                
                # Atualizar melhor divisão se o ganho for maior
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        # Verificar se o ganho é suficiente para justificar a divisão
        if best_gain < self.min_impurity_decrease:
            return None, None, 0
            
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """
        Constrói a árvore de decisão recursivamente.
        
        Parâmetros:
        -----------
        X : array-like
            Atributos.
        y : array-like
            Rótulos ou valores.
        depth : int
            Profundidade atual na árvore.
            
        Retorna:
        --------
        node : Node
            Nó raiz da árvore construída.
        """
        n_samples, n_features = X.shape
        
        # Verificar condições de parada
        if (self.max_depth is not None and depth >= self.max_depth) or n_samples < self.min_samples_split:
            if self.task == 'classification':
                # Para classificação, o valor é a classe mais frequente
                value = Counter(y).most_common(1)[0][0]
            else:
                # Para regressão, o valor é a média
                value = np.mean(y)
            return Node(value=value)
        
        # Encontrar a melhor divisão
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        
        # Se não encontrou uma divisão válida
        if best_feature is None:
            if self.task == 'classification':
                value = Counter(y).most_common(1)[0][0]
            else:
                value = np.mean(y)
            return Node(value=value)
        
        # Dividir o conjunto
        left_idx = X[:, best_feature] <= best_threshold
        right_idx = ~left_idx
        
        # Construir subárvores
        left_subtree = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_subtree = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        
        return Node(best_feature, best_threshold, left_subtree, right_subtree)
    
    def fit(self, X, y):
        """
        Treina a Árvore de Decisão.
        
        Parâmetros:
        -----------
        X : array-like
            Atributos das amostras de treinamento.
        y : array-like
            Rótulos/valores das amostras de treinamento.
            
        Retorna:
        --------
        self : objeto
            Retorna o próprio objeto.
        """
        X = np.array(X)
        y = np.array(y)
        
        self.root = self._build_tree(X, y)
        
        return self
    
    def _predict_sample(self, x, node):
        """
        Prediz o rótulo/valor para uma única amostra.
        
        Parâmetros:
        -----------
        x : array-like
            Vetor de atributos da amostra.
        node : Node
            Nó atual da árvore.
            
        Retorna:
        --------
        prediction : object
            Rótulo/valor predito.
        """
        # Se o nó é uma folha, retornar o valor
        if node.is_leaf():
            return node.value
        
        # Decidir qual lado seguir
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X):
        """
        Prediz os rótulos/valores para um conjunto de amostras.
        
        Parâmetros:
        -----------
        X : array-like
            Amostras para predição.
            
        Retorna:
        --------
        y_pred : array
            Rótulos/valores preditos.
        """
        X = np.array(X)
        y_pred = np.array([self._predict_sample(x, self.root) for x in X])
        
        return y_pred
    
    def score(self, X, y):
        """
        Calcula a acurácia (classificação) ou o R² (regressão) do modelo.
        
        Parâmetros:
        -----------
        X : array-like
            Amostras de teste.
        y : array-like
            Rótulos/valores verdadeiros.
            
        Retorna:
        --------
        score : float
            Acurácia ou R².
        """
        y_pred = self.predict(X)
        
        if self.task == 'classification':
            return np.mean(y_pred == y)
        else:
            u = np.sum((y - y_pred) ** 2)
            v = np.sum((y - np.mean(y)) ** 2)
            if v == 0:
                return 1
            return 1 - (u / v)
    
    def print_tree(self, node=None, depth=0):
        """
        Imprime a árvore de decisão em formato de texto.
        
        Parâmetros:
        -----------
        node : Node
            Nó atual da árvore.
        depth : int
            Profundidade atual na árvore.
        """
        if node is None:
            node = self.root
        
        indent = '    ' * depth
        
        if node.is_leaf():
            print(f"{indent}Folha: {node.value}")
        else:
            print(f"{indent}Atributo[{node.feature}] <= {node.threshold}")
            print(f"{indent}Esquerda:")
            self.print_tree(node.left, depth + 1)
            print(f"{indent}Direita:")
            self.print_tree(node.right, depth + 1)
    
    @staticmethod
    def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
        """
        Divide os dados em conjuntos de treino e teste.
        
        Parâmetros:
        -----------
        X : array-like
            Atributos.
        y : array-like
            Rótulos/valores.
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