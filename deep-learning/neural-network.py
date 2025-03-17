# Rede Neural

import numpy as np
import matplotlib.pyplot as plt

class Layer:
    """
    Camada de uma rede neural.
    """
    def __init__(self):
        self.input = None
        self.output = None
        
    def forward(self, input):
        """
        Passo forward: calcula a saída da camada.
        """
        pass
    
    def backward(self, output_gradient, learning_rate):
        """
        Passo backward: atualiza os parâmetros e retorna o gradiente de entrada.
        """
        pass


class Dense(Layer):
    """
    Camada densa (totalmente conectada).
    """
    def __init__(self, input_size, output_size):
        """
        Inicializa a camada densa com pesos e bias aleatórios.
        
        Parâmetros:
        -----------
        input_size : int
            Tamanho da entrada da camada.
        output_size : int
            Tamanho da saída da camada.
        """
        super().__init__()
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.bias = np.zeros((output_size, 1))
        
    def forward(self, input):
        """
        Passo forward: y = wx + b
        
        Parâmetros:
        -----------
        input : array, shape (input_size, batch_size)
            Entrada da camada.
            
        Retorna:
        --------
        output : array, shape (output_size, batch_size)
            Saída da camada.
        """
        self.input = input
        self.output = np.dot(self.weights, self.input) + self.bias
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        """
        Passo backward: atualiza os pesos e bias, retorna o gradiente de entrada.
        
        Parâmetros:
        -----------
        output_gradient : array, shape (output_size, batch_size)
            Gradiente da função de custo em relação à saída da camada.
        learning_rate : float
            Taxa de aprendizado.
            
        Retorna:
        --------
        input_gradient : array, shape (input_size, batch_size)
            Gradiente da função de custo em relação à entrada da camada.
        """
        # Gradiente em relação aos pesos: dL/dW = dL/dY * X^T
        weights_gradient = np.dot(output_gradient, self.input.T)
        
        # Gradiente em relação ao bias: dL/dB = dL/dY
        bias_gradient = np.sum(output_gradient, axis=1, keepdims=True)
        
        # Gradiente em relação à entrada: dL/dX = W^T * dL/dY
        input_gradient = np.dot(self.weights.T, output_gradient)
        
        # Atualizar parâmetros
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient
        
        return input_gradient


class Activation(Layer):
    """
    Camada de ativação.
    """
    def __init__(self, activation, activation_prime):
        """
        Inicializa a camada de ativação.
        
        Parâmetros:
        -----------
        activation : função
            Função de ativação.
        activation_prime : função
            Derivada da função de ativação.
        """
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime
        
    def forward(self, input):
        """
        Passo forward: aplica a função de ativação.
        
        Parâmetros:
        -----------
        input : array
            Entrada da camada.
            
        Retorna:
        --------
        output : array
            Saída após a aplicação da função de ativação.
        """
        self.input = input
        self.output = self.activation(self.input)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        """
        Passo backward: retorna o gradiente da entrada.
        
        Parâmetros:
        -----------
        output_gradient : array
            Gradiente da função de custo em relação à saída da camada.
        learning_rate : float
            Taxa de aprendizado (não utilizada nesta camada).
            
        Retorna:
        --------
        input_gradient : array
            Gradiente da função de custo em relação à entrada da camada.
        """
        # Aplicação da regra da cadeia: dL/dX = dL/dY * dY/dX
        return self.activation_prime(self.input) * output_gradient


# Funções de ativação comuns e suas derivadas
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return np.where(x > 0, 1, 0)

def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)

def leaky_relu_prime(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def softmax(x):
    # Subtrai o máximo para estabilidade numérica
    exp_values = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_values / np.sum(exp_values, axis=0, keepdims=True)

def softmax_prime(x):
    # Na prática, esta derivada não é utilizada diretamente, 
    # pois a derivada da cross-entropy com softmax é simplificada
    s = softmax(x)
    return s * (1 - s)


# Camadas de ativação pré-definidas
class Sigmoid(Activation):
    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime)
        
class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)
        
class ReLU(Activation):
    def __init__(self):
        super().__init__(relu, relu_prime)
        
class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        super().__init__(
            lambda x: leaky_relu(x, alpha),
            lambda x: leaky_relu_prime(x, alpha)
        )

class Softmax(Layer):
    def forward(self, input):
        self.input = input
        self.output = softmax(input)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        # Este é um caso especial, normalmente calculado junto com a função de custo
        # Para simplificar, assumimos que estamos usando cross-entropy
        n_samples = self.input.shape[1]
        return output_gradient / n_samples


# Funções de custo e suas derivadas
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.shape[1]

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / y_true.shape[1]

def categorical_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[1]

def categorical_cross_entropy_prime(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -y_true / y_pred / y_true.shape[1]

# Para cross-entropy com softmax, a derivada é simplificada: dL/dX = Y_pred - Y_true
def softmax_cross_entropy_prime(y_true, y_pred):
    return y_pred - y_true


class NeuralNetwork:
    """
    Rede neural de múltiplas camadas.
    """
    def __init__(self):
        """
        Inicializa a rede neural com uma lista vazia de camadas.
        """
        self.layers = []
        self.loss = None
        self.loss_prime = None
        
    def add(self, layer):
        """
        Adiciona uma camada à rede.
        
        Parâmetros:
        -----------
        layer : Layer
            Camada a ser adicionada.
        """
        self.layers.append(layer)
        
    def use(self, loss, loss_prime):
        """
        Define a função de custo a ser utilizada.
        
        Parâmetros:
        -----------
        loss : função
            Função de custo.
        loss_prime : função
            Derivada da função de custo.
        """
        self.loss = loss
        self.loss_prime = loss_prime
        
    def predict(self, input):
        """
        Realiza a predição para uma entrada.
        
        Parâmetros:
        -----------
        input : array
            Entrada da rede.
            
        Retorna:
        --------
        output : array
            Saída predita pela rede.
        """
        # Propagação forward através de todas as camadas
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def fit(self, x_train, y_train, epochs, learning_rate, batch_size=None, x_val=None, y_val=None, verbose=True):
        """
        Treina a rede neural.
        
        Parâmetros:
        -----------
        x_train : array
            Dados de entrada para treinamento.
        y_train : array
            Rótulos/valores alvo para treinamento.
        epochs : int
            Número de épocas de treinamento.
        learning_rate : float
            Taxa de aprendizado.
        batch_size : int ou None
            Tamanho do lote. Se None, usa todo o conjunto de dados.
        x_val : array ou None
            Dados de entrada para validação.
        y_val : array ou None
            Rótulos/valores alvo para validação.
        verbose : bool
            Se True, exibe progresso durante o treinamento.
            
        Retorna:
        --------
        history : dict
            Histórico de treinamento com perda por época.
        """
        # Histórico de treinamento
        history = {
            'loss': [],
            'val_loss': [] if x_val is not None and y_val is not None else None
        }
        
        n_samples = x_train.shape[1]
        
        # Dividir dados em lotes, se necessário
        if batch_size is None:
            batch_size = n_samples
        
        n_batches = n_samples // batch_size
        
        # Loop de treinamento
        for e in range(epochs):
            # Embaralhar os dados a cada época
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[:, indices]
            y_shuffled = y_train[:, indices]
            
            loss = 0
            
            # Loop pelos lotes
            for i in range(n_batches):
                # Preparar lote
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                x_batch = x_shuffled[:, start_idx:end_idx]
                y_batch = y_shuffled[:, start_idx:end_idx]
                
                # Propagação forward
                output = self.predict(x_batch)
                
                # Computar perda
                loss += self.loss(y_batch, output)
                
                # Propagação backward
                grad = self.loss_prime(y_batch, output)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)
            
            # Calcular perda média por época
            loss /= n_batches
            history['loss'].append(loss)
            
            # Calcular perda de validação, se fornecida
            if x_val is not None and y_val is not None:
                val_output = self.predict(x_val)
                val_loss = self.loss(y_val, val_output)
                history['val_loss'].append(val_loss)
                
                if verbose:
                    print(f"Época {e+1}/{epochs}, Perda: {loss:.4f}, Perda de validação: {val_loss:.4f}")
            else:
                if verbose:
                    print(f"Época {e+1}/{epochs}, Perda: {loss:.4f}")
        
        return history
    
    def evaluate(self, x_test, y_test):
        """
        Avalia o desempenho da rede em dados de teste.
        
        Parâmetros:
        -----------
        x_test : array
            Dados de entrada para teste.
        y_test : array
            Rótulos/valores alvo para teste.
            
        Retorna:
        --------
        loss : float
            Valor da função de custo nos dados de teste.
        """
        output = self.predict(x_test)
        return self.loss(y_test, output)

    def plot_decision_boundary(self, X, y, h=0.01, ax=None):
        """
        Plota a fronteira de decisão para dados 2D.
        
        Parâmetros:
        -----------
        X : array, shape (2, n_samples)
            Dados de entrada com 2 atributos.
        y : array, shape (n_classes, n_samples)
            Rótulos one-hot.
        h : float
            Passo da grade.
        ax : matplotlib.axes.Axes
            Eixo para plotagem.
            
        Retorna:
        --------
        ax : matplotlib.axes.Axes
            Eixo com a plotagem.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Definir limites da grade
        x_min, x_max = X[0].min() - 1, X[0].max() + 1
        y_min, y_max = X[1].min() - 1, X[1].max() + 1
        
        # Criar grade
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # Achatamos a grade para um array 2D e fazemos a predição
        grid = np.c_[xx.ravel(), yy.ravel()].T
        Z = self.predict(grid)
        
        # Converter saída softmax em classes
        if Z.shape[0] > 1:  # Classificação multiclasse
            Z = np.argmax(Z, axis=0)
        else:  # Classificação binária
            Z = (Z > 0.5).astype(int).ravel()
        
        # Remodelar para a forma da grade
        Z = Z.reshape(xx.shape)
        
        # Plotar contorno da fronteira de decisão
        ax.contourf(xx, yy, Z, alpha=0.3)
        
        # Converter rótulos one-hot para classes
        if y.shape[0] > 1:  # One-hot encoding
            y_classes = np.argmax(y, axis=0)
        else:  # Binário
            y_classes = y.ravel()
        
        # Plotar pontos de dados
        scatter = ax.scatter(X[0], X[1], c=y_classes, edgecolors='k', alpha=0.8)
        
        # Adicionar legenda
        legend1 = ax.legend(*scatter.legend_elements(),
                           loc="upper right", title="Classes")
        ax.add_artist(legend1)
        
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Atributo 1')
        ax.set_ylabel('Atributo 2')
        ax.set_title('Fronteira de Decisão')
        
        return ax

    def plot_training_history(self, history, ax=None):
        """
        Plota o histórico de treinamento.
        
        Parâmetros:
        -----------
        history : dict
            Histórico retornado pelo método fit.
        ax : matplotlib.axes.Axes
            Eixo para plotagem.
            
        Retorna:
        --------
        ax : matplotlib.axes.Axes
            Eixo com a plotagem.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(history['loss']) + 1)
        
        ax.plot(epochs, history['loss'], 'b-', label='Perda de treinamento')
        
        if history['val_loss'] is not None:
            ax.plot(epochs, history['val_loss'], 'r-', label='Perda de validação')
        
        ax.set_xlabel('Época')
        ax.set_ylabel('Perda')
        ax.set_title('Histórico de Treinamento')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return ax


# Exemplo de uso
if __name__ == "__main__":
    # Exemplo de XOR
    # Dados de entrada
    X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    
    # Saída esperada (XOR: 0⊕0=0, 0⊕1=1, 1⊕0=1, 1⊕1=0)
    Y = np.array([[0, 1, 1, 0]])
    
    # Criar rede neural
    model = NeuralNetwork()
    model.add(Dense(2, 4))
    model.add(ReLU())
    model.add(Dense(4, 1))
    model.add(Sigmoid())
    
    # Definir função de custo
    model.use(binary_cross_entropy, binary_cross_entropy_prime)
    
    # Treinar rede
    history = model.fit(X, Y, epochs=1000, learning_rate=0.1, verbose=True)
    
    # Avaliar resultados
    output = model.predict(X)
    print("Entrada:")
    print(X.T)
    print("\nSaída Esperada:")
    print(Y.T)
    print("\nSaída Predita:")
    print(output.T)
    print("\nSaída Arredondada:")
    print(np.round(output).T)
    
    # Plotar fronteira de decisão
    model.plot_decision_boundary(X, Y)
    plt.show()
    
    # Plotar histórico de treinamento
    model.plot_training_history(history)
    plt.show() 