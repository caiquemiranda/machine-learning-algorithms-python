# Algoritmos de Machine Learning em Python

Este repositório contém implementações do zero de vários algoritmos de Machine Learning em Python, usando principalmente NumPy. As implementações são educacionais, destinadas a fornecer uma compreensão clara dos princípios fundamentais por trás de cada algoritmo.

## Estrutura do Repositório

O repositório está organizado nas seguintes seções:

### Algoritmos de Aprendizado Supervisionado

Implementações de algoritmos que aprendem uma função que mapeia entradas para saídas com base em pares de exemplos entrada-saída.

- **K-Nearest Neighbors (KNN)**: Algoritmo baseado em instâncias que classifica novos dados com base na similaridade com os dados de treinamento.
- **Naive Bayes**: Classificador probabilístico baseado no teorema de Bayes que assume independência entre os atributos.
- **Árvore de Decisão**: Modelo que constrói uma árvore de decisão a partir dos dados de treinamento para regressão e classificação.
- **Máquina de Vetores de Suporte (SVM)**: Algoritmo que encontra um hiperplano ótimo para separar classes em um espaço de alta dimensionalidade.
- **Regressão Linear**: Algoritmo para modelar a relação entre uma variável dependente e uma ou mais variáveis independentes.

### Algoritmos de Aprendizado Não Supervisionado

Implementações de algoritmos que aprendem padrões e estruturas a partir de dados não rotulados.

- **K-Means**: Algoritmo de agrupamento que divide os dados em K clusters, com cada ponto pertencendo ao cluster com o centroide mais próximo.
- **PCA (Principal Component Analysis)**: Técnica de redução de dimensionalidade que encontra as direções de maior variância nos dados.

### Algoritmos de Aprendizado por Reforço

Implementações de algoritmos que aprendem a tomar ações em um ambiente para maximizar alguma noção de recompensa cumulativa.

- **Q-Learning**: Algoritmo de aprendizado por reforço baseado em valores que aprende uma política ótima através da função Q.

### Deep Learning

Implementações de algoritmos de aprendizado profundo baseados em redes neurais.

- **Rede Neural**: Implementação de uma rede neural feed-forward com retropropagação implementada manualmente.

## Requisitos

- Python 3.6+
- NumPy
- Matplotlib (para visualizações)

## Como Usar

Cada algoritmo está contido em seu próprio arquivo Python e pode ser importado e utilizado em seus projetos:

```python
# Exemplo de uso do KNN
from supervised_learning.k_nearest_neighbor import KNN

# Criar classificador
knn = KNN(k=3)

# Treinar o modelo
knn.fit(X_train, y_train)

# Fazer predições
y_pred = knn.predict(X_test)

# Calcular acurácia
accuracy = knn.accuracy_score(y_test, y_pred)
```

Muitos algoritmos incluem funções auxiliares para dividir dados em conjuntos de treinamento e teste, normalizar dados, plotar resultados, etc.

## Exemplos

Cada arquivo contém exemplos de uso no bloco `if __name__ == "__main__":` que demonstram como usar o algoritmo. Execute o arquivo diretamente para ver o algoritmo em ação:

```bash
python supervised-learning/k-nearest-neighborg.py
```

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests para melhorar as implementações ou adicionar novos algoritmos.

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.
