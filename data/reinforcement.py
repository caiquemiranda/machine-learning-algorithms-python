"""
Ambiente simples para aprendizado por reforço.
Este módulo contém uma implementação de um ambiente de grade (grid world)
que pode ser usado para testar algoritmos de aprendizado por reforço.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

class AmbienteGrade:
    """
    Implementação de um ambiente de grade (Grid World) para aprendizado por reforço.
    O agente pode se mover nas quatro direções (cima, baixo, esquerda, direita).
    O ambiente contém recompensas, obstáculos e um estado terminal.
    """
    
    # Ações possíveis
    ACIMA = 0
    DIREITA = 1
    ABAIXO = 2
    ESQUERDA = 3
    
    def __init__(self, largura=5, altura=5, posicao_inicial=(0, 0), 
                recompensas=None, obstaculos=None, posicao_terminal=None,
                prob_sucesso=1.0, semente=42):
        """
        Inicializa o ambiente de grade.
        
        Parâmetros:
        -----------
        largura : int
            Largura da grade.
        altura : int
            Altura da grade.
        posicao_inicial : tuple (x, y)
            Posição inicial do agente.
        recompensas : dict {(x, y): valor}
            Dicionário mapeando posições para valores de recompensa.
        obstaculos : list [(x, y), ...]
            Lista de posições com obstáculos.
        posicao_terminal : tuple (x, y)
            Posição terminal que encerra o episódio.
        prob_sucesso : float
            Probabilidade de a ação ter sucesso (caso contrário, movimento aleatório).
        semente : int
            Semente para reprodutibilidade.
        """
        self.largura = largura
        self.altura = altura
        self.posicao_inicial = posicao_inicial
        self.posicao_atual = posicao_inicial
        self.prob_sucesso = prob_sucesso
        self.recompensa_acumulada = 0.0
        np.random.seed(semente)
        
        # Se não for especificado, criar valores padrão
        if recompensas is None:
            self.recompensas = {(largura-1, altura-1): 1.0}  # Recompensa no canto oposto
        else:
            self.recompensas = recompensas
            
        if obstaculos is None:
            self.obstaculos = []
        else:
            self.obstaculos = obstaculos
            
        if posicao_terminal is None:
            self.posicao_terminal = (largura-1, altura-1)  # Terminal no canto oposto
        else:
            self.posicao_terminal = posicao_terminal
        
        # Penalidade padrão para cada passo
        self.recompensa_padrao = -0.04
    
    def reset(self):
        """
        Reinicia o ambiente para o estado inicial.
        
        Retorna:
        --------
        tuple
            Posição inicial do agente (x, y).
        """
        self.posicao_atual = self.posicao_inicial
        self.recompensa_acumulada = 0.0
        return self.posicao_atual
    
    def passo(self, acao):
        """
        Executa uma ação no ambiente e retorna a nova posição, recompensa e status terminal.
        
        Parâmetros:
        -----------
        acao : int
            Ação a ser executada (0: acima, 1: direita, 2: abaixo, 3: esquerda).
            
        Retorna:
        --------
        tuple
            Nova posição (x, y).
        float
            Recompensa recebida.
        bool
            Indica se o estado é terminal.
        """
        # Verificar se a ação é bem-sucedida ou se há uma ação aleatória
        if np.random.random() > self.prob_sucesso:
            acao = np.random.randint(0, 4)
        
        # Calcular nova posição baseada na ação
        x, y = self.posicao_atual
        
        if acao == self.ACIMA:
            nova_y = max(0, y - 1)
            nova_x = x
        elif acao == self.DIREITA:
            nova_x = min(self.largura - 1, x + 1)
            nova_y = y
        elif acao == self.ABAIXO:
            nova_y = min(self.altura - 1, y + 1)
            nova_x = x
        elif acao == self.ESQUERDA:
            nova_x = max(0, x - 1)
            nova_y = y
        else:
            raise ValueError(f"Ação inválida: {acao}")
        
        # Verificar se a nova posição é um obstáculo
        if (nova_x, nova_y) in self.obstaculos:
            nova_x, nova_y = x, y  # Permanecer na mesma posição
        
        # Atualizar posição atual
        self.posicao_atual = (nova_x, nova_y)
        
        # Verificar se há recompensa na posição
        recompensa = self.recompensas.get(self.posicao_atual, self.recompensa_padrao)
        self.recompensa_acumulada += recompensa
        
        # Verificar se é estado terminal
        terminado = self.posicao_atual == self.posicao_terminal
        
        return self.posicao_atual, recompensa, terminado
    
    def obter_espaco_estados(self):
        """
        Retorna o número total de estados no ambiente.
        
        Retorna:
        --------
        int
            Número de estados possíveis (largura * altura).
        """
        return self.largura * self.altura
    
    def obter_espaco_acoes(self):
        """
        Retorna o número de ações possíveis.
        
        Retorna:
        --------
        int
            Número de ações (4 para este ambiente).
        """
        return 4
    
    def renderizar(self, ax=None, politica=None):
        """
        Renderiza o ambiente e, opcionalmente, a política do agente.
        
        Parâmetros:
        -----------
        ax : matplotlib.axes.Axes
            Eixo para plotagem. Se None, cria um novo.
        politica : array
            Matriz de política, onde cada posição (x, y) contém a ação recomendada.
            
        Retorna:
        --------
        matplotlib.axes.Axes
            Eixo com a plotagem.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))
        
        # Configurar tamanho e grade
        ax.set_xlim(0, self.largura)
        ax.set_ylim(0, self.altura)
        ax.set_xticks(np.arange(0, self.largura + 1, 1))
        ax.set_yticks(np.arange(0, self.altura + 1, 1))
        ax.grid(True)
        
        # Desenhar células
        for x in range(self.largura):
            for y in range(self.altura):
                cor = 'white'
                
                # Colorir obstáculos
                if (x, y) in self.obstaculos:
                    cor = 'gray'
                
                # Colorir recompensas
                elif (x, y) in self.recompensas:
                    valor = self.recompensas[(x, y)]
                    if valor > 0:
                        cor = 'green'
                    elif valor < 0:
                        cor = 'red'
                
                # Colorir posição terminal
                if (x, y) == self.posicao_terminal:
                    borda = 'gold'
                    ax.add_patch(Rectangle((x, y), 1, 1, fill=False, edgecolor=borda, lw=3))
                
                # Adicionar patch da célula
                ax.add_patch(Rectangle((x, y), 1, 1, facecolor=cor, alpha=0.3))
                
                # Adicionar valores das recompensas
                if (x, y) in self.recompensas:
                    ax.text(x + 0.5, y + 0.5, f"{self.recompensas[(x, y)]:.2f}", 
                           ha='center', va='center', fontsize=10)
        
        # Desenhar posição atual do agente
        agente_x, agente_y = self.posicao_atual
        ax.scatter(agente_x + 0.5, agente_y + 0.5, color='blue', s=200, marker='o')
        
        # Desenhar política, se fornecida
        if politica is not None:
            for x in range(self.largura):
                for y in range(self.altura):
                    # Pular obstáculos
                    if (x, y) in self.obstaculos:
                        continue
                    
                    acao = politica[y, x]  # Nota: y, x para acessar matriz
                    
                    # Desenhar seta indicando a direção da política
                    if acao == self.ACIMA:
                        dx, dy = 0, -0.3
                    elif acao == self.DIREITA:
                        dx, dy = 0.3, 0
                    elif acao == self.ABAIXO:
                        dx, dy = 0, 0.3
                    elif acao == self.ESQUERDA:
                        dx, dy = -0.3, 0
                    
                    ax.arrow(x + 0.5, y + 0.5, dx, dy, head_width=0.1, head_length=0.1, 
                            fc='black', ec='black')
        
        ax.set_title('Ambiente de Grade para Aprendizado por Reforço')
        ax.invert_yaxis()  # Para que (0,0) fique no canto superior esquerdo
        
        return ax

def criar_ambiente_padrao():
    """
    Cria um ambiente de grade padrão para demonstração.
    
    Retorna:
    --------
    AmbienteGrade
        Um ambiente de grade 5x5 com obstáculos e recompensas.
    """
    # Criar ambiente 5x5
    largura, altura = 5, 5
    
    # Definir obstáculos
    obstaculos = [(1, 1), (1, 2), (2, 1), (3, 3)]
    
    # Definir recompensas
    recompensas = {
        (4, 4): 1.0,    # Recompensa positiva no canto inferior direito
        (4, 0): -1.0,   # Recompensa negativa em uma posição
        (0, 4): -1.0    # Recompensa negativa em outra posição
    }
    
    # Posição terminal
    posicao_terminal = (4, 4)
    
    # Criar ambiente
    return AmbienteGrade(
        largura=largura,
        altura=altura,
        posicao_inicial=(0, 0),
        recompensas=recompensas,
        obstaculos=obstaculos,
        posicao_terminal=posicao_terminal,
        prob_sucesso=0.8
    )

# Exemplo de uso
if __name__ == "__main__":
    # Criar ambiente
    ambiente = criar_ambiente_padrao()
    
    # Criar política aleatória para exemplo
    politica = np.random.randint(0, 4, size=(ambiente.altura, ambiente.largura))
    
    # Renderizar ambiente
    plt.figure(figsize=(10, 10))
    ambiente.renderizar(politica=politica)
    plt.tight_layout()
    plt.savefig("ambiente_reinforcement.png", dpi=100)
    plt.show()
    
    # Demonstrar alguns passos no ambiente
    print("Demonstração de passos no ambiente:")
    ambiente.reset()
    terminado = False
    
    for i in range(10):
        if terminado:
            print("Episódio terminado!")
            break
            
        # Escolher ação aleatória
        acao = np.random.randint(0, 4)
        acao_nome = ["ACIMA", "DIREITA", "ABAIXO", "ESQUERDA"][acao]
        
        print(f"Passo {i+1}: Ação = {acao_nome}", end=" ")
        
        # Executar passo
        posicao, recompensa, terminado = ambiente.passo(acao)
        
        print(f"-> Nova posição: {posicao}, Recompensa: {recompensa}")
        
    print(f"Recompensa total acumulada: {ambiente.recompensa_acumulada:.2f}") 