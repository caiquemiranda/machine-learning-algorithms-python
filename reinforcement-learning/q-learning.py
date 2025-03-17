# Q-Learning

import numpy as np
import matplotlib.pyplot as plt
import time

class QLearning:
    """
    Implementação do algoritmo Q-Learning para aprendizado por reforço.
    """
    
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.9, 
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01):
        """
        Inicializa o agente Q-Learning.
        
        Parâmetros:
        -----------
        n_states : int
            Número de estados no ambiente.
        n_actions : int
            Número de ações possíveis.
        learning_rate : float
            Taxa de aprendizado (alpha) - controla o quanto novos valores substituem os antigos.
        discount_factor : float
            Fator de desconto (gamma) - controla a importância de recompensas futuras.
        exploration_rate : float
            Taxa de exploração inicial (epsilon) - probabilidade de escolher uma ação aleatória.
        exploration_decay : float
            Taxa de decaimento da exploração - reduz epsilon a cada episódio.
        min_exploration_rate : float
            Taxa mínima de exploração - limite inferior para epsilon.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        
        # Inicializar tabela Q com zeros
        self.q_table = np.zeros((n_states, n_actions))
        
        # Histórico de treinamento
        self.rewards_history = []
        self.steps_history = []
        
    def choose_action(self, state):
        """
        Seleciona uma ação usando a política epsilon-greedy.
        
        Parâmetros:
        -----------
        state : int
            Estado atual do ambiente.
            
        Retorna:
        --------
        action : int
            Ação escolhida.
        """
        # Exploração: escolher ação aleatória
        if np.random.random() < self.exploration_rate:
            return np.random.randint(self.n_actions)
        
        # Explotação: escolher a melhor ação da tabela Q
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """
        Atualiza a tabela Q com base na experiência.
        
        Parâmetros:
        -----------
        state : int
            Estado atual.
        action : int
            Ação tomada.
        reward : float
            Recompensa recebida.
        next_state : int
            Próximo estado.
        done : bool
            Se o episódio terminou.
        """
        # Valor Q atual
        current_q = self.q_table[state, action]
        
        # Valor Q máximo para o próximo estado
        if done:
            max_next_q = 0  # Não há próximo estado se o episódio terminou
        else:
            max_next_q = np.max(self.q_table[next_state])
        
        # Fórmula de atualização Q-Learning: Q(s,a) = Q(s,a) + α * (r + γ * max Q(s',a') - Q(s,a))
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        # Atualizar valor na tabela Q
        self.q_table[state, action] = new_q
    
    def decay_exploration(self):
        """
        Reduz a taxa de exploração (epsilon) ao longo do tempo.
        """
        self.exploration_rate = max(self.min_exploration_rate, 
                                   self.exploration_rate * self.exploration_decay)
    
    def train(self, env, n_episodes=1000, max_steps=100, render=False, render_freq=100):
        """
        Treina o agente Q-Learning no ambiente.
        
        Parâmetros:
        -----------
        env : objeto ambiente
            Ambiente que implementa reset(), step(action) e render().
        n_episodes : int
            Número de episódios para treinar.
        max_steps : int
            Número máximo de passos por episódio.
        render : bool
            Se True, renderiza o ambiente durante o treinamento.
        render_freq : int
            Frequência de renderização (a cada quantos episódios).
            
        Retorna:
        --------
        self : objeto
            Retorna o próprio objeto.
        """
        for episode in range(n_episodes):
            # Resetar o ambiente
            state = env.reset()
            total_reward = 0
            steps = 0
            
            for step in range(max_steps):
                # Escolher uma ação
                action = self.choose_action(state)
                
                # Executar a ação e obter resultados
                next_state, reward, done, info = env.step(action)
                
                # Renderizar se necessário
                if render and episode % render_freq == 0:
                    env.render()
                    time.sleep(0.1)  # Pequeno atraso para visualização
                
                # Atualizar a tabela Q
                self.update(state, action, reward, next_state, done)
                
                # Transição para o próximo estado
                state = next_state
                total_reward += reward
                steps += 1
                
                # Terminar episódio se o ambiente indicar
                if done:
                    break
            
            # Reduzir taxa de exploração
            self.decay_exploration()
            
            # Registrar histórico
            self.rewards_history.append(total_reward)
            self.steps_history.append(steps)
            
            # Log a cada 100 episódios
            if episode % 100 == 0:
                avg_reward = np.mean(self.rewards_history[-100:]) if len(self.rewards_history) >= 100 else np.mean(self.rewards_history)
                print(f"Episódio {episode}: Recompensa média = {avg_reward:.2f}, Epsilon = {self.exploration_rate:.4f}")
        
        return self
    
    def plot_training_results(self):
        """
        Plota os resultados do treinamento: recompensas e passos por episódio.
        
        Retorna:
        --------
        fig : matplotlib.figure.Figure
            Figura com os gráficos.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plotar recompensas
        ax1.plot(self.rewards_history)
        ax1.set_xlabel('Episódio')
        ax1.set_ylabel('Recompensa Total')
        ax1.set_title('Recompensas por Episódio')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Adicionar média móvel para recompensas
        if len(self.rewards_history) >= 100:
            avg_rewards = [np.mean(self.rewards_history[max(0, i-100):i]) 
                           for i in range(1, len(self.rewards_history)+1)]
            ax1.plot(avg_rewards, 'r-', label='Média Móvel (100 episódios)')
            ax1.legend()
        
        # Plotar passos
        ax2.plot(self.steps_history)
        ax2.set_xlabel('Episódio')
        ax2.set_ylabel('Número de Passos')
        ax2.set_title('Passos por Episódio')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Adicionar média móvel para passos
        if len(self.steps_history) >= 100:
            avg_steps = [np.mean(self.steps_history[max(0, i-100):i]) 
                         for i in range(1, len(self.steps_history)+1)]
            ax2.plot(avg_steps, 'r-', label='Média Móvel (100 episódios)')
            ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def save_model(self, filename):
        """
        Salva a tabela Q em um arquivo.
        
        Parâmetros:
        -----------
        filename : str
            Nome do arquivo para salvar a tabela Q.
        """
        np.save(filename, self.q_table)
        print(f"Modelo salvo em {filename}")
    
    def load_model(self, filename):
        """
        Carrega a tabela Q de um arquivo.
        
        Parâmetros:
        -----------
        filename : str
            Nome do arquivo para carregar a tabela Q.
            
        Retorna:
        --------
        self : objeto
            Retorna o próprio objeto.
        """
        try:
            self.q_table = np.load(filename)
            print(f"Modelo carregado de {filename}")
            
            # Verificar se as dimensões da tabela carregada correspondem às esperadas
            if self.q_table.shape != (self.n_states, self.n_actions):
                raise ValueError(f"Dimensões da tabela Q carregada ({self.q_table.shape}) não correspondem às esperadas ({self.n_states}, {self.n_actions}).")
                
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            
        return self
    
    def visualize_policy(self, env, grid_size=None):
        """
        Visualiza a política aprendida para ambientes em grade 2D.
        
        Parâmetros:
        -----------
        env : objeto ambiente
            Ambiente para visualização.
        grid_size : tuple
            Tamanho da grade (linhas, colunas). Se None, tenta inferir do ambiente.
            
        Retorna:
        --------
        fig : matplotlib.figure.Figure
            Figura com a visualização da política.
        """
        if grid_size is None:
            # Tentar inferir tamanho da grade do ambiente
            # Isso depende da implementação específica do ambiente
            try:
                grid_size = env.shape
            except:
                grid_size = (int(np.sqrt(self.n_states)), int(np.sqrt(self.n_states)))
                print(f"Tamanho da grade não especificado. Usando {grid_size}.")
        
        nrows, ncols = grid_size
        
        # Verificar se o número de estados corresponde ao tamanho da grade
        if nrows * ncols != self.n_states:
            print(f"Aviso: Número de estados ({self.n_states}) não corresponde ao tamanho da grade ({nrows * ncols}).")
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Definir símbolos para ações (assumindo 4 ações: cima, direita, baixo, esquerda)
        action_symbols = ['↑', '→', '↓', '←']
        
        # Definir cores para ações
        action_colors = ['blue', 'red', 'green', 'purple']
        
        # Plotar setas para cada estado
        for row in range(nrows):
            for col in range(ncols):
                state = row * ncols + col
                if state < self.n_states:
                    best_action = np.argmax(self.q_table[state])
                    action_val = np.max(self.q_table[state])
                    
                    # Posição no centro da célula
                    x, y = col + 0.5, nrows - row - 0.5
                    
                    if self.n_actions == 4:  # Se for ambiente com 4 ações, usar setas
                        # Plotar seta na direção da melhor ação
                        dx, dy = 0, 0
                        if best_action == 0:  # Cima
                            dx, dy = 0, 0.3
                        elif best_action == 1:  # Direita
                            dx, dy = 0.3, 0
                        elif best_action == 2:  # Baixo
                            dx, dy = 0, -0.3
                        elif best_action == 3:  # Esquerda
                            dx, dy = -0.3, 0
                        
                        ax.arrow(x, y, dx, dy, head_width=0.15, head_length=0.15, 
                                fc=action_colors[best_action], ec=action_colors[best_action])
                    else:
                        # Usar texto para representar a ação
                        ax.text(x, y, str(best_action), color='black', ha='center', va='center',
                               fontsize=12, fontweight='bold')
                    
                    # Adicionar valor Q
                    if action_val != 0:
                        ax.text(x, y-0.2, f"{action_val:.1f}", color='black', ha='center', va='center',
                               fontsize=8)
        
        # Configurar grade
        ax.set_xlim(0, ncols)
        ax.set_ylim(0, nrows)
        ax.set_xticks(np.arange(0, ncols + 1, 1))
        ax.set_yticks(np.arange(0, nrows + 1, 1))
        ax.grid(True)
        ax.set_title('Política Aprendida')
        
        # Adicionar legenda para ações
        if self.n_actions == 4:
            for i, (symbol, color) in enumerate(zip(action_symbols, action_colors)):
                ax.plot([], [], color=color, marker=symbol, linestyle='none', 
                       markersize=15, label=f'Ação {i}: {symbol}')
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
        
        plt.tight_layout()
        return fig


# Exemplo de uso com um ambiente simples (grade 4x4 com recompensa no canto)
class SimpleGridEnv:
    """
    Ambiente simples de grade para demonstração do Q-Learning.
    """
    
    def __init__(self, size=4):
        self.size = size
        self.shape = (size, size)
        self.n_states = size * size
        self.n_actions = 4  # Cima, Direita, Baixo, Esquerda
        self.reset()
        
    def reset(self):
        # Começar no canto superior esquerdo
        self.position = (0, 0)
        return self._get_state()
    
    def _get_state(self):
        # Converter posição (linha, coluna) para estado único
        return self.position[0] * self.size + self.position[1]
    
    def step(self, action):
        # Cima
        if action == 0:
            next_position = (max(0, self.position[0] - 1), self.position[1])
        # Direita
        elif action == 1:
            next_position = (self.position[0], min(self.size - 1, self.position[1] + 1))
        # Baixo
        elif action == 2:
            next_position = (min(self.size - 1, self.position[0] + 1), self.position[1])
        # Esquerda
        elif action == 3:
            next_position = (self.position[0], max(0, self.position[1] - 1))
        else:
            raise ValueError("Ação inválida")
        
        self.position = next_position
        state = self._get_state()
        
        # Recompensa quando atinge o objetivo (canto inferior direito)
        if self.position == (self.size - 1, self.size - 1):
            reward = 1.0
            done = True
        else:
            reward = 0.0
            done = False
        
        return state, reward, done, {}
    
    def render(self):
        # Renderização simples da grade
        grid = [['[ ]' for _ in range(self.size)] for _ in range(self.size)]
        
        # Marcar posição do agente
        grid[self.position[0]][self.position[1]] = '[A]'
        
        # Marcar objetivo
        grid[self.size-1][self.size-1] = '[G]'
        
        # Imprimir grade
        print('-' * (self.size * 3 + 2))
        for row in grid:
            print('|' + ''.join(row) + '|')
        print('-' * (self.size * 3 + 2))


# Demonstração de uso
if __name__ == "__main__":
    # Criar ambiente
    env = SimpleGridEnv(size=4)
    
    # Criar agente Q-Learning
    agent = QLearning(n_states=env.n_states, 
                      n_actions=env.n_actions,
                      learning_rate=0.1,
                      discount_factor=0.9,
                      exploration_rate=1.0,
                      exploration_decay=0.995,
                      min_exploration_rate=0.01)
    
    # Treinar agente
    agent.train(env, n_episodes=1000, max_steps=100, render=True, render_freq=500)
    
    # Plotar resultados
    agent.plot_training_results()
    plt.show()
    
    # Visualizar política aprendida
    agent.visualize_policy(env)
    plt.show() 