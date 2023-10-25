
import gym
from Flappybat import *
from stable_baselines3 import PPO
from gym import spaces
import numpy as np
modelo_carregado = PPO.load("FlappyBat.zip")

class FlappyBatEnv(gym.Env):
    def __init__(self):
        # Inicialização do Pygame
        pygame.init()
        self.tela = pygame.display.set_mode((TELA_LARGURA, TELA_ALTURA))
        pygame.display.set_caption('Flappy Bat Environment')
        self.clock = pygame.time.Clock()
        self.last_action = None
        self.highest_score = 0
        # Definir ações possíveis (0: Não pular, 1: Pular)
        self.action_space = spaces.Discrete(2)

        # Definir espaço de observação
        self.observation_space = spaces.Box(
            low=np.array([0, -np.inf, 0, -np.inf, 0, 0]),  # adicionado limites inferiores para novas observações
            high=np.array([TELA_ALTURA, np.inf, TELA_ALTURA, np.inf, TELA_ALTURA, TELA_ALTURA]),  # adicionado limites superiores para novas observações
            dtype=np.float32
        )

        # Inicializar jogo
        self.reset()

    def reset(self):
        self.morcego = bat(230, 350)
        self.chao = Chao(730)
        self.canos = [Cano(700)]
        self.pontos = 0

        # Estado inicial
        proximo_cano = self.canos[0]
        estado = [
            self.morcego.y,
            proximo_cano.x - self.morcego.x,
            proximo_cano.altura,
            self.morcego.velocidade,
            proximo_cano.pos_topo,
            proximo_cano.pos_base
        ]
        return np.array(estado)

    def step(self, action):
        # Aplicar ação
        if action == 1:
            self.morcego.pular()

        # Atualizar jogo
        self.morcego.mover()
        self.chao.mover()
        recompensa = 0
        done = False

        adicionar_cano = False
        remover_canos = []
        for cano in self.canos:
            if cano.colidir(self.morcego):
                recompensa -= 100 + self.pontos
                done = True
            if not cano.passou and self.morcego.x > cano.x:
                cano.passou = True
                adicionar_cano = True
            cano.mover()
            if cano.x + cano.CANO_TOPO.get_width() < 0:
                remover_canos.append(cano)

        if adicionar_cano:
            recompensa += 5 + (self.pontos + 1)
            self.pontos += 1
            
            self.canos.append(Cano(600))
        for cano in remover_canos:
            self.canos.remove(cano)

        if (self.morcego.y + self.morcego.imagem.get_height()) > self.chao.y or self.morcego.y < 0:
            done = True
            recompensa -= 100
        
        #if action == 1 and self.last_action == 1:
            #recompensa -= 1
        
        self.last_action = action

        # Atualizar observação
        if not done:
            proximo_cano = self.canos[0]
            for cano in self.canos:
                if cano.x > self.morcego.x:
                    proximo_cano = cano
                    if self.morcego.y < proximo_cano.pos_topo+ 100 or self.morcego.y > proximo_cano.pos_base - 100:
                        recompensa -= 10
                    #elif self.morcego.y > proximo_cano.pos_topo+ 100 or self.morcego.y < proximo_cano.pos_base - 100:
                        #recompensa += 0.7
                    break
            estado = [
                self.morcego.y,
                proximo_cano.x - self.morcego.x,
                proximo_cano.altura,
                self.morcego.velocidade,
                proximo_cano.pos_topo,
                proximo_cano.pos_base
            ]
        else:
            estado = self.reset()
        self.render()
        return np.array(estado), recompensa, done, {}

    def render(self, mode='human'):
        if mode == 'human':
            # Limpar tela
            self.tela.fill((0, 0, 0))  # Preenche com preto, pode ser substituído por outra cor ou imagem de fundo
            self.desenhar_pontuacao(self.pontos)
            if self.pontos > self.highest_score:
                self.highest_score = self.pontos
            #self.clock.tick(60)
            #if self.pontos>1:
                #self.clock.tick(60)
            # Desenhar elementos (a ordem é importante)
            for cano in self.canos:
                cano.desenhar(self.tela)
            self.chao.desenhar(self.tela)
            self.morcego.desenhar(self.tela)

            # Atualizar display
            pygame.display.flip()

            # Tratar eventos (opcional, mas pode evitar que a janela pare de responder)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
    def desenhar_pontuacao(self, pontos):
        font = pygame.font.SysFont(None, 55)
        texto = font.render(str(pontos), True, (255, 255, 255))
        self.tela.blit(texto, (TELA_LARGURA // 2 - texto.get_width() // 2, 50))


def lr_schedule(progress_remaining):
    return 0.0003 * progress_remaining

env = FlappyBatEnv()
modelo_carregado.set_env(env)
modelo_carregado.learn(total_timesteps=100000)
print("Highest Score During Training:", env.highest_score)