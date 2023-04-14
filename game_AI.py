import pygame, random
import torch as T # we only use : T.tensor
from model import Net_FlappyBird

from constants import *
from game import Game, Bird


class Bird_AI(Bird):

    def __init__(self):
        super().__init__()
        self.brain = Net_FlappyBird(input_size=5)
    
    # 0 -> not jump / 1 -> jump
    def move(self, actual_pipe_top, actual_pipe_bottom):

        # get data (vision) 
        dist_top    = abs(self.rect.centery - actual_pipe_top.rect.bottom)
        dist_bottom = abs(self.rect.centery - actual_pipe_bottom.rect.top)
        dist_pipe   = abs(self.rect.centerx - actual_pipe_top.rect.centerx) 
        height      = abs(YLIM - self.rect.centery) 
        velocity    = self.movement
        
        # feed it to the neural network
        state = [dist_top, dist_bottom, dist_pipe, height, velocity]
        state = T.tensor(state, dtype=T.float)
        evaluation = self.brain.forward(state)
        jump = True if evaluation > 0.5 else False
        
        # move or not depending on the results
        if jump:
            self.movement = 0
            if self.rect.top > 0: 
                self.movement -= 5

    # def kill(self):
    #     super().kill()
    #     return self.score

    def load_model(self, filename):
        self.brain.load_state_dict(T.load(filename))


class Game_AI(Game):

    def __init__(self):
        super().__init__()
        self.bird = Bird_AI()

    def move(self, event):
        self.bird.move(self.actual_pipe_top, self.actual_pipe_bottom)

    def train(self, n_generations, n_birds):
        for g in n_generations:
            birds_gen = pygame.sprite.Group()
            for _ in range(n_birds):
                bird = Bird_AI()
                birds_gen.add(bird)
            scores = self.play_generation(birds_gen)
            print(f"Gen [{g+1}] Scores : {scores}")



    def play_generation(self, birds_gen):

        scores = []

        clock = pygame.time.Clock()
        run = True
        SPAWNPIPE = False
        self.reset_pipes()

        i = 0
        while run:
            i += 1
    
            clock.tick(120)
            
            for bird in birds_gen:
                bird.move(self.actual_pipe_top, self.actual_pipe_bottom)
                        
            # Create new pipes
            if SPAWNPIPE:
                SPAWNPIPE = False
                self.add_pipes()
            elif len(self.pipes) > 0 and self.pipes.sprites()[-1].rect.x < 450-300:
                SPAWNPIPE = True

            # update actual pipe
            if self.actual_pipe_bottom.rect.right < self.bird.rect.centerx:
                self.actual_pipe_top = self.pipes.sprites()[-2]
                self.actual_pipe_bottom = self.pipes.sprites()[-1]
                self.bird.score += 1 

            # Collides 
            for bird in birds_gen:
                if bird.rect.bottom > YLIM:
                    scores.append(bird.score)
                    bird.kill()
            for bird, _ in pygame.sprite.groupcollide(birds_gen, self.pipes, False, False).items():
                scores.append(bird.score)
                bird.kill()
                

            # Update frame
            self.update(False, True) # die , game_active

        return scores