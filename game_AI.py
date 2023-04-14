import pygame, random, os
import torch as T # we only use : T.tensor

from model import Net_FlappyBird
from genetic_algorith import Genetic_Model

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

    def train(self, n_generations, n_birds):

        gen_model = Genetic_Model()
        
        next_gen = None
        
        print("SCORES:")
        for g in range(n_generations):

            for i in range(n_birds):
                bird = Bird_AI()
                if next_gen != None:
                    bird.brain.load_state_dict(next_gen[i])
                self.birds.add(bird)
            
            self.play_generation(gen_model)
            
            print(f"\nGen [{g+1}]")
            for i, s in enumerate(gen_model.scores):
                print(f"\t Bird {i}: {s}")

            # first we play 5 generations randomly (to ensure that we search enough) 
            # and then we begin the genetic algorithm with the best results of the past
            if g > 5:
                next_gen = gen_model.create_next_gen()

            gen_model.reset()

        n = len(os.listdir("models"))
        T.save(gen_model.get_best_parameters(), f"models/best_brain_{n}.pkl")



    def play_generation(self, gen_model):

        pygame.init()
        self.window = pygame.display.set_mode(SIZE_window)

        score = 0

        clock = pygame.time.Clock()
        SPAWNPIPE = False
        self.reset_pipes()

        i = 0
        while len(self.birds) > 0:
            i += 1
    
            clock.tick(120)
            
            for bird in self.birds:
                bird.move(self.actual_pipe_top, self.actual_pipe_bottom)
                        
            # Create new pipes
            if SPAWNPIPE:
                SPAWNPIPE = False
                self.add_pipes()
            elif len(self.pipes) > 0 and self.pipes.sprites()[-1].rect.x < 450-300:
                SPAWNPIPE = True

            # update actual pipe
            if self.actual_pipe_bottom.rect.right < 120: # bird centerx
                self.actual_pipe_top = self.pipes.sprites()[-2]
                self.actual_pipe_bottom = self.pipes.sprites()[-1]
                for bird in self.birds: bird.score += 1
                score += 1 

            # Collides 
            for bird in self.birds:
                if bird.rect.bottom > YLIM:
                    gen_model.save_results(bird)
                    bird.kill()
            for bird, _ in pygame.sprite.groupcollide(self.birds, self.pipes, False, False).items():
                gen_model.save_results(bird)
                bird.kill()
                

            # Update frame
            self.update(score, False, True) # die , game_active

        pygame.quit()