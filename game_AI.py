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

    def load_model(self, filename):
        self.brain.load_state_dict(T.load(filename))


class Game_AI(Game):

    def __init__(self):
        super().__init__()

    def train(self, n_generations, n_birds, limit_score=50, draw=True):

        gen_model = Genetic_Model()
        
        next_gen = None
        
        print("SCORES:")
        for g in range(n_generations):

            for i in range(n_birds):
                bird = Bird_AI()
                if next_gen != None:
                    bird.brain.load_state_dict(next_gen[i])
                self.birds.add(bird)
            
            self.play_generation(gen_model, limit_score, draw)
            
            print(f"\nGen [{g+1}]")
            for i, s in enumerate(gen_model.scores):
                print(f"\t Bird {i}: {s}")

            # first we play 10 generations randomly (to ensure that we search enough) 
            # and then we begin the genetic algorithm with the best results of the past
            if g > min(10, n_generations // 2):
                next_gen = gen_model.create_next_gen()

            gen_model.reset()

    def play_generation(self, gen_model, limit_score=50, draw=True):

        if draw: 
            pygame.init()
            self.window = pygame.display.set_mode(SIZE_window)
        
        clock = pygame.time.Clock()
        SPAWNPIPE = False
        self.reset_pipes()

        score = 0

        i = 0
        while len(self.birds) > 0:
            i += 1

            # avoid having (almost perfect) birds who never dies
            if score == limit_score:
                for bird in self.birds:
                    gen_model.save_results(bird)
                    bird.kill()
    
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
            self.update(score, False, True, draw) # die , game_active

        if draw: pygame.quit()

    def play_AI(self, particular_bird=None):

        bird = Bird() if particular_bird == None else particular_bird
        self.birds.add(bird)

        pygame.init()
        self.window = pygame.display.set_mode(SIZE_window)

        score = 0

        clock = pygame.time.Clock()
        die = False
        run = True
        game_active = False
        SPAWNPIPE = False
        self.reset_pipes()

        i = 0
        while run:
            i += 1
    
            clock.tick(120)

            for event in pygame.event.get():
                
                # quit display
                if event.type == pygame.QUIT: 
                    run = False
                    pygame.quit()

                if not game_active:
                    # screen : before start the game -> check SPACE to begin
                    if event.type == pygame.KEYDOWN:
                        if event.key  == pygame.K_SPACE:
                            game_active = True

                elif die:
                    # bird on the floor (end game) -> check SPACE to reset
                    if event.type == pygame.KEYDOWN:
                        if event.key  == pygame.K_SPACE:
                            self.reset_pipes()
                            bird.reset()
                            game_active = False
                            die = False
                            score = 0

            # Move bird
            if not die and game_active:
                bird.move(self.actual_pipe_top, self.actual_pipe_bottom)
                        
            # Create new pipes
            if SPAWNPIPE:
                SPAWNPIPE = False
                self.add_pipes()
            elif len(self.pipes) > 0 and self.pipes.sprites()[-1].rect.x < 450-300:
                SPAWNPIPE = True

            # update actual pipe
            if self.actual_pipe_bottom.rect.right < bird.rect.centerx:
                self.actual_pipe_top = self.pipes.sprites()[-2]
                self.actual_pipe_bottom = self.pipes.sprites()[-1]
                bird.score += 1 
                score += 1

            # Collides 
            if bird.rect.bottom > YLIM:
                bird.movement = 0
                bird.gravity = 0
                die = True
            if pygame.sprite.spritecollide(bird, self.pipes, dokill=False):
                die = True

            # Update frame
            self.update(score, die, game_active)

        bird.kill()
        pygame.quit()
        bird.kill()

    def update(self, score, die, game_active, draw=True):
        if draw: self.window.blit(IMG_backgroung, (0,0))
        
        if game_active:
            # Bird
            self.birds.update()
            # Pipes
            if not die: 
                self.pipes.update()
            if draw: self.pipes.draw(self.window)
            # Floor
            self.floor.update()
        else:
            if draw: self.window.blit(IMG_mesage, (50, 50))

        if draw: 
            self.birds.draw(self.window)
            self.floor.draw(self.window)

            for bird in self.birds:
                pygame.draw.line(self.window,(0,0,0), bird.rect.center, (self.actual_pipe_top.rect.centerx, self.actual_pipe_top.rect.bottom))
                pygame.draw.line(self.window,(0,0,0), bird.rect.center, (self.actual_pipe_bottom.rect.centerx, self.actual_pipe_bottom.rect.top))
        
            if die: 
                self.window.blit(IMG_game_over, (50,270))
            if game_active: 
                self.draw_score(score)

            pygame.display.update()