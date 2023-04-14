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


class Game_AI(Game):

    def __init__(self):
        super().__init__()
        self.bird = Bird_AI()

    def move(self, event):
        self.bird.move(self.actual_pipe_top, self.actual_pipe_bottom)