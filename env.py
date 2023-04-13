from game import ImagesBG, Bird
from genetic_algorith import Genetic_Algorith
import torch as T

import pygame
import random
import time
import os


class Enviroment:

	def __init__(self):
		
		pygame.init()

		self.n_samples = 1

		self.bird_color = "yellow" # blue / red / yellow
		self.birds_sample = [Bird(self) for _ in range(self.n_samples)]
		self.bird = self.birds_sample[0]

		self.gen_algorith = Genetic_Algorith()

		window_size = (400, 700)
		self.window = pygame.display.set_mode(window_size)
		
		type_bg = "day"       # day / night
		self.img = ImagesBG(type_bg, window_size)
		self.floor_rect = pygame.Rect((0, 620, 400, 110)) 
		self.floor_x = 0

		self.pipe_heights = [250, 350, 450, 550]
		self.pipe_list = []
		self.actual_pipe = 0
		self.reset_pipes()

	def reset_pipes(self):
		self.pipe_list = []#[self.create_pipe(self.img.pipe)]
		self.actual_pipe = 0

	def create_pipe(self, img_pipe):
		random_height = random.choice(self.pipe_heights)
		top_pipe = img_pipe.get_rect(midtop = (450, random_height))
		bottom_pipe = img_pipe.get_rect(midbottom = (450, random_height - 200))
		return (top_pipe, bottom_pipe) 

	def move_pipes(self):
		for i, (top_pipe, bottom_pipe) in enumerate(self.pipe_list):
			top_pipe.x -= 2
			bottom_pipe.x -= 2
			if i == self.actual_pipe and top_pipe.x < self.bird.surface.x: #########################  A LO MEJOR HAY QUE SUMAR LA ANCHURA DEL PIPE
				self.actual_pipe += 1
		if len(self.pipe_list) > 0 and self.pipe_list[0][0].x < -self.img.size_pipe[0]:
			del self.pipe_list[0]
			# update the indicator of the actual pipe
			if self.actual_pipe != 0:
				self.actual_pipe -= 1 

	def draw_floor(self):
		self.window.blit(self.img.floor, (self.floor_x, 600))
		self.window.blit(self.img.floor, (self.floor_x + 400, 600))

	def draw_bird(self, all=False):
		if all:
			for bird in self.birds_sample:
				self.window.blit(bird.img, bird.surface)
				break
		else:
			self.window.blit(self.bird.img, self.bird.surface)

	def draw_pipes(self):
		for (t_pipe, b_pipe) in self.pipe_list:
			pipe_fliped = pygame.transform.flip(self.img.pipe, False, True)
			self.window.blit(pipe_fliped, b_pipe)
			self.window.blit(self.img.pipe, t_pipe)

	def draw_score(self):
		if self.bird.score - 1 < 10:
			i = self.bird.score
			if self.bird.score == 0: i = 1
			self.window.blit(self.img.numbers[i-1], (200-15, 50))
		elif self.bird.score - 1 < 100:
			number = self.bird.score - 1
			i1 = str(number)[0]
			i2 = str(number)[1]
			self.window.blit(self.img.numbers[int(i1)], (200-33, 50))
			self.window.blit(self.img.numbers[int(i2)], (200+3, 50))
		elif self.bird.score - 1 < 1000:
			number = self.bird.score - 1
			i1 = str(number)[0]
			i2 = str(number)[1]
			i3 = str(number)[2]
			self.window.blit(self.img.numbers[int(i1)], (200-60, 50))
			self.window.blit(self.img.numbers[int(i2)], (200-20, 50))
			self.window.blit(self.img.numbers[int(i3)], (200+20, 50))

	def update(self, die, game_active):
		self.window.blit(self.img.bg_window, (0,0))
		
		if game_active:
			# Bird
			self.bird.movement += self.bird.gravity
			self.bird.surface.y += self.bird.movement 
			if self.bird.movement < 0:
				self.bird.img = self.bird.images.bird_up
			elif self.bird.movement == 0:
				self.bird.img = self.bird.images.bird_mid
			else:
				self.bird.img = self.bird.images.bird_down
			# Pipes
			if not die: self.move_pipes()
			self.draw_pipes()
			# Floor
			self.floor_x -= 1
			if self.floor_x == -400: self.floor_x = 0
		else:
			self.window.blit(self.img.mesage, (50, 50))

		self.draw_bird()
		self.draw_floor()
		if die: self.window.blit(self.img.game_over, (50,270))
		if game_active: self.draw_score()
		pygame.display.update()

	def updateAI(self, die, game_active):
		self.window.blit(self.img.bg_window, (0,0))
		
		if game_active:
			# Birds
			for bird in self.birds_sample:
				bird.movement += bird.gravity
				bird.surface.y += bird.movement 
				if bird.movement < 0:
					bird.img = bird.images.bird_up
				elif bird.movement == 0:
					bird.img = bird.images.bird_mid
				else:
					bird.img = bird.images.bird_down
			# Pipes
			if not die: self.move_pipes()
			self.draw_pipes()
			# Floor
			self.floor_x -= 1
			if self.floor_x == -400: self.floor_x = 0
		else:
			self.window.blit(self.img.mesage, (50, 50))

		self.draw_bird(all=True)
		self.draw_floor()
		if die: self.window.blit(self.img.game_over, (50,270))
		if game_active: self.draw_score()
		pygame.display.update()

	def play(self):
		clock = pygame.time.Clock()
		die = False
		run = True
		game_active = False
		SPAWNPIPE = True # pygame.USEREVENT
		# pygame.time.set_timer(SPAWNPIPE, 1200)
		self.reset_pipes()

		i = 0
		while run:
			i += 1

			if len(self.pipe_list) == 0:
				SPAWNPIPE = True
	
			clock.tick(120)

			for event in pygame.event.get():
				if event.type == pygame.QUIT: 
					run = False
					pygame.quit()
				if not game_active:
					if event.type == pygame.KEYDOWN:
						if event.key  == pygame.K_SPACE:
							game_active = True
				elif game_active and not die:
					if event.type == pygame.KEYDOWN:
						if event.key  == pygame.K_SPACE:
							self.bird.movement = 0
							if self.bird.surface.top > 0: self.bird.movement -= 5 
					# if event.type == SPAWNPIPE:
					# 	new_pair_pipe = self.create_pipe(self.img.pipe)
					# 	self.pipe_list.append(new_pair_pipe)
					# 	self.bird.score += 1
				elif die:
					if event.type == pygame.KEYDOWN:
						if event.key  == pygame.K_SPACE:
							self.pipe_list.clear()
							self.bird.surface.center = (120, 350)
							game_active = False
							die = False
							self.bird.gravity = 0.25
							self.bird.score = 0
							self.img.bg_window = self.img.bg_day

			# Create new pipes
			if SPAWNPIPE:
				SPAWNPIPE = False
				new_pair_pipe = self.create_pipe(self.img.pipe)
				self.pipe_list.append(new_pair_pipe)
				self.bird.score += 1 
			elif len(self.pipe_list) > 0 and self.pipe_list[-1][0].x < 450-300:
				SPAWNPIPE = True


			# Collides 
			if self.bird.surface.colliderect(self.floor_rect):
				self.bird.movement = 0
				self.bird.gravity = 0
				die = True
			for (t_pipe, b_pipe) in self.pipe_list:
				if self.bird.surface.colliderect(t_pipe) or self.bird.surface.colliderect(b_pipe):
					die = True

			# Update frame
			self.update(die, game_active)

	def playAI_Gen(self, n_epochs, n_simulations):
		self.birds_gen = [Bird(self) for _ in range(n_simulations)]

		for g in range(n_epochs):

			print("\nGeneration:", g+1)

			b = 0
			for bird in self.birds_gen:
				b += 1
				self.bird = bird
				self.birds_sample = [bird]
				bird.t = time.time()
				self.playAI()
				print("Bird:", b, "(Score:", self.gen_algorith.scores[-1], ")")
			
			next_gen = self.gen_algorith.create_next_gen()
			for bird, d in zip(self.birds_gen, next_gen):
				bird.model.load_state_dict(d)

	def playAI(self):
		clock = pygame.time.Clock()
		die = False
		run = True
		game_active = True
		SPAWNPIPE = pygame.USEREVENT
		pygame.time.set_timer(SPAWNPIPE, 1200)
		self.reset_pipes()

		while run:

			clock.tick(120)

			if not die:
				#self.agent.save_frame_enviroment()
				for bird in self.birds_sample:
					actual_pipe = self.pipe_list[self.actual_pipe]
					# except:
					# 	print(self.pipe_list)
					# 	print(self.actual_pipe)
					# 	exit(0)
					d1 = actual_pipe[0].center[0] - bird.surface.center[0]  # horizontal
					d2 = (actual_pipe[0].bottom + actual_pipe[1].top) / 2 - bird.surface.center[1]
					x = T.tensor([bird.movement, d1, d2], dtype=T.float)
					move = bird.play(x)
					if move == 1:  
						bird.movement = 0
						if bird.surface.top > 0: bird.movement -= 5 

			for event in pygame.event.get():
				if event.type == pygame.QUIT: 
					run = False
					pygame.quit()
				if not die:
					if event.type == SPAWNPIPE:
						new_pair_pipe = self.create_pipe(self.img.pipe)
						self.pipe_list.append(new_pair_pipe)
						for bird in self.birds_sample:
							bird.score += 1
				else:
					self.pipe_list.clear()
					die = False
					self.img.bg_window = self.img.bg_day
					return 

			# Collides 
			e = 0  # number of die birds in this frame
			for i in range(len(self.birds_sample)):
				bird = self.birds_sample[i-e]
				if bird.surface.colliderect(self.floor_rect):
					self.gen_algorith.save_results(bird)
					del self.birds_sample[i-e]
					e += 1
					# self.bird.movement = 0
					# self.bird.gravity = 0
					# die = True
				else:
					for (t_pipe, b_pipe) in self.pipe_list:
						if bird.surface.colliderect(t_pipe) or bird.surface.colliderect(b_pipe):
							self.gen_algorith.save_results(bird)
							del self.birds_sample[i-e]
							e += 1
							break
			
			die = self.birds_sample == []

			# Update self.window
			self.updateAI(die, game_active)
