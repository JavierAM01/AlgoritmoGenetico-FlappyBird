import random
import numpy as np
import copy
import time
import torch as T
from model import Net_FlappyBird as Net


class Genetic_Model:

    def __init__(self):
        self.parameters      = []
        self.scores          = []
        self.best_parameters = []
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def get_best_parameters(self):
        bs, bp = -1, None
        for s, p in self.best_parameters:
            if s > bs:
                bp = p
        return bp

    def save_results(self, bird):
        weights = bird.brain.state_dict()
        score = bird.score
        self.parameters.append(weights)
        self.scores.append(score)

    def create_next_gen(self):

        N = len(self.parameters)
        n50 = N // 2  # 50 % 
        n20 = N // 5  # 20 % 
        n10 = N // 10 # 10 % 

        next_gen = []

        # find best last 20% parameters

        if len(self.best_parameters) > 0:
            for i in range(min(n20, len(self.best_parameters))):
                best = self.best_parameters[i][1]
                next_gen.append(best)

        # find best actual 20% parameters

        for _ in range(n20):
            i = np.argmax(self.scores)
            best = self.parameters[i]
            score = self.scores[i]
            self.best_parameters.append((score, best))
            next_gen.append(best)
            del self.parameters[i]
            del self.scores[i]

        # create the rest 50% parameters by mutating and genetics changes

        for _ in range(n50 // 2):
            w = [8]*len(next_gen) + [2]*len(self.parameters)
            p1, p2 = random.choices(next_gen + self.parameters, weights=w, k=2)
            params1, params2 = self.f(p1, p2)
            next_gen.extend([params1, params2])
            
        # create a 10% (rest) random gen

        for _ in range(N - len(next_gen)):
            net = Net(input_size=5)
            next_gen.append(net.state_dict())
        
        # save the best 20% models
        
        self.best_parameters = sorted(self.best_parameters, key = lambda x : x[0], reverse=True) # sort by scores
        T.save(self.best_parameters[0][1], f'models/best_{self.best_parameters[0][0]}.pkl')

        while len(self.best_parameters) > n20: 
            del self.best_parameters[n20]

        # reset generation
        self.reset()
        
        return next_gen

    def f(self, p1, p2):
        
        option = np.random.randint(2)
        
        net = Net(input_size=5)
        params = net.state_dict()
        net2 = Net(input_size=5)
        params2 = net2.state_dict()

        # chose diferent parts of both params: p1 & p2
        if option == 1: 
            for key in params:
                p = [p1[key], p2[key]]
                random.shuffle(p)
                params[key] = p[0]
                params2[key] = p[1]
        
        # change a little the actual param: p1
        else: #if option == 2:
            for key in params:
                # crate a matrix of 1.001 or 0.999 (randomly) and multiply to the actual values
                c1 = [1 + (0.001 * (-1)**np.random.randint(2)) for _ in range(np.product(params[key].shape))]
                c2 = [1 + (0.001 * (-1)**np.random.randint(2)) for _ in range(np.product(params[key].shape))]
                c1 = T.tensor(c1, dtype=T.float).reshape(params[key].shape).to(self.device)
                c2 = T.tensor(c2, dtype=T.float).reshape(params[key].shape).to(self.device)
                params[key] = c1 * p1[key]
                params2[key] = c2 * p2[key]
            
        # chose (one by one) diferent parts of both params: p1 & p2
        # else:
        #     for key in params:
        #         c1 = np.random.randint(0,2, np.product(params[key].shape))
        #         c2 = [1 - c for c in c1]
        #         c1 = T.tensor(c1, dtype=T.float).reshape(params[key].shape)
        #         c2 = T.tensor(c2, dtype=T.float).reshape(params[key].shape)
        #         params[key] = c1 * p1[key] + c2 * p2[key]

        return params, params2
    
    def reset(self):
        self.parameters.clear()
        self.scores.clear()
