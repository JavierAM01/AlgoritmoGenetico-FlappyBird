import random
import numpy as np
import torch as T
from model import Net_FlappyBird as Net


class Genetic_Model:

    def __init__(self):
        self.parameters      = []
        self.scores          = []
        self.best_parameters = []
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.actual_best_score = 0

        # about the network
        net = Net(input_size=5)
        dic = net.state_dict()
        self.params = dict()
        for key, param in dic.items():
            self.params[key] = tuple(param.shape)
        self.keys = list(self.params.keys())
        self.weights = [np.prod(shape) for shape in self.params.values()]

    def get_best_parameters(self):
        # sort by scores
        self.best_parameters = sorted(self.best_parameters, key = lambda x : x[0], reverse=True) 
        return self.best_parameters[0][1]

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

        # save best result of the generetion

        i = np.argmax(self.scores)
        if len(self.best_parameters) == 0 or self.best_parameters[-1][0] < self.scores[i]:
            self.best_parameters.append((self.scores[i], self.parameters[i]))

        # find best actual 20% parameters

        for _ in range(n20):
            i = np.argmax(self.scores)
            best = self.parameters[i]
            score = self.scores[i]
            self.best_parameters.append((score, best))
            next_gen.append(best)
            del self.parameters[i]
            del self.scores[i]

        # create the rest 50% parameters by mutating 

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
        if self.actual_best_score < self.best_parameters[0][0]:
            self.actual_best_score = self.best_parameters[0][0]
            T.save(self.best_parameters[0][1], f'models/best_{self.best_parameters[0][0]}.pkl')

        while len(self.best_parameters) > n20: 
            del self.best_parameters[n20]

        # reset generation
        self.reset()
        
        return next_gen

    def f(self, p1, p2):
        
        option = random.randint(0,1)
        
        net = Net(input_size=5)
        net.load_state_dict(p1)
        params = net.state_dict()
        net2 = Net(input_size=5)
        net2.load_state_dict(p2)
        params2 = net2.state_dict()

        # chose part to change

        key = random.choices(self.keys, weights=self.weights, k=1)[0]
        index = tuple([random.randint(0,x-1) for x in list(self.params[key])])

        def change(params, new_params):
            if len(index) == 1:
                a = index[0]
                params[key][a] = new_params[key][a]
            elif len(index) == 2:
                a, b = index[0], index[1]
                params[key][a,b] = new_params[key][a,b]
            elif len(index) == 3:
                a, b, c = index[0], index[1], index[2]
                params[key][a,b,c] = new_params[key][a,b,c]

        def change2(params):
            if len(index) == 1:
                a = index[0]
                params[key][a]      *= (1 + (0.01 * (-1)**random.randint(0,1)))
            elif len(index) == 2:
                a, b = index[0], index[1]
                params[key][a,b]    *= (1 + (0.01 * (-1)**random.randint(0,1)))
            elif len(index) == 3:
                a, b, c = index[0], index[1], index[2]
                params[key][a,b,c]  *= (1 + (0.01 * (-1)**random.randint(0,1)))

        # chose diferent parts of both params: p1 & p2
        if option == 0: 
            change(params, p1)
            change(params2, p2)

        elif option == 1:
            change2(params)
            change2(params2)
        
        # # change a little the actual param: p1
        # else: 
        #     for key in self.params:
        #         # crate a matrix of 1.01 or 0.99 (randomly) and multiply to the actual values
        #         c1 = [1 + (0.01 * (-1)**np.random.randint(2)) for _ in range(np.product(params[key].shape))]
        #         c2 = [1 + (0.01 * (-1)**np.random.randint(2)) for _ in range(np.product(params[key].shape))]
        #         c1 = T.tensor(c1, dtype=T.float).reshape(params[key].shape).to(self.device)
        #         c2 = T.tensor(c2, dtype=T.float).reshape(params[key].shape).to(self.device)
        #         params[key] = c1 * p1[key]
        #         params2[key] = c2 * p2[key]

        return params, params2
    
    def reset(self):
        self.parameters.clear()
        self.scores.clear()
