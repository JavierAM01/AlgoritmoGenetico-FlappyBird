import torch as T
from torch import nn
from torch.nn import functional as F
from torch import optim


class Net_FlappyBird(nn.Module):

    def __init__(self, input_size):
        super(Net_FlappyBird, self).__init__()

        # architecture 
        self.l1 = nn.Linear(input_size, 32)
        self.l2 = nn.Linear(32, 1)
        # self.l3 = nn.Linear(64, 32)
        # self.l4 = nn.Linear(32, 1)

        # init
        self.l1.weight.data = T.randn((32,input_size))
        self.l2.weight.data = T.randn((1, 32))
        # self.l3.weight.data = T.randn(0,1, size=(32,64)).float()
        # self.l4.weight.data = T.randn(0,1, size=(1,32)).float()

        # device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):

        x = x.to(self.device)
        
        for layer in [self.l1, self.l2]: # self.l3, self.l4]:
            x = F.relu(layer(x))
        
        return T.sigmoid(x)

    def save_model(self, path):
        T.save(self.state_dict, f'models/{path}.pkl')