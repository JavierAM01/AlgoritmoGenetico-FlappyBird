import torch as T
from torch import nn
from torch.nn import functional as F
from torch import optim


class Net_FlappyBird(nn.Module):

    def __init__(self, input_size, lr):
        super(Net_FlappyBird, self).__init__()

        # architecture 
        self.l1 = nn.Linear(input_size, 32)
        self.l2 = nn.Linear(32, 1)
        # self.l3 = nn.Linear(64, 32)
        # self.l4 = nn.Linear(32, 1)

        # init binary
        self.l1.weight.data = T.randn((32,input_size))
        self.l2.weight.data = T.randn((1, 32))
        # self.l3.weight.data = T.randn(0,1, size=(32,64)).float()
        # self.l4.weight.data = T.randn(0,1, size=(1,32)).float()

        # optimizer & loss
        self.optimizer   = optim.Adam(self.parameters(), lr=lr)
        self.loss_value  = nn.MSELoss()
        self.loss_policy = nn.CrossEntropyLoss()

        # device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):

        x = x.to(self.device)
        
        for layer in [self.l1, self.l2]: # self.l3, self.l4]:
            x = F.relu(layer(x))
        
        return T.sigmoid(x)

    def fit(self, X, Y, batch_size, epochs):

        n = len(X)
        losses = []
        
        for _ in range(epochs):
            for i in range(0,n,batch_size):

                # get sample
                X_batch = X[i:i+batch_size] if i+batch_size <= n else X[i:]
                Y_batch = Y[i:i+batch_size] if i+batch_size <= n else Y[i:]
                
                # get predictions
                Z_batch = self.forward(X_batch)
                
                # train
                self.optimizer.zero_grad()
                loss = self.loss(Z_batch, Y_batch).to(self.device)
                losses.append(loss)
                loss.backward()
                self.optimizer.step()

        return T.tensor(losses).tolist()

    def save_model(self, path):
        T.save(self, f'models/{path}.pkl')