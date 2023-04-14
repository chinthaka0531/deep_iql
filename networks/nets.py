import torch
import torch.nn as nn
import torch.nn.functional as F

class Phi(nn.Module):

    def __init__(self):
        super(Phi, self).__init__()
        self.fc1 = nn.Linear(3, 120)
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, 80)
        self.relu = nn.ReLU()

    def forward(self, X_dyn):
        Phi_out = torch.zeros((len(X_dyn), self.fc3.out_features))
        for i, sample in enumerate(X_dyn):
            for x_dyn in sample:
                x = self.fc1(x_dyn)
                x = self.relu(x)
                x = self.fc2(x)
                x = self.relu(x)
                x = self.fc3(x)
                x = self.relu(x)
                Phi_out[i] = Phi_out[i]+x

        return Phi_out

class Rho(nn.Module):

    def __init__(self):
        super(Rho, self).__init__()
        self.fc1 = nn.Linear(80, 60)
        self.fc2 = nn.Linear(60, 60)
        self.fc3 = nn.Linear(60, 40)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

class Q(nn.Module):

    def __init__(self):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(43, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x
    
class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.phi = Phi()
        self.rho = Rho()
        self.Q = Q()

    def forward(self, dyn, static):
        static= torch.Tensor(static)
        x = self.phi(dyn)
        x = self.rho(x)
        x = torch.cat((x, static), dim=1)
        x = self.Q(x)

        return x