import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(output_size, output_size)
        self.relu1 = nn.ReLU()   

    def forward(self, x):
        x = self.relu1(self.fc(x))
        x = self.fc2(x)
        return x