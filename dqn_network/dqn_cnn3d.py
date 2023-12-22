import torch
import torch.nn as nn   
import torch.nn.functional as F 

class CNN3D(nn.Module):
    def __init__(self, map_size: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        action_size = 26
        self.fc = nn.Linear(32*(map_size**3), action_size)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.flatten(x)
        x = self.fc(x)  
        
        return x