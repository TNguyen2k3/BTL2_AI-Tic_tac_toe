import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import os
import random
import numpy as np
import torch.onnx

BOARD_SIZE = 9
STATE_SIZE = BOARD_SIZE * BOARD_SIZE
ACTION_SIZE = STATE_SIZE

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(STATE_SIZE, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, ACTION_SIZE)

    def forward(self, x):
        x = x.view(-1, STATE_SIZE)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
model = DQN()
model.load_state_dict(torch.load("caro_dqn_model.pth"))
model.eval()

dummy_input = torch.randn(1, STATE_SIZE)  # input giả để trace model

torch.onnx.export(model, dummy_input, "caro_dqn_model.onnx",
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                  opset_version=11)
