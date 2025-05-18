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
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * BOARD_SIZE * BOARD_SIZE, 512)
        self.fc2 = nn.Linear(512, ACTION_SIZE)

    def forward(self, x):  # x: (batch, 2, 9, 9)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
# Giả định mô hình CNN đã được định nghĩa và load như bạn đã làm
model = DQN()
model.load_state_dict(torch.load("caro_dqn_model.pth"))
model.eval()

# Dummy input đúng shape: (batch_size=1, 2 channels, 9x9 board)
dummy_input = torch.randn(1, 2, 9, 9)

# Xuất sang ONNX
torch.onnx.export(
    model,
    dummy_input,
    "caro_dqn_model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
    opset_version=11
)
