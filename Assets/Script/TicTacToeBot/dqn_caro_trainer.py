import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import os
import random
import numpy as np

BOARD_SIZE = 9
STATE_SIZE = BOARD_SIZE * BOARD_SIZE
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE

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

def load_data(json_folder):
    data = []
    for file_name in os.listdir(json_folder):
        if file_name.endswith(".json"):
            with open(os.path.join(json_folder, file_name), 'r') as f:
                sample = json.load(f)
                data.append(sample)
    return data

def board_to_tensor(moves, current_index):
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for i in range(current_index):
        move = moves[i]
        board[move['x']][move['y']] = move['player']
    return torch.tensor(board).flatten()

def move_to_action(x, y):
    return x * BOARD_SIZE + y

def train(model, data, optimizer, epochs=50, gamma=0.99):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for game in data:
            moves = game["moves"]
            result = game["gameResult"]
            bot_role = game["botRole"]
            
            for i in range(len(moves) - 1):
                move = moves[i]
                if move["player"] != bot_role:
                    continue  # chỉ train cho bot đã chơi

                state = board_to_tensor(moves, i)
                next_state = board_to_tensor(moves, i + 1)
                action = move_to_action(move["x"], move["y"])

                reward = -0.01
                if i == len(moves) - 2:
                    reward = 0.3 if result == bot_role else (-0.3 if result != 0 else 0)
                q_values = model(state)
                q_value = q_values[0, action].unsqueeze(0)

                with torch.no_grad():
                    model.eval()
                    next_q_values = model(next_state)
                    max_next_q = torch.max(next_q_values)
                    expected_q = reward + gamma * max_next_q.detach()
                    expected_q = torch.tensor([expected_q], dtype=torch.float32)

                loss = F.mse_loss(q_value, expected_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
        print(board_to_tensor(moves, i).view(9, 9))
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

if __name__ == "__main__":
    model = DQN()
    optimizer = optim.Adam(model.parameters(), lr=0.0000001)
    data = load_data("E:/HocBachKhoa/BTL nhập môn AI/BTL2/BTL2_AI-Tic_tac_toe/Assets/LearnData")
    train(model, data, optimizer)

    torch.save(model.state_dict(), "caro_dqn_model.pth")
    print("Model saved to caro_dqn_model.pth")
