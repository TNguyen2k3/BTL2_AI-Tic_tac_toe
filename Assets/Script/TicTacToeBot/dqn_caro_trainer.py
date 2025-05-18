import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import os
import numpy as np

# --- Cấu hình ---
BOARD_SIZE = 9
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE

# --- Mô hình DQN ---
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

# --- Hàm xử lý dữ liệu ---
def load_data(json_folder):
    data = []
    for file_name in os.listdir(json_folder):
        if file_name.endswith(".json"):
            with open(os.path.join(json_folder, file_name), 'r') as f:
                sample = json.load(f)
                data.append(sample)
    return data

def board_to_tensor(moves, current_index, bot_role):
    board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for i in range(current_index):
        move = moves[i]
        x, y, player = move['x'], move['y'], move['player']
        if player == bot_role:
            board[0][x][y] = 1.0  # Kênh 0: bot
        else:
            board[1][x][y] = 1.0  # Kênh 1: đối thủ
    return torch.tensor(board)

def move_to_action(x, y):
    return x * BOARD_SIZE + y

# --- Huấn luyện ---
def train(model, data, optimizer, epochs=50, gamma=0.3):
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
                    continue  # Chỉ train nước đi của bot

                state = board_to_tensor(moves, i, bot_role).unsqueeze(0)  # (1, 2, 9, 9)
                next_state = board_to_tensor(moves, i + 1, bot_role).unsqueeze(0)

                action = move_to_action(move["x"], move["y"])

                # Thiết lập phần thưởng
                reward = -0.01  # Mặc định nhỏ
                if i == len(moves) - 2:  # Nước cuối cùng
                    if result == bot_role:
                        reward = 0.3
                    elif result == 0:
                        reward = 0.0
                    else:
                        reward = -0.3

                q_values = model(state)
                q_value = q_values[0, action].unsqueeze(0)

                with torch.no_grad():
                    model.eval()
                    next_q_values = model(next_state)
                    max_next_q = torch.max(next_q_values)
                    expected_q = reward + gamma * max_next_q.detach()
                    expected_q = torch.tensor([expected_q], dtype=torch.float32)
                    model.train()

                loss = F.mse_loss(q_value, expected_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# --- Chạy ---
if __name__ == "__main__":
    model = DQN()
    optimizer = optim.Adam(model.parameters(), lr=1e-6)  # lr nhỏ cho ổn định
    data = load_data("E:/HocBachKhoa/BTL nhập môn AI/BTL2/BTL2_AI-Tic_tac_toe/Assets/LearnData")
    train(model, data, optimizer)
    torch.save(model.state_dict(), "caro_dqn_model.pth")
    print("Model saved to caro_dqn_model.pth")