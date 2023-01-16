import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):  # Prediction function: Predicts the Q values for each action
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name="model.pth"):
        model_folder_path = "./model"
        # Create folder if it doesn't exist
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        # Generate save file path
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name="model.pth"):
        model_folder_path = "./model"
        file_name = os.path.join(model_folder_path, file_name)
        self.load_state_dict(torch.load(file_name))


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        # Optimization algorithm
        self.optimizer = optim.Adam(
            model.parameters(), lr=self.lr
        )  # Adam optimizer; adapts learning rate for each parameter based on historical gradients
        # Error/loss function
        self.criterion = (
            nn.MSELoss()
        )  # Mean Squared Error loss; used for regression; goal is to predict a continuous value

    def train_step(
        self, state, action, reward, next_state, done
    ):  # State: list of 11 values, action: 0, 1, or 2, reward: int, next_state: list of 11 values, done: boolean
        # Convert to tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:  # 1D tensor
            state = torch.unsqueeze(
                state, 0
            )  # Add a dimension of size 1 at the beginning
            next_state = torch.unsqueeze(next_state, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: Predict Q values for current state
        pred = self.model(state)

        target = pred.clone()  # Clone pred tensor
        for idx in range(len(done)):  # Iterate through each sample in the batch
            Q_new = reward[idx]  # If done, Q_new = reward
            if not done[idx]:  # If this is not the last step
                Q_new = reward[idx] + self.gamma * torch.max(
                    self.model(next_state[idx])
                )  # Calculate Q_new

            target[idx][torch.argmax(action).item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> Only continue if not done
        # pred.clone() creates a copy of pred
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()  # Reset gradients
        loss = self.criterion(
            target, pred
        )  # Calculate loss between target and pred using MSE
        loss.backward()  # Calculate gradients

        self.optimizer.step()  # Update weights
