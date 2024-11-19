import torch
import torch.nn as nn
import torch.optim as optim

# Define a deeper neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(5, 10)  # Increase the number of neurons
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(10, 8)  # Add more layers with decreasing neurons
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(8, 5)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

# Instantiate the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Hardcoded input and target
inputs = torch.tensor([[0.5, -0.2, 0.1, 0.7, -0.3]], dtype=torch.float32)
target = torch.tensor([[1.0]], dtype=torch.float32)

# Training for one epoch
outputs = model(inputs)
loss = criterion(outputs, target)
optimizer.zero_grad()
loss.backward()
optimizer.step()

# print the gradient of the first layer
print(model.fc1.weight.grad)

# Print the output and target for verification
print("Deep model output:", outputs)
print("Target:", target)
