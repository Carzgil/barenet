import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Set the random seed for reproducibility
seed = 1
torch.manual_seed(seed)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print('Using GPU, device name:', torch.cuda.get_device_name(0))
else:
    print('Using CPU')

# Load datasets
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST('./data', train=False, transform=transforms.ToTensor())

# Data loaders
batch_size = 32
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define the neural network model
class TwoLayerMLP(nn.Module):
    def __init__(self):
        super(TwoLayerMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(28*28, 16)
        self.af = nn.ReLU()
        self.l2 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.l1(x)
        x = self.af(x)
        x = self.l2(x)
        return x

# Instantiate the model
model = TwoLayerMLP().to(device)
criterion = nn.CrossEntropyLoss()

# Function to calculate accuracy
def correct(output, target):
    predicted_digits = output.argmax(1)
    correct_ones = (predicted_digits == target).type(torch.float)
    return correct_ones.sum().item()

# Manually calculate gradients
def compute_gradients(data, target, model, criterion):
    output = model(data)
    loss = criterion(output, target)
    loss.backward()  # Compute gradients
    gradients = []
    for param in model.parameters():
        gradients.append(param.grad.clone())
        param.grad = None  # Clear gradients
    return loss.item(), gradients

# Update model parameters manually
def update_parameters(model, gradients, learning_rate):
    with torch.no_grad():
        for param, grad in zip(model.parameters(), gradients):
            param -= learning_rate * grad

# Training function
def train(data_loader, model, criterion, learning_rate):
    model.train()
    total_loss = 0
    total_correct = 0
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        loss, gradients = compute_gradients(data, target, model, criterion)
        update_parameters(model, gradients, learning_rate)
        total_loss += loss
        total_correct += correct(model(data), target)
        
    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / len(data_loader.dataset)
    print(f"Training loss: {avg_loss:.6f}, accuracy: {accuracy:.2%}")

# Training loop
start_time = time.time()  # Record start time
epochs = 50
learning_rate = 0.01
for epoch in range(epochs):
    print(f"Training epoch: {epoch + 1}")
    train(train_loader, model, criterion, learning_rate)
end_time = time.time()  # Record end time
training_time = end_time - start_time
print(f"Total training time: {training_time:.2f} seconds")
# Testing function
def test(data_loader, model, criterion):
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            total_correct += correct(output, target)
            
    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / len(data_loader.dataset)
    print(f"Test loss: {avg_loss:.6f}, accuracy: {accuracy:.2%}")

# Evaluate the model on the test set
test(test_loader, model, criterion)
