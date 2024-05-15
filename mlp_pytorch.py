import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import numpy as np

seed = 1
torch.manual_seed(seed)

if torch.cuda.is_available():
    print('Using GPU, device name:', torch.cuda.get_device_name(0))
    device = torch.device('cuda')
else:
    print('Using CPU')
    device = torch.device('cpu')

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
# If we want to normalize data to have zero mean and 1 std, use the following. But we do not do that in barenet.
#train_dataset = datasets.MNIST('./data', train=True, download=True, 
                              #transform=transforms.Normalize((0.1307,), (0.3081,)))
test_dataset = datasets.MNIST('./data', train=False, transform=transforms.ToTensor())

alltrain_loader = DataLoader(dataset=train_dataset, batch_size=train_dataset.data.size()[0], shuffle=False)
for (data, _) in alltrain_loader:    
    print(f'mnist dataset mean={torch.mean(data)}')


batch_size = 32
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    
class TwoLayerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
        )
        self.l1 = nn.Linear(28*28, 16)
        #print(f'l1 init weights={self.l1.weight}')
        self.af = nn.ReLU()
        self.l2 = nn.Linear(16, 10)

    def forward(self, x):
        x0 = torch.flatten(x, start_dim=1)
        x1 = self.l1(x0)
        #print(f'x1={x1} weights={self.l1.weight} bias={self.l1.bias}')
        x1_a = self.af(x1)
        x2 = self.l2(x1_a)
        return x2

model = TwoLayerMLP().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# this is the same as the correct(...) function in barenet's train_mlp.cu
# it calculates the number of correct predictions based on the logit values in output
def correct(output, target):
    predicted_digits = output.argmax(1)                            
    correct_ones = (predicted_digits == target).type(torch.float)  
    return correct_ones.sum().item()                               

def train(data_loader, model, criterion, optimizer):
    model.train()

    num_batches = len(data_loader)
    num_items = len(data_loader.dataset)

    total_loss = 0
    total_correct = 0
    for data, target in data_loader:
        # Copy data and targets to GPU
        data = data.to(device)
        target = target.to(device)

        # Do a forward pass
        output = model(data)
        #print(f'output={output}')
        # Calculate the loss
        loss = criterion(output, target)
        #print(f'loss={loss}')

        total_loss += loss

        # Count number of correct digits
        total_correct += correct(output, target)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_loss = total_loss/num_batches
    accuracy = total_correct/num_items
    print(f"Training loss: {train_loss:7f}, accuracy: {accuracy:.2%} num_batches: {num_batches}")

epochs = 50
start_time = time.time()  # Record start time
for epoch in range(epochs):
    print(f"Training epoch: {epoch+1}")
    train(train_loader, model, criterion, optimizer)
end_time = time.time()  # Record end time
training_time = end_time - start_time
print(f"Total training time: {training_time:.2f} seconds")

def test(test_loader, model, criterion):
    model.eval()

    num_batches = len(test_loader)
    num_items = len(test_loader.dataset)

    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        #test_loader is constructed such that there is only one batch
        for data, target in test_loader: 
            # Copy data and targets to GPU
            data = data.to(device)
            target = target.to(device)

            # Do a forward pass
            output = model(data)

            # Calculate the loss
            loss = criterion(output, target)
            total_loss += loss.item()

            # Count number of correct digits
            total_correct += correct(output, target)
        test_loss = total_loss / num_batches
        accuracy = total_correct/num_items

        print(f"Test loss: {test_loss:>7f} accuracy: {100*accuracy:>0.1f}%")

test(test_loader, model, criterion)