import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import classSimpleNN

# **Hyperparameters**
batch_size = 32
learning_rate = 0.01
num_epochs = 30

# **Data Preparation**
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download MNIST dataset
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Split dataset into training (50,000), validation (10,000), and test (10,000)
train_size = 50000
val_size = 10000
test_size = 10000

train_data, val_data = random_split(mnist_dataset, [train_size, val_size])
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

# **Model Initialization**
model = classSimpleNN.SimpleNN()
criterion = nn.CrossEntropyLoss()  # Loss function
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # Optimizer

# **Training Loop**
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for images, labels in train_loader:
        optimizer.zero_grad()  # Zero the gradients

        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss

        loss.backward()  # Backward pass (gradient calculation)
        optimizer.step()  # Update weights

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    # **Validation Accuracy Calculation**
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = correct / total

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# **Testing the Model**
model.eval()
correct_test = 0
total_test = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted_test = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted_test == labels).sum().item()

test_accuracy = correct_test / total_test
print(f'Test Accuracy: {test_accuracy:.4f}')