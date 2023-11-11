import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# Assuming the histogram data has shape (num_samples, channels, height, width)
# If not, you will need to reshape the data accordingly.

class ConvBlock(nn.Module):
    """
    A 2D convolution, followed by batchnorm, a ReLU activation, and dropout.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return self.dropout(x)


class ConvNet(nn.Module):
    """
    A CNN for processing histogram data.
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.layer1 = ConvBlock(in_channels, 64, kernel_size=3, stride=1, dropout=0.5)
        self.layer2 = ConvBlock(64, 128, kernel_size=3, stride=1, dropout=0.5)
        # Add more layers as needed
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def regression_accuracy(output, target, threshold=0.05):
    """
    Calculate the 'accuracy' for a regression task. A prediction is considered
    correct if it's within a specified threshold of the actual value.
    
    Parameters:
    output (torch.Tensor): The model's predictions.
    target (torch.Tensor): The actual values.
    threshold (float): The acceptable relative error for a prediction to be considered correct.

    Returns:
    float: The 'accuracy' of the predictions.
    """
    correct = torch.abs((output - target) / target) < threshold
    accuracy = torch.mean(correct.float()) * 100
    return accuracy.item()

def l1_l2_loss(output, target, l1_weight=0.5):
    mse_loss = nn.MSELoss()(output, target)
    l1_loss = nn.L1Loss()(output, target)
    return l1_weight * l1_loss + (1 - l1_weight) * mse_loss

# Define training function
def train_model(model, train_loader, val_loader, epochs, learning_rate, weight_decay, patience, device):
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer =torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.99,weight_decay=weight_decay)
    best_val_loss = np.inf
    best_val_accuracy = 0
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    best_epoch_info = None
    patience_counter = 0
    #criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        total_loss, total_train_accuracy = 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = l1_l2_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_train_accuracy += regression_accuracy(outputs, targets)

        avg_train_loss = total_loss / len(train_loader)
        avg_train_accuracy = total_train_accuracy / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss, total_val_accuracy = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss = l1_l2_loss(outputs, targets)
                total_val_loss += val_loss.item()
                total_val_accuracy += regression_accuracy(outputs, targets)

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_accuracy = total_val_accuracy / len(val_loader)

        # Update best validation accuracy and save model state
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), Path("model.pth"))
            patience_counter = 0
        else:
            #print("best validation loss is: ", best_val_loss)
            patience_counter += 1
            if patience_counter > patience:
                print("Early stopping")
                break
        if avg_val_accuracy > best_val_accuracy:
            best_val_accuracy = avg_val_accuracy
            best_epoch_info = epoch + 1
        # Append losses and accuracies for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(avg_train_accuracy)
        val_accuracies.append(avg_val_accuracy)
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.2f}%')
        print(f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy:.2f}%')
    print(f"Training completed. Best validation accuracy: {best_val_accuracy:.2f}% at Epoch {best_epoch_info}")

    # Plot the training and validation loss
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(bottom = 0, top = 5000)
    plt.title('Loss Curve')
    plt.legend()

    # Plot the training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.savefig("training_performance.png")

    return train_losses, val_losses, train_accuracies, val_accuracies

# Load data
data = np.load('data/img_output/histogram_all_full.npz')
images = data['output_image']
yields = data['output_yield']
years = data['output_year']

# Normalize images
images = (images - images.mean(axis=(0, 2, 3), keepdims=True))

# Split the data into training and validation sets
train_indices = (years >= 2018) & (years <= 2021)
val_indices = years == 2022
train_images, val_images = images[train_indices], images[val_indices]
train_yields, val_yields = yields[train_indices], yields[val_indices]

# Convert to PyTorch tensors
train_images_tensor = torch.tensor(train_images, dtype=torch.float32)
val_images_tensor = torch.tensor(val_images, dtype=torch.float32)
train_yields_tensor = torch.tensor(train_yields, dtype=torch.float32).view(-1, 1)
val_yields_tensor = torch.tensor(val_yields, dtype=torch.float32).view(-1, 1)

# Create DataLoaders
train_dataset = TensorDataset(train_images_tensor, train_yields_tensor)
val_dataset = TensorDataset(val_images_tensor, val_yields_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNet(in_channels=12, num_classes=1)  # Adjust in_channels according to your data
model.to(device)

# Train the model
train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=500,  # Set the number of epochs
    learning_rate=1e-3,  # Set the learning rate
    weight_decay=5e-4,  # Set the weight decay
    patience=50,  # Set the patience for early stopping
    device=device
)