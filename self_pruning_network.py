import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

class PrunableLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None
        
        # Learnable gate scores
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.zeros_(self.gate_scores)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        gate_values = torch.sigmoid(self.gate_scores)
        effective_weight = self.weight * gate_values
        output = F.linear(input, effective_weight, self.bias)
        return output

class PrunableNetwork(nn.Module):
    def __init__(self):
        super(PrunableNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = PrunableLinear(32 * 32 * 3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def compute_sparsity_loss(model: nn.Module) -> torch.Tensor:
    """
    Compute the sparsity loss for the model by summing the L1 norm of
    sigmoid-transformed gate scores in all PrunableLinear layers.
    """
    sparsity_loss = 0.0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gate_values = torch.sigmoid(module.gate_scores)
            sparsity_loss += torch.sum(torch.abs(gate_values))
    return sparsity_loss

def train_model(model: nn.Module, train_loader, lambda_val: float, device: str = "cpu"):
    """
    Train the model with classification and sparsity loss.
    """
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()

    for epoch in range(10):  # Train for 10 epochs
        model.train()
        total_loss, classification_loss, sparsity_loss = 0.0, 0.0, 0.0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            class_loss = criterion(outputs, targets)
            sparse_loss = compute_sparsity_loss(model)
            loss = class_loss + lambda_val * sparse_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track losses
            total_loss += loss.item()
            classification_loss += class_loss.item()
            sparsity_loss += sparse_loss.item()

        print(f"Epoch {epoch+1}: Total Loss = {total_loss:.4f}, Classification Loss = {classification_loss:.4f}, Sparsity Loss = {sparsity_loss:.4f}")

def evaluate(model: nn.Module, dataloader, device: str = "cpu") -> float:
    """
    Evaluate the model on the test dataset and return accuracy.
    """
    model.to(device)
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy

def compute_sparsity(model: nn.Module, threshold: float = 1e-2) -> float:
    """
    Compute the sparsity of the model as the percentage of gate values below the threshold.
    """
    total_gates, pruned_gates = 0, 0

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gate_values = torch.sigmoid(module.gate_scores)
            total_gates += gate_values.numel()
            pruned_gates += (gate_values < threshold).sum().item()

    sparsity = 100 * pruned_gates / total_gates
    print(f"Sparsity: {sparsity:.2f}%")
    return sparsity

def get_data_loaders(batch_size: int = 64):
    """
    Load CIFAR-10 dataset and return DataLoader for training and testing.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def run_experiments():
    """
    Train and evaluate the model for different lambda values.
    """
    lambda_values = [0.0001, 0.001, 0.01]
    results = []

    train_loader, test_loader = get_data_loaders()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for lambda_val in lambda_values:
        print(f"\nTraining with lambda = {lambda_val}")
        model = PrunableNetwork()
        train_model(model, train_loader, lambda_val, device)

        print("Evaluating model...")
        accuracy = evaluate(model, test_loader, device)
        sparsity = compute_sparsity(model)

        results.append((lambda_val, accuracy, sparsity))

    print("\nResults:")
    print("Lambda | Accuracy (%) | Sparsity (%)")
    for lambda_val, accuracy, sparsity in results:
        print(f"{lambda_val:.4f} | {accuracy:.2f} | {sparsity:.2f}")

    return results

def plot_gate_distribution(model):
    """
    Plot the histogram of gate values for the given model.
    """
    gate_values = []
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gate_values.append(torch.sigmoid(module.gate_scores).detach().cpu().numpy().flatten())

    gate_values = np.concatenate(gate_values)
    plt.hist(gate_values, bins=50, alpha=0.75, color="blue", edgecolor="black")
    plt.xlabel("Gate Values")
    plt.ylabel("Frequency")
    plt.title("Histogram of Gate Values")
    plt.show()

def plot_sparsity_vs_accuracy(results):
    """
    Plot sparsity vs. accuracy for different lambda values.
    """
    lambdas = [result[0] for result in results]
    accuracies = [result[1] for result in results]
    sparsities = [result[2] for result in results]

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Lambda (λ)')
    ax1.set_ylabel('Accuracy (%)', color=color)
    ax1.plot(lambdas, accuracies, marker='o', color=color, label="Accuracy")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Sparsity (%)', color=color)
    ax2.plot(lambdas, sparsities, marker='o', linestyle='--', color=color, label="Sparsity")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title("Sparsity vs. Accuracy Trade-Off")
    plt.show()

def main():
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load CIFAR10 dataset
    train_loader, val_loader = get_data_loaders(batch_size=64)
    
    # Create model
    model = PrunableNetwork()
    
    # Train model
    train_model(model, train_loader, lambda_val=1e-2, device=device)
    
    # Evaluate model
    accuracy = evaluate(model, val_loader)
    print(f"Accuracy: {accuracy}")
    
    # Compute sparsity
    sparsity = compute_sparsity(model)
    print(f"Sparsity: {sparsity}")

    # Run experiments
    results = run_experiments()
    print("Results:", results)

    # Plot sparsity vs. accuracy
    plot_sparsity_vs_accuracy(results)

if __name__ == "__main__":
    main()
