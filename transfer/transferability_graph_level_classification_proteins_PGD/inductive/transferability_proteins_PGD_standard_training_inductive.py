import torch
import os
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.datasets import TUDataset
import torch.nn.functional as F
import logging
from torch_geometric.loader import DataLoader as GeoDataLoader
from tqdm import tqdm  # Import tqdm for progress bars

# Set up logging to save the results in the file
log_dir = './results/transferability/performance_pgd'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'performance.log')

logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')

# Determine device (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 1: Load PROTEINS Dataset from PyTorch Geometric
dataset = TUDataset(root='./data', name='PROTEINS')

# Define GCN, GAT, and GPRGNN models with global pooling
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # Apply global mean pooling
        return F.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.conv1 = GATConv(dataset.num_node_features, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, dataset.num_classes, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # Apply global mean pooling
        return F.log_softmax(x, dim=1)

class GPRGNN(torch.nn.Module):
    def __init__(self, k_hops=10, alpha=0.1):
        super(GPRGNN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)
        self.k_hops = k_hops
        self.alpha = alpha
        self.linear = torch.nn.Linear(16, 16)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h = F.relu(self.conv1(x, edge_index))
        for _ in range(self.k_hops):
            h = self.alpha * F.relu(self.linear(h)) + (1 - self.alpha) * h
        out = self.conv2(h, edge_index)
        out = global_mean_pool(out, batch)  # Apply global mean pooling
        return F.log_softmax(out, dim=1)

# PGD Adversarial Attack with fixed perturbed data usage
def pgd_attack(data, model, epsilon=0.3, alpha=0.01, num_iter=20):
    perturbed_data = data.clone()  # Clone the original data
    perturbed_data.x = perturbed_data.x.clone().detach().to(device)  # Clone and detach features
    perturbed_data.x.requires_grad = True

    for _ in range(num_iter):
        output = model(perturbed_data)
        loss = F.nll_loss(output, data.y)
        model.zero_grad()
        loss.backward()

        # Apply perturbation
        perturbed_data.x = perturbed_data.x + alpha * perturbed_data.x.grad.sign()
        perturbation = torch.clamp(perturbed_data.x - data.x, min=-epsilon, max=epsilon)
        perturbed_data.x = torch.clamp(data.x + perturbation, min=0, max=1).detach()
        perturbed_data.x.requires_grad = True  # Retain grad for next iteration

    return perturbed_data.x

# Train the model
def train_model(model, loader, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for epoch in tqdm(range(epochs), desc="Training"):
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
    return model

# Adjusted evaluate_model function
def evaluate_model(model, loader, apply_adversarial=True, epsilon=0.5, alpha=0.05, num_iter=20):
    model.eval()
    clean_accuracy = 0
    adversarial_accuracy = 0

    for data in tqdm(loader, desc="Evaluating"):
        data = data.to(device)
        logits_clean = model(data)
        clean_accuracy += accuracy(logits_clean, data.y)

        if apply_adversarial:
            # Generate perturbed data and use it for adversarial evaluation
            perturbed_data = pgd_attack(data, model, epsilon, alpha, num_iter)
            data_adv = data.clone()
            data_adv.x = perturbed_data  # Use the perturbed data features
            logits_adversarial = model(data_adv)
            adversarial_accuracy += accuracy(logits_adversarial, data.y)

    clean_accuracy /= len(loader)
    adversarial_accuracy /= len(loader)
    return clean_accuracy, adversarial_accuracy

# Accuracy calculation helper
def accuracy(output, labels):
    _, pred = output.max(dim=1)
    correct = (pred == labels).sum().item()
    total = labels.size(0)
    return correct / total

# Main Execution
train_loader = GeoDataLoader(dataset[:len(dataset) // 2], batch_size=32, shuffle=True)
test_loader = GeoDataLoader(dataset[len(dataset) // 2:], batch_size=32)

gcn_model = GCN().to(device)
gat_model = GAT().to(device)
gprgnn_model = GPRGNN().to(device)

# Train models
gcn_model = train_model(gcn_model, train_loader)
gat_model = train_model(gat_model, train_loader)
gprgnn_model = train_model(gprgnn_model, train_loader)

# Evaluate transferability for adversarial examples generated by each model

# Step 1: Generate adversarial examples using GCN and test on all models
gcn_clean_acc, gcn_adv_acc = evaluate_model(gcn_model, test_loader, apply_adversarial=True, epsilon=0.7, alpha=0.05, num_iter=50)
gat_clean_acc_on_gcn_adv, gat_adv_acc_on_gcn_adv = evaluate_model(gat_model, test_loader, apply_adversarial=True, epsilon=0.7, alpha=0.05, num_iter=50)
gprgnn_clean_acc_on_gcn_adv, gprgnn_adv_acc_on_gcn_adv = evaluate_model(gprgnn_model, test_loader, apply_adversarial=True, epsilon=0.7, alpha=0.05, num_iter=50)

# Step 2: Generate adversarial examples using GAT and test on all models
gat_clean_acc, gat_adv_acc = evaluate_model(gat_model, test_loader, apply_adversarial=True, epsilon=0.7, alpha=0.05, num_iter=50)
gcn_clean_acc_on_gat_adv, gcn_adv_acc_on_gat_adv = evaluate_model(gcn_model, test_loader, apply_adversarial=True, epsilon=0.7, alpha=0.05, num_iter=50)
gprgnn_clean_acc_on_gat_adv, gprgnn_adv_acc_on_gat_adv = evaluate_model(gprgnn_model, test_loader, apply_adversarial=True, epsilon=0.7, alpha=0.05, num_iter=50)

# Step 3: Generate adversarial examples using GPRGNN and test on all models
gprgnn_clean_acc, gprgnn_adv_acc = evaluate_model(gprgnn_model, test_loader, apply_adversarial=True, epsilon=0.7, alpha=0.05, num_iter=50)
gcn_clean_acc_on_gprgnn_adv, gcn_adv_acc_on_gprgnn_adv = evaluate_model(gcn_model, test_loader, apply_adversarial=True, epsilon=0.7, alpha=0.05, num_iter=50)
gat_clean_acc_on_gprgnn_adv, gat_adv_acc_on_gprgnn_adv = evaluate_model(gat_model, test_loader, apply_adversarial=True, epsilon=0.7, alpha=0.05, num_iter=50)

# Print and log results
results = (
    f"Model Transferability Results with PGD Test Set Perturbation:\n"
    f"-----------------------------------------\n"
    f"From GCN (Adversarially Trained):\n"
    f"GCN Clean Accuracy: {gcn_clean_acc:.4f}, GCN Adversarial Accuracy: {gcn_adv_acc:.4f}\n"
    f"GAT Clean on GCN Adv: {gat_clean_acc_on_gcn_adv:.4f}, GAT Adversarial on GCN Adv: {gat_adv_acc_on_gcn_adv:.4f}\n"
    f"GPRGNN Clean on GCN Adv: {gprgnn_clean_acc_on_gcn_adv:.4f}, GPRGNN Adversarial on GCN Adv: {gprgnn_adv_acc_on_gcn_adv:.4f}\n"
    f"-----------------------------------------\n"
    f"From GAT (Adversarially Trained):\n"
    f"GAT Clean Accuracy: {gat_clean_acc:.4f}, GAT Adversarial Accuracy: {gat_adv_acc:.4f}\n"
    f"GCN Clean on GAT Adv: {gcn_clean_acc_on_gat_adv:.4f}, GCN Adversarial on GAT Adv: {gcn_adv_acc_on_gat_adv:.4f}\n"
    f"GPRGNN Clean on GAT Adv: {gprgnn_clean_acc_on_gat_adv:.4f}, GPRGNN Adversarial on GAT Adv: {gprgnn_adv_acc_on_gat_adv:.4f}\n"
    f"-----------------------------------------\n"
    f"From GPRGNN (Adversarially Trained):\n"
    f"GPRGNN Clean Accuracy: {gprgnn_clean_acc:.4f}, GPRGNN Adversarial Accuracy: {gprgnn_adv_acc:.4f}\n"
    f"GCN Clean on GPRGNN Adv: {gcn_clean_acc_on_gprgnn_adv:.4f}, GCN Adversarial on GPRGNN Adv: {gcn_adv_acc_on_gprgnn_adv:.4f}\n"
    f"GAT Clean on GPRGNN Adv: {gat_clean_acc_on_gprgnn_adv:.4f}, GAT Adversarial on GPRGNN Adv: {gat_adv_acc_on_gprgnn_adv:.4f}\n"
)

print(results)
logging.info(results)
