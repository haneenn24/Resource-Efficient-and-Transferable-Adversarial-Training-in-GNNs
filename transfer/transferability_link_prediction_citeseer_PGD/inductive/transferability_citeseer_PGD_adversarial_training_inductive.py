import torch
import os
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
import torch_geometric.utils as utils
import logging
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # Import tqdm for progress bar

# Set up logging to save the results in the file
log_dir = './results/transferability/performance'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'performance.log')

logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')

# Step 1: Load Citeseer Dataset from PyTorch Geometric
dataset = Planetoid(root='./data', name='Citeseer')
data = dataset[0].to(torch.device('cuda'))

# Step 2: Create separate subgraphs for training and testing (inductive)
def create_inductive_split(data):
    train_idx, test_idx = train_test_split(range(data.num_nodes), test_size=0.3, random_state=42)
    
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device='cuda')
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device='cuda')
    
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    
    train_data = data.clone()
    test_data = data.clone()
    
    train_data.train_mask = train_mask
    train_data.test_mask = torch.zeros_like(train_mask)
    
    test_data.train_mask = torch.zeros_like(test_mask)
    test_data.test_mask = test_mask
    
    train_data.edge_index, _ = utils.subgraph(train_mask, data.edge_index)
    test_data.edge_index, _ = utils.subgraph(test_mask, data.edge_index)
    
    return train_data, test_data

train_data, test_data = create_inductive_split(data)

# Define GCN, GAT, and GPRGNN models
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = data.edge_weight if 'edge_weight' in data else None
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.conv1 = GATConv(dataset.num_node_features, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8*8, dataset.num_classes, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))  # No edge_weight for GATConv
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)  # No edge_weight for GATConv
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
        x, edge_index = data.x, data.edge_index
        h = F.relu(self.conv1(x, edge_index))  # Initial embedding
        for _ in range(self.k_hops):
            h = self.alpha * F.relu(self.linear(h)) + (1 - self.alpha) * h  # Propagation step
        out = self.conv2(h, edge_index)
        return F.log_softmax(out, dim=1)

# PGD Adversarial Attack
def pgd_attack(data, model, epsilon=0.3, alpha=0.01, num_iter=20):
    original_data = data.x.clone()
    for _ in tqdm(range(num_iter), desc="PGD Attack"):
        data.x.requires_grad = True
        output = model(data)
        loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        model.zero_grad()
        loss.backward()
        data_grad = data.x.grad.data
        perturbed_data = data.x + alpha * data_grad.sign()
        perturbation = torch.clamp(perturbed_data - original_data, min=-epsilon, max=epsilon)
        data.x = torch.clamp(original_data + perturbation, min=0, max=1).detach()
    return data.x

# PGD Adversarial Attack on Test Data
def pgd_attack_test(data, model, epsilon=0.3, alpha=0.01, num_iter=20):
    original_data = data.x.clone()
    for _ in tqdm(range(num_iter), desc="PGD Attack Test Data"):
        data.x.requires_grad = True
        output = model(data)
        loss = F.nll_loss(output[data.test_mask], data.y[data.test_mask])
        model.zero_grad()
        loss.backward()
        data_grad = data.x.grad.data
        perturbed_data = data.x + alpha * data_grad.sign()
        perturbation = torch.clamp(perturbed_data - original_data, min=-epsilon, max=epsilon)
        data.x = torch.clamp(original_data + perturbation, min=0, max=1).detach()
    return data.x

# Adversarial training (train with PGD adversarial examples)
def adversarial_train_model(model, data, epochs=100, epsilon=0.3, alpha=0.01, num_iter=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for epoch in tqdm(range(epochs), desc=f'Training with PGD epsilon={epsilon}'):
        model.train()
        optimizer.zero_grad()
        
        # Standard forward pass
        out = model(data)
        clean_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

        # Generate adversarial examples using PGD
        perturbed_data = pgd_attack(data, model, epsilon, alpha, num_iter)
        
        # Forward pass on adversarial data
        out_adv = model(data.clone())
        adv_loss = F.nll_loss(out_adv[data.train_mask], data.y[data.train_mask])

        # Total loss is the sum of clean and adversarial losses
        loss = clean_loss + adv_loss
        loss.backward()
        optimizer.step()

    return model

# Evaluate models on clean and adversarial examples
def evaluate_transferability(model, data, perturbed_data, apply_adversarial=True, epsilon=0.3, alpha=0.01, num_iter=20):
    logits_clean = model(data)
    clean_accuracy = accuracy(logits_clean, data.y, data.test_mask)

    if apply_adversarial:
        perturbed_data = pgd_attack_test(data, model, epsilon, alpha, num_iter)
        
        perturbed_data_copy = data.clone()
        perturbed_data_copy.x = perturbed_data
        logits_adversarial = model(perturbed_data_copy)
        adversarial_accuracy = accuracy(logits_adversarial, data.y, data.test_mask)
    else:
        adversarial_accuracy = None

    return clean_accuracy, adversarial_accuracy

# Accuracy calculation helper
def accuracy(output, labels, mask):
    _, pred = output.max(dim=1)
    correct = (pred[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    return correct / total

# Main Execution (Inductive Adversarial Training)
device = torch.device('cuda')
gcn_model = GCN().to(device)
gat_model = GAT().to(device)
gprgnn_model = GPRGNN().to(device)

# Adversarial training on training subgraph
gcn_model = adversarial_train_model(gcn_model, train_data)
gat_model = adversarial_train_model(gat_model, train_data)
gprgnn_model = adversarial_train_model(gprgnn_model, train_data)

# Generate adversarial examples using GCN
perturbed_data_gcn = pgd_attack(train_data, gcn_model)

# Evaluate on test subgraph (inductive evaluation)
gcn_clean_acc, gcn_adv_acc = evaluate_transferability(gcn_model, test_data, perturbed_data_gcn)
gat_clean_acc_on_gcn_adv, gat_adv_acc_on_gcn_adv = evaluate_transferability(gat_model, test_data, perturbed_data_gcn)
gprgnn_clean_acc_on_gcn_adv, gprgnn_adv_acc_on_gcn_adv = evaluate_transferability(gprgnn_model, test_data, perturbed_data_gcn)

# Generate adversarial examples using GAT
perturbed_data_gat = pgd_attack(train_data, gat_model)

# Evaluate on test subgraph
gat_clean_acc, gat_adv_acc = evaluate_transferability(gat_model, test_data, perturbed_data_gat)
gcn_clean_acc_on_gat_adv, gcn_adv_acc_on_gat_adv = evaluate_transferability(gcn_model, test_data, perturbed_data_gat)
gprgnn_clean_acc_on_gat_adv, gprgnn_adv_acc_on_gat_adv = evaluate_transferability(gprgnn_model, test_data, perturbed_data_gat)

# Generate adversarial examples using GPRGNN
perturbed_data_gprgnn = pgd_attack(train_data, gprgnn_model)

# Evaluate on test subgraph
gprgnn_clean_acc, gprgnn_adv_acc = evaluate_transferability(gprgnn_model, test_data, perturbed_data_gprgnn)
gcn_clean_acc_on_gprgnn_adv, gcn_adv_acc_on_gprgnn_adv = evaluate_transferability(gcn_model, test_data, perturbed_data_gprgnn)
gat_clean_acc_on_gprgnn_adv, gat_adv_acc_on_gprgnn_adv = evaluate_transferability(gat_model, test_data, perturbed_data_gprgnn)

# Print and log results
results = (
    f"Inductive Adversarial Training Results with PGD:\n"
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
