import torch
import os
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
import torch_geometric.utils as utils
import logging
from sklearn.model_selection import train_test_split

# Set up logging to save the results in the file
log_dir = './results/transferability/performance_pgd'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'performance_pgd.log')

logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')

# Step 1: Load Cora Dataset from PyTorch Geometric
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]

# Step 2: Create separate subgraphs for training and testing (inductive)
def create_inductive_split(data):
    train_idx, test_idx = train_test_split(range(data.num_nodes), test_size=0.3, random_state=42)
    
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    
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
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
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
        h = F.relu(self.conv1(x, edge_index))
        for _ in range(self.k_hops):
            h = self.alpha * F.relu(self.linear(h)) + (1 - self.alpha) * h
        out = self.conv2(h, edge_index)
        return F.log_softmax(out, dim=1)

# PGD Adversarial Attack on Training Data
def pgd_attack(data, model, epsilon=0.3, alpha=0.01, num_iter=20):
    original_data = data.x.clone()
    for _ in range(num_iter):
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
    for _ in range(num_iter):
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

# Train the model on the training subgraph
def train_model(model, data, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    return model

# Evaluate models on clean and adversarial data
def evaluate_model(model, data, apply_adversarial=True, epsilon=0.3, alpha=0.01, num_iter=20):
    # model.eval()
    # with torch.no_grad():
    # Clean evaluation
    logits_clean = model(data)
    clean_accuracy = accuracy(logits_clean, data.y, data.test_mask)

    if apply_adversarial:
        # Generate adversarial examples on the test data using PGD
        perturbed_data = pgd_attack_test(data, model, epsilon, alpha, num_iter)
        
        # Adversarial evaluation
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

# Main Execution (Inductive Learning with Adversarial Evaluation)
gcn_model = GCN().to(torch.device('cpu'))
gat_model = GAT().to(torch.device('cpu'))
gprgnn_model = GPRGNN().to(torch.device('cpu'))

# Train GCN, GAT, and GPRGNN on the training subgraph
gcn_model = train_model(gcn_model, train_data)
gat_model = train_model(gat_model, train_data)
gprgnn_model = train_model(gprgnn_model, train_data)

# Step 1: Generate adversarial examples using GCN on the test set
gcn_clean_acc, gcn_adv_acc = evaluate_model(gcn_model, test_data, apply_adversarial=True, epsilon=0.3, alpha=0.01, num_iter=40)
gat_clean_acc_on_gcn_adv, gat_adv_acc_on_gcn_adv = evaluate_model(gat_model, test_data, apply_adversarial=True, epsilon=0.3, alpha=0.01, num_iter=40)
gprgnn_clean_acc_on_gcn_adv, gprgnn_adv_acc_on_gcn_adv = evaluate_model(gprgnn_model, test_data, apply_adversarial=True, epsilon=0.3, alpha=0.01, num_iter=40)

# Step 2: Generate adversarial examples using GAT on the test set
gat_clean_acc, gat_adv_acc = evaluate_model(gat_model, test_data, apply_adversarial=True, epsilon=0.3, alpha=0.01, num_iter=40)
gcn_clean_acc_on_gat_adv, gcn_adv_acc_on_gat_adv = evaluate_model(gcn_model, test_data, apply_adversarial=True, epsilon=0.3, alpha=0.01, num_iter=40)
gprgnn_clean_acc_on_gat_adv, gprgnn_adv_acc_on_gat_adv = evaluate_model(gprgnn_model, test_data, apply_adversarial=True, epsilon=0.3, alpha=0.01, num_iter=40)

# Step 3: Generate adversarial examples using GPRGNN on the test set
gprgnn_clean_acc, gprgnn_adv_acc = evaluate_model(gprgnn_model, test_data, apply_adversarial=True, epsilon=0.3, alpha=0.01, num_iter=40)
gcn_clean_acc_on_gprgnn_adv, gcn_adv_acc_on_gprgnn_adv = evaluate_model(gcn_model, test_data, apply_adversarial=True, epsilon=0.3, alpha=0.01, num_iter=40)
gat_clean_acc_on_gprgnn_adv, gat_adv_acc_on_gprgnn_adv = evaluate_model(gat_model, test_data, apply_adversarial=True, epsilon=0.3, alpha=0.01, num_iter=40)

# Print and log results
results = (
    f"Standard Training Results with PGD Test Set Perturbation:\n"
    f"-----------------------------------------\n"
    f"From GCN (Standard Trained):\n"
    f"GCN Clean Accuracy: {gcn_clean_acc:.4f}, GCN Adversarial Accuracy: {gcn_adv_acc:.4f}\n"
    f"GAT Clean on GCN Adv: {gat_clean_acc_on_gcn_adv:.4f}, GAT Adversarial on GCN Adv: {gat_adv_acc_on_gcn_adv:.4f}\n"
    f"GPRGNN Clean on GCN Adv: {gprgnn_clean_acc_on_gcn_adv:.4f}, GPRGNN Adversarial on GCN Adv: {gprgnn_adv_acc_on_gcn_adv:.4f}\n"
    f"-----------------------------------------\n"
    f"From GAT (Standard Trained):\n"
    f"GAT Clean Accuracy: {gat_clean_acc:.4f}, GAT Adversarial Accuracy: {gat_adv_acc:.4f}\n"
    f"GCN Clean on GAT Adv: {gcn_clean_acc_on_gat_adv:.4f}, GCN Adversarial on GAT Adv: {gcn_adv_acc_on_gat_adv:.4f}\n"
    f"GPRGNN Clean on GAT Adv: {gprgnn_clean_acc_on_gat_adv:.4f}, GPRGNN Adversarial on GAT Adv: {gprgnn_adv_acc_on_gat_adv:.4f}\n"
    f"-----------------------------------------\n"
    f"From GPRGNN (Standard Trained):\n"
    f"GPRGNN Clean Accuracy: {gprgnn_clean_acc:.4f}, GPRGNN Adversarial Accuracy: {gprgnn_adv_acc:.4f}\n"
    f"GCN Clean on GPRGNN Adv: {gcn_clean_acc_on_gprgnn_adv:.4f}, GCN Adversarial on GPRGNN Adv: {gcn_adv_acc_on_gprgnn_adv:.4f}\n"
    f"GAT Clean on GPRGNN Adv: {gat_clean_acc_on_gprgnn_adv:.4f}, GAT Adversarial on GPRGNN Adv: {gat_adv_acc_on_gprgnn_adv:.4f}\n"
)

print(results)
logging.info(results)
