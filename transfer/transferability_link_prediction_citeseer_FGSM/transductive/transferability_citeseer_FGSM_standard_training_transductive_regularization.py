import torch
import os
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
import torch_geometric.utils as utils
import logging
from torch.nn import BatchNorm1d
from tqdm import tqdm  # Import tqdm for progress bar

# Set up logging to save the results in the file
log_dir = './results/transferability/performance'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'performance.log')

logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')

# Load the Citeseer Dataset
dataset = Planetoid(root='./data', name='Citeseer')
data = dataset[0].to(torch.device('cuda'))

# Define GCN model with Batch Normalization
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)
        self.bn1 = BatchNorm1d(16)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Define GAT model with Batch Normalization
class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.conv1 = GATConv(dataset.num_node_features, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, dataset.num_classes, heads=1, concat=False, dropout=0.6)
        self.bn1 = BatchNorm1d(8 * 8)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Define GPRGNN model with Batch Normalization
class GPRGNN(torch.nn.Module):
    def __init__(self, k_hops=10, alpha=0.1):
        super(GPRGNN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)
        self.k_hops = k_hops
        self.alpha = alpha
        self.linear = torch.nn.Linear(16, 16)
        self.bn1 = BatchNorm1d(16)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = F.relu(self.conv1(x, edge_index))
        h = self.bn1(h)
        for _ in range(self.k_hops):
            h = self.alpha * F.relu(self.linear(h)) + (1 - self.alpha) * h
        out = self.conv2(h, edge_index)
        return F.log_softmax(out, dim=1)

# Train the model with tqdm for progress
def train_model(model, data, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for epoch in tqdm(range(epochs), desc="Training Model"):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    return model

# FGSM Adversarial Attack
def fgsm_attack(data, model, epsilon=0.1):
    data.x.requires_grad = True
    output = model(data)
    loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    model.zero_grad()
    loss.backward()
    data_grad = data.x.grad.data
    perturbed_data = data.x + epsilon * data_grad.sign()
    return perturbed_data.detach()

# Handle adjacency matrix with self-loops
def process_adj(edge_index, num_nodes):
    edge_index, edge_weight = utils.add_remaining_self_loops(edge_index, None, fill_value=1, num_nodes=num_nodes)
    return edge_index, edge_weight

# Evaluate Transferability
def evaluate_transferability(model, data, perturbed_data):
    model.eval()
    with torch.no_grad():
        logits_clean = model(data)
        clean_accuracy = accuracy(logits_clean, data.y, data.test_mask)
        perturbed_data_copy = data.clone()
        perturbed_data_copy.x = perturbed_data
        logits_adversarial = model(perturbed_data_copy)
        adversarial_accuracy = accuracy(logits_adversarial, data.y, data.test_mask)
    return clean_accuracy, adversarial_accuracy

# Accuracy calculation helper
def accuracy(output, labels, mask):
    _, pred = output.max(dim=1)
    correct = (pred[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    return correct / total

# Main Execution
device = torch.device('cuda')
gcn_model = GCN().to(device)
gat_model = GAT().to(device)
gprgnn_model = GPRGNN().to(device)

# Process the adjacency matrix to handle self-loops
data.edge_index, data.edge_weight = process_adj(data.edge_index, data.num_nodes)

# Train GCN, GAT, and GPRGNN
gcn_model = train_model(gcn_model, data)
gat_model = train_model(gat_model, data)
gprgnn_model = train_model(gprgnn_model, data)

# Generate Adversarial Examples using GCN
perturbed_data_gcn = fgsm_attack(data, gcn_model)

# Test GCN adversarial examples on GAT and GPRGNN
gcn_clean_acc, gcn_adv_acc = evaluate_transferability(gcn_model, data, perturbed_data_gcn)
gat_clean_acc_on_gcn_adv, gat_adv_acc_on_gcn_adv = evaluate_transferability(gat_model, data, perturbed_data_gcn)
gprgnn_clean_acc_on_gcn_adv, gprgnn_adv_acc_on_gcn_adv = evaluate_transferability(gprgnn_model, data, perturbed_data_gcn)

# Generate Adversarial Examples using GAT
perturbed_data_gat = fgsm_attack(data, gat_model)

# Test GAT adversarial examples on GCN and GPRGNN
gat_clean_acc, gat_adv_acc = evaluate_transferability(gat_model, data, perturbed_data_gat)
gcn_clean_acc_on_gat_adv, gcn_adv_acc_on_gat_adv = evaluate_transferability(gcn_model, data, perturbed_data_gat)
gprgnn_clean_acc_on_gat_adv, gprgnn_adv_acc_on_gat_adv = evaluate_transferability(gprgnn_model, data, perturbed_data_gat)

# Generate Adversarial Examples using GPRGNN
perturbed_data_gprgnn = fgsm_attack(data, gprgnn_model)

# Test GPRGNN adversarial examples on GCN and GAT
gprgnn_clean_acc, gprgnn_adv_acc = evaluate_transferability(gprgnn_model, data, perturbed_data_gprgnn)
gcn_clean_acc_on_gprgnn_adv, gcn_adv_acc_on_gprgnn_adv = evaluate_transferability(gcn_model, data, perturbed_data_gprgnn)
gat_clean_acc_on_gprgnn_adv, gat_adv_acc_on_gprgnn_adv = evaluate_transferability(gat_model, data, perturbed_data_gprgnn)

# Print and log results
results = (
    f"Model Transferability Results:\n"
    f"-----------------------------------------\n"
    f"From GCN:\n"
    f"GCN Clean Accuracy: {gcn_clean_acc:.4f}, GCN Adversarial Accuracy: {gcn_adv_acc:.4f}\n"
    f"GAT Clean on GCN Adv: {gat_clean_acc_on_gcn_adv:.4f}, GAT Adversarial on GCN Adv: {gat_adv_acc_on_gcn_adv:.4f}\n"
    f"GPRGNN Clean on GCN Adv: {gprgnn_clean_acc_on_gcn_adv:.4f}, GPRGNN Adversarial on GCN Adv: {gprgnn_adv_acc_on_gcn_adv:.4f}\n"
    f"-----------------------------------------\n"
    f"From GAT:\n"
    f"GAT Clean Accuracy: {gat_clean_acc:.4f}, GAT Adversarial Accuracy: {gat_adv_acc:.4f}\n"
    f"GCN Clean on GAT Adv: {gcn_clean_acc_on_gat_adv:.4f}, GCN Adversarial on GAT Adv: {gcn_adv_acc_on_gat_adv:.4f}\n"
    f"GPRGNN Clean on GAT Adv: {gprgnn_clean_acc_on_gat_adv:.4f}, GPRGNN Adversarial on GAT Adv: {gprgnn_adv_acc_on_gat_adv:.4f}\n"
    f"-----------------------------------------\n"
    f"From GPRGNN:\n"
    f"GPRGNN Clean Accuracy: {gprgnn_clean_acc:.4f}, GPRGNN Adversarial Accuracy: {gprgnn_adv_acc:.4f}\n"
    f"GCN Clean on GPRGNN Adv: {gcn_clean_acc_on_gprgnn_adv:.4f}, GCN Adversarial on GPRGNN Adv: {gcn_adv_acc_on_gprgnn_adv:.4f}\n"
    f"GAT Clean on GPRGNN Adv: {gat_clean_acc_on_gprgnn_adv:.4f}, GAT Adversarial on GPRGNN Adv: {gat_adv_acc_on_gprgnn_adv:.4f}\n"
)

print(results)
logging.info(results)
