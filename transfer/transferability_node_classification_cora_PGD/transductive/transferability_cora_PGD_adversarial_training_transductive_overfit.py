import torch
import os
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
import torch_geometric.utils as utils
import logging

# Set up logging to save the results in the file
log_dir = './results/transferability/performance'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'performance.log')

logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')

# Load the dataset
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]

# Define GCN model
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

# Define GAT model
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

# Define GPRGNN model
class GPRGNN(torch.nn.Module):
    def __init__(self, k_hops=10, alpha=0.1):
        super(GPRGNN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)
        self.k_hops = k_hops
        self.alpha = alpha
        self.linear = torch.nn.Linear(16, 16)  # Linear layer to propagate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = F.relu(self.conv1(x, edge_index))  # Initial embedding

        for _ in range(self.k_hops):
            h = self.alpha * F.relu(self.linear(h)) + (1 - self.alpha) * h  # Propagation step

        out = self.conv2(h, edge_index)
        return F.log_softmax(out, dim=1)

# PGD Adversarial Attack
def pgd_attack(data, model, epsilon=0.01, alpha=0.01, num_iter=40):
    # Clone the original data and enable gradient tracking
    perturbed_data = data.x.clone().detach().requires_grad_(True)
    
    for _ in range(num_iter):
        # Perform forward pass with the perturbed data
        data_adv = data.clone()
        data_adv.x = perturbed_data
        output = model(data_adv)
        
        # Compute loss on the adversarially perturbed data
        loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        
        # Zero the previous gradients
        model.zero_grad()

        # Perform backpropagation to compute the gradients of perturbed_data
        loss.backward()

        # Get the gradient of the perturbed data
        data_grad = perturbed_data.grad

        # Update the perturbed data by applying the gradient step
        perturbed_data = perturbed_data + alpha * data_grad.sign()

        # Clip the perturbed data to ensure it's within the epsilon-ball around the original data
        perturbation = torch.clamp(perturbed_data - data.x, min=-epsilon, max=epsilon)
        perturbed_data = torch.clamp(data.x + perturbation, min=0, max=1).detach().requires_grad_(True)

    return perturbed_data.detach()



# Adversarial training (train with adversarial examples)
def adversarial_train_model(model, data, epochs=200, epsilon=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Standard forward pass
        out = model(data)
        clean_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

        # Generate adversarial examples using PGD
        perturbed_data = pgd_attack(data, model, epsilon)
        
        # Forward pass on adversarial data
        data_adv = data.clone()
        data_adv.x = perturbed_data
        out_adv = model(data_adv)
        adv_loss = F.nll_loss(out_adv[data.train_mask], data.y[data.train_mask])

        # Total loss is the sum of clean and adversarial losses
        loss = clean_loss + adv_loss
        loss.backward()
        optimizer.step()

    return model

# Handle adjacency matrix with self-loops
def process_adj(edge_index, num_nodes):
    edge_index, edge_weight = utils.add_remaining_self_loops(edge_index, None, fill_value=1, num_nodes=num_nodes)
    return edge_index, edge_weight

# Evaluate Transferability (Clean and Adversarial)
def evaluate_transferability(model, data, perturbed_data):
    model.eval()
    with torch.no_grad():
        # Clean evaluation
        logits_clean = model(data)
        clean_accuracy = accuracy(logits_clean, data.y, data.test_mask)

        # Adversarial evaluation (perturbed)
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
gcn_model = GCN().to(torch.device('cpu'))
gat_model = GAT().to(torch.device('cpu'))
gprgnn_model = GPRGNN().to(torch.device('cpu'))  # GPRGNN model

# Process the adjacency matrix to handle self-loops
data.edge_index, data.edge_weight = process_adj(data.edge_index, data.num_nodes)

# Step 1: Adversarially train GCN, GAT, and GPRGNN
gcn_model = adversarial_train_model(gcn_model, data)
gat_model = adversarial_train_model(gat_model, data)
gprgnn_model = adversarial_train_model(gprgnn_model, data)

# Step 2: Generate Adversarial Examples using GCN
perturbed_data_gcn = pgd_attack(data, gcn_model)

# Step 3: Test GCN adversarial examples on GAT and GPRGNN (Robustness Transfer)
gcn_clean_acc, gcn_adv_acc = evaluate_transferability(gcn_model, data, perturbed_data_gcn)
gat_clean_acc_on_gcn_adv, gat_adv_acc_on_gcn_adv = evaluate_transferability(gat_model, data, perturbed_data_gcn)
gprgnn_clean_acc_on_gcn_adv, gprgnn_adv_acc_on_gcn_adv = evaluate_transferability(gprgnn_model, data, perturbed_data_gcn)

# Step 4: Generate Adversarial Examples using GAT
perturbed_data_gat = pgd_attack(data, gat_model)

# Step 5: Test GAT adversarial examples on GCN and GPRGNN
gat_clean_acc, gat_adv_acc = evaluate_transferability(gat_model, data, perturbed_data_gat)
gcn_clean_acc_on_gat_adv, gcn_adv_acc_on_gat_adv = evaluate_transferability(gcn_model, data, perturbed_data_gat)
gprgnn_clean_acc_on_gat_adv, gprgnn_adv_acc_on_gat_adv = evaluate_transferability(gprgnn_model, data, perturbed_data_gat)

# Step 6: Generate Adversarial Examples using GPRGNN
perturbed_data_gprgnn = pgd_attack(data, gprgnn_model)

# Step 7: Test GPRGNN adversarial examples on GCN and GAT
gprgnn_clean_acc, gprgnn_adv_acc = evaluate_transferability(gprgnn_model, data, perturbed_data_gprgnn)
gcn_clean_acc_on_gprgnn_adv, gcn_adv_acc_on_gprgnn_adv = evaluate_transferability(gcn_model, data, perturbed_data_gprgnn)
gat_clean_acc_on_gprgnn_adv, gat_adv_acc_on_gprgnn_adv = evaluate_transferability(gat_model, data, perturbed_data_gprgnn)

# Print the results and log them in a table format
results = (
    f"Model Transferability Results after Adversarial Training with PGD:\n"
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

# Print and log the results
print(results)
logging.info(results)
