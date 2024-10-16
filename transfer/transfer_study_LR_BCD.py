import torch
import os
from torch_geometric.nn import GCNConv, GATConv  # Import GATConv
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, remove_self_loops
import logging
from collections import defaultdict
from torch_sparse import SparseTensor
import numpy as np
from tqdm import tqdm

# Set up logging to save the results in the file
log_dir = './results/transferability/performance'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'performance.log')

logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')

# Step 1: Load Cora Dataset from PyTorch Geometric
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]

# Step 2: Define GCN Model
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Remove any existing self-loops
        edge_index, _ = remove_self_loops(edge_index)
        
        # Add self-loops to the edge_index tensor, with correct number of nodes
        edge_index, _ = add_self_loops(edge_index, num_nodes=data.x.size(0))

        # Process the input with GCN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        
        # Return log softmax output
        return F.log_softmax(x, dim=1)


# Step 3: Define GAT Model using GATConv instead of WeightedGATConv
class GAT(torch.nn.Module):
    def __init__(self,  n_features: int, n_classes: int,
                 hidden_dim: int = 16, dropout = 0.5, **kwargs):
        super().__init__()
        self.dropout = dropout
        # We use standard GATConv from torch_geometric
        self.conv1 = GATConv(n_features, hidden_dim, add_self_loops=True)
        self.conv2 = GATConv(hidden_dim, n_classes, add_self_loops=True)

    def forward(self, data, adj=None, **kwargs):
        edge_index = data.edge_index
        x = data.x
        
        # Apply GAT layers
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)

# Step 4: Train the model
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

# Step 5: LRBCD Adversarial Attack
class LRBCD:
    def __init__(self, model, attr, adj, labels, idx_train, lr_factor=100, epochs=400, **kwargs):
        self.model = model
        self.attr = attr
        self.adj = adj
        self.labels = labels
        self.idx_train = idx_train
        self.lr_factor = lr_factor
        self.epochs = epochs
        self.kwargs = kwargs
        self.perturbed_edge_weight = None
        self.attack_statistics = defaultdict(list)
    
    def attack(self, n_perturbations):
        for epoch in tqdm(range(self.epochs)):
            logits = self._get_logits()
            loss = F.nll_loss(logits[self.idx_train], self.labels[self.idx_train])
            loss.backward()
            # Gradient-based updates on edge weights
            self._update_edge_weights(n_perturbations)
        return self.adj  # Return perturbed adjacency
    
    def _get_logits(self):
        data = self._prepare_data(self.attr, self.adj)
        return self.model(data)

    def _prepare_data(self, attr, adj):
        # Create a new data object that includes both attributes and the adjacency matrix
        data = Data(x=attr, edge_index=adj)
        return data

    def _update_edge_weights(self, n_perturbations):
        # Calculate the gradient for the perturbed edge weights
        gradient = self.perturbed_edge_weight.grad

        # Ensure gradient is not None and handle numerical issues (optional)
        if gradient is None:
            raise ValueError("Gradient for perturbed edge weights is None. Ensure backward() is called.")
        
        gradient = torch.nan_to_num(gradient)

        # Perform a gradient update step (you may scale this with a learning rate if needed)
        self.perturbed_edge_weight.data.add_(self.lr_factor * gradient)

        # Ensure edge weights are non-negative (or handle other constraints like sparsity)
        self.perturbed_edge_weight.data.clamp_(min=self.eps)

        # Project back to the valid perturbation budget
        self.perturbed_edge_weight = self.projection.project(self.perturbed_edge_weight, self.modified_edge_index)


def lr_bcd_attack(data, model, epsilon=0.1):
    # Adjacency matrix for LRBCD
    adj = SparseTensor.from_edge_index(data.edge_index).to_dense()  # Convert to dense for simplicity
    lr_bcd = LRBCD(model, data.x, adj, data.y, data.train_mask)
    perturbed_adj = lr_bcd.attack(n_perturbations=20)
    return perturbed_adj

# Step 6: Evaluate Transferability
def evaluate_transferability(model, data, perturbed_adj):
    model.eval()
    with torch.no_grad():
        # Clean evaluation (unperturbed graph)
        logits_clean = model(data)
        clean_accuracy = accuracy(logits_clean, data.y, data.test_mask)

        # Adversarial evaluation (perturbed adjacency)
        data.edge_index = torch.tensor(perturbed_adj.nonzero()).to(data.edge_index.device)  # Update edge index with perturbed
        logits_adversarial = model(data)
        adversarial_accuracy = accuracy(logits_adversarial, data.y, data.test_mask)

    return clean_accuracy, adversarial_accuracy

# Accuracy calculation
def accuracy(output, labels, mask):
    _, pred = output.max(dim=1)
    correct = (pred[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    return correct / total

# Main Execution:
gcn_model = GCN().to(torch.device('cpu'))
gat_model = GAT(n_features=dataset.num_node_features, n_classes=dataset.num_classes).to(torch.device('cpu'))

# Train GCN
gcn_model = train_model(gcn_model, data)

# Train GAT
gat_model = train_model(gat_model, data)

# Step 7: Generate Adversarial Examples using LRBCD Attack
perturbed_adj = lr_bcd_attack(data, gcn_model)

# Step 8: Evaluate Transferability on GCN and GAT
clean_acc_gcn, perturbed_acc_gcn = evaluate_transferability(gcn_model, data, perturbed_adj)
clean_acc_gat, perturbed_acc_gat = evaluate_transferability(gat_model, data, perturbed_adj)

# Step 9: Print and log results
results = (
    f"GCN Clean Accuracy: {clean_acc_gcn:.4f}, GCN Adversarial Accuracy: {perturbed_acc_gcn:.4f}\n"
    f"GAT Clean Accuracy: {clean_acc_gat:.4f}, GAT Adversarial Accuracy: {perturbed_acc_gat:.4f}\n"
)

print(results)
logging.info(results)
