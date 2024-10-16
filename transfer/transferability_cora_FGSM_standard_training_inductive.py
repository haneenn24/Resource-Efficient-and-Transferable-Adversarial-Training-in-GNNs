import torch
import os
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
import torch_geometric.utils as utils
import logging
from sklearn.model_selection import train_test_split


# Set up logging to save the results in the file
log_dir = './results/transferability/performance'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'performance.log')

logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')

# Step 1: Load Cora Dataset from PyTorch Geometric
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]

# Step 2: Create separate subgraphs for training and testing (inductive)
def create_inductive_split(data):
    # Split the nodes into training and test sets
    train_idx, test_idx = train_test_split(range(data.num_nodes), test_size=0.3, random_state=42)
    
    # Create masks for the training and test sets
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    
    # Define training and test subgraphs
    train_data = data.clone()
    test_data = data.clone()
    
    train_data.train_mask = train_mask
    train_data.test_mask = torch.zeros_like(train_mask)  # No test mask for training subgraph
    
    test_data.train_mask = torch.zeros_like(test_mask)  # No training mask for test subgraph
    test_data.test_mask = test_mask
    
    # Keep only the training edges in train_data, and only test edges in test_data
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

# Main Execution (Inductive Learning)
gcn_model = GCN().to(torch.device('cpu'))
gat_model = GAT().to(torch.device('cpu'))
gprgnn_model = GPRGNN().to(torch.device('cpu'))  # GPRGNN model

# Train GCN, GAT, and GPRGNN on the training subgraph (inductive)
gcn_model = train_model(gcn_model, train_data)
gat_model = train_model(gat_model, train_data)
gprgnn_model = train_model(gprgnn_model, train_data)

# Evaluate models on the test subgraph (inductive)
def evaluate_model(model, test_data):
    model.eval()
    with torch.no_grad():
        out = model(test_data)
        acc = accuracy(out, test_data.y, test_data.test_mask)
    return acc

# Accuracy calculation helper
def accuracy(output, labels, mask):
    _, pred = output.max(dim=1)
    correct = (pred[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    return correct / total

# Evaluate models on unseen test subgraph
gcn_acc = evaluate_model(gcn_model, test_data)
gat_acc = evaluate_model(gat_model, test_data)
gprgnn_acc = evaluate_model(gprgnn_model, test_data)

# Log the results
results = (
    f"Inductive Learning Results:\n"
    f"GCN Accuracy on Unseen Test Graph: {gcn_acc:.4f}\n"
    f"GAT Accuracy on Unseen Test Graph: {gat_acc:.4f}\n"
    f"GPRGNN Accuracy on Unseen Test Graph: {gprgnn_acc:.4f}\n"
)
print(results)
logging.info(results)
