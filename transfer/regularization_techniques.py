import torch
import torch.nn as nn
import dgl
from dgl.nn import GraphConv, GATConv
import torch.nn.functional as F

# Define Regularization Techniques
class GNNWithRegularization(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, task_type, use_batch_norm=True, dropout_rate=0.5, edge_dropout_rate=0.2):
        super(GNNWithRegularization, self).__init__()
        self.task_type = task_type
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, output_dim)
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.edge_dropout_rate = edge_dropout_rate
        
        if use_batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
            self.batch_norm2 = nn.BatchNorm1d(output_dim)
    
    def forward(self, g, features):
        # Edge dropout for graph data augmentation
        if self.edge_dropout_rate > 0:
            edge_mask = torch.rand(g.num_edges()) > self.edge_dropout_rate
            g = dgl.edge_subgraph(g, edge_mask, preserve_nodes=True)
        
        h = self.conv1(g, features)
        if self.use_batch_norm:
            h = self.batch_norm1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        
        h = self.conv2(g, h)
        if self.use_batch_norm:
            h = self.batch_norm2(h)
        
        if self.task_type == 'node_classification':
            return F.log_softmax(h, dim=1)
        elif self.task_type == 'edge_prediction':
            g.edata['h'] = h
            return torch.sigmoid(dgl.function.u_dot_v('h', 'h', 'score'))
        elif self.task_type == 'graph_classification':
            g.ndata['h'] = h
            return dgl.mean_nodes(g, 'h')

# Example Usage
def train_gnn_model(graph, features, labels, task_type, train_mask=None, test_mask=None):
    input_dim = features.shape[1]
    hidden_dim = 32
    output_dim = labels.max().item() + 1 if task_type == 'node_classification' else 1
    
    model = GNNWithRegularization(input_dim, hidden_dim, output_dim, task_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    for epoch in range(50):
        model.train()
        logits = model(graph, features)
        
        if task_type == 'node_classification':
            loss = F.nll_loss(logits[train_mask], labels[train_mask])
        elif task_type == 'edge_prediction':
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        elif task_type == 'graph_classification':
            loss = F.binary_cross_entropy(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        if task_type == 'node_classification':
            preds = logits[test_mask].argmax(dim=1)
            acc = (preds == labels[test_mask]).float().mean().item()
            print(f"Test Accuracy: {acc * 100:.2f}%")
        else:
            print("Evaluation for edge or graph tasks can be extended.")
    return model

# Example datasets for node classification, edge prediction, and graph classification
# Replace with specific dataset loading logic
graph = dgl.rand_graph(100, 500)  # Dummy graph
features = torch.randn(100, 10)  # Dummy features
labels = torch.randint(0, 3, (100,))  # Dummy labels
train_mask = torch.rand(100) < 0.8
test_mask = ~train_mask

# Train a node classification model
train_gnn_model(graph, features, labels, 'node_classification', train_mask, test_mask)
