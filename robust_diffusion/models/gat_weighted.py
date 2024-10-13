import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_sparse import SparseTensor
import torch.cuda.amp as amp

class WeightedGATConv(GATConv):
    """Extended GAT to allow for weighted edges."""
    
    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0.0, **kwargs):
        super(WeightedGATConv, self).__init__(in_channels, out_channels, heads=heads, concat=concat, **kwargs)
        self.negative_slope = negative_slope
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        # Ensure edge_index has correct shape
        if edge_index.size(0) != 2:
            raise ValueError(f"edge_index should have two dimensions (2, num_edges), but got {edge_index.size(0)}")

        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

        # Call the base class's forward method
        return super(WeightedGATConv, self).forward(x, edge_index)

    def message(self, x_j, alpha_j, edge_weight=None):
        # Apply LeakyReLU activation to attention coefficients
        alpha_j = F.leaky_relu(alpha_j, self.negative_slope)

        # Apply edge weights to the attention coefficients if edge weights exist
        if edge_weight is not None:
            alpha_j = alpha_j * edge_weight.view(-1, 1)

        return x_j * alpha_j

class GAT(torch.nn.Module):
    """GAT that supports weighted edges and AMP support for memory optimization."""
    
    def __init__(self, n_features: int, n_classes: int, hidden_dim: int = 16, dropout=0.5, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.conv1 = WeightedGATConv(in_channels=n_features, out_channels=hidden_dim, heads=1, add_self_loops=True)
        self.conv2 = WeightedGATConv(in_channels=hidden_dim, out_channels=n_classes, add_self_loops=True)

    def forward(self, data, adj):
        # Ensure adj is properly batched
        if isinstance(adj, SparseTensor):
            row, col, edge_weight = adj.coo()
            edge_index = torch.stack([row, col], dim=0)
        elif isinstance(adj, tuple):
            edge_index, edge_weight = adj
        else:
            raise ValueError(f"Unexpected adj type: {type(adj)}")

        # Ensure edge_weight is not None
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

        # Fixing single-dimensional edge_index issue
        if edge_index.ndim == 1:
            edge_index = edge_index.unsqueeze(0)

        # Forward pass through GAT layers
        data = self.conv1(data, edge_index, edge_weight).relu()
        data = F.dropout(data, p=self.dropout, training=self.training)
        data = self.conv2(data, edge_index, edge_weight)

        return data



