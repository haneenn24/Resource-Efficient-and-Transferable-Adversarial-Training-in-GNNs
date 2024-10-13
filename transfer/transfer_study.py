import torch
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import sys
import pandas as pd
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_sparse import SparseTensor
from torch.cuda.amp import autocast, GradScaler  # For mixed precision training
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.nn import DataParallel  # For multi-GPU support

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from robust_diffusion.data import prep_graph
from robust_diffusion.models import create_model
from robust_diffusion.models.gat_weighted import GAT, WeightedGATConv
from robust_diffusion.attacks import create_attack
from robust_diffusion.train import train_inductive
from robust_diffusion.helper.utils import accuracy, calculate_loss

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = GradScaler()  # Gradient Scaler for mixed precision training

def load_dataset(dataset_name='cora', data_device=0):
    """Load and preprocess the dataset (Cora or PubMed)."""
    graph = prep_graph(name=dataset_name, device=data_device, dataset_root='./data', make_undirected=True)
    attr_orig, adj_orig, labels = graph[:3]
    return attr_orig, adj_orig, labels, graph

def load_cora_data(content_path, cites_path):
    # Load the content file
    content = pd.read_csv(content_path, sep='\t', header=None)
    cites = pd.read_csv(cites_path, sep='\t', header=None)

    # Processing the node features and labels
    node_ids = content[0].values
    features = content.iloc[:, 1:-1].values  # Assuming features are in the middle columns
    labels = pd.Categorical(content.iloc[:, -1].values).codes  # Convert class labels to integers

    # Create a mapping from node ID to index
    node_id_map = {node_id: i for i, node_id in enumerate(node_ids)}

    # Processing the citations (edges)
    edge_index = []
    for _, (src, dest) in cites.iterrows():
        if src in node_id_map and dest in node_id_map:
            edge_index.append([node_id_map[src], node_id_map[dest]])

    edge_index = torch.tensor(edge_index).t().contiguous()

    # Convert features and labels to PyTorch tensors
    features = torch.tensor(features, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)

    return Data(x=features, edge_index=edge_index, y=labels)

def train_gnn_model(model_name, attr_orig, adj_orig, labels, idx_train, idx_val, idx_test, batch_size=8, accumulation_steps=4):
    """Train a GNN model (GCN, GAT) using mini-batches and multi-GPU support with gradient accumulation."""

    print(f"Initial adj_orig type: {type(adj_orig)}")

    # Convert SparseTensor to edge_index and edge_weight
    if isinstance(adj_orig, SparseTensor):
        print("Converting SparseTensor to dense edge_index and edge_weight")
        row, col, edge_weight = adj_orig.coo()
        edge_index = torch.stack([row, col], dim=0)
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    else:
        # If adj_orig is a tuple, extract edge_index and edge_weight
        edge_index, edge_weight = adj_orig
        if edge_weight is None:
            print("Edge weight is None, setting it to ones.")
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

    # Ensure edge_index has two dimensions
    print(f"Edge Index shape: {edge_index.shape}")
    print(f"Edge Weight shape: {edge_weight.shape if edge_weight is not None else 'None'}")

    if edge_index.size(0) != 2:
        raise ValueError(f"edge_index should have two dimensions, but got {edge_index.size(0)}")

    max_node_index = edge_index.max().item()
    num_nodes = attr_orig.size(0)
    print(f"Max node index in edge_index: {max_node_index}, Number of nodes: {num_nodes}")

    if max_node_index >= num_nodes:
        raise ValueError(f"Edge index contains out-of-bounds node index: {max_node_index} >= {num_nodes}")

    # Create PyG Data object for graph
    print("Creating PyG Data object")
    data = Data(x=attr_orig, edge_index=edge_index, edge_attr=edge_weight, y=labels)

    print("Creating DataLoader for batching")
    train_loader = DataLoader([data], batch_size=batch_size, shuffle=True)

    # Define model parameters
    n_features = attr_orig.shape[1]
    n_classes = labels.max().item() + 1

    print(f"Creating GAT model with {n_features} features and {n_classes} classes")
    model = GAT(n_features=n_features, n_classes=n_classes).to(device)
    model = DataParallel(model, device_ids=[0, 1, 2, 3])  # Multi-GPU support

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()

    for epoch in range(10):
        total_loss = 0
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            print(f"Processing batch {i + 1}")
            batch = batch.to(device)

            # Ensure edge_index has correct shape and fix it if necessary
            print(f"Batch edge_index shape: {batch.edge_index.shape}")
            if batch.edge_index.size(0) != 2:
                print(f"Reshaping edge_index: {batch.edge_index.shape}")
                batch.edge_index = batch.edge_index.view(2, -1)  # Reshape to [2, num_edges]

            # Forward pass through the model
            logits = model(batch.x, (batch.edge_index, batch.edge_attr))

            loss = criterion(logits[idx_train], batch.y[idx_train])
            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    return model



def evaluate_gnn_model(model, attr_orig, adj_orig, adj_perturbed, labels, idx_test):
    """Evaluate the GNN model on both clean and adversarial graphs."""
    model.eval()
    logits_clean = model(attr_orig.to(device), adj_orig.to(device))  # Clean graph accuracy
    logits_adversarial = model(attr_orig.to(device), adj_perturbed.to(device))  # Adversarial graph accuracy
    
    accuracy_clean = accuracy(logits_clean.cpu(), labels.cpu(), idx_test)
    accuracy_adversarial = accuracy(logits_adversarial.cpu(), labels.cpu(), idx_test)
    
    return accuracy_clean, accuracy_adversarial

def log_results(file_path, content):
    """Log the results into the specified file."""
    with open(file_path, 'a') as f:
        f.write(content + "\n")

def generate_adversarial_attack(model, attr_orig, adj_orig, labels, idx_train, make_undirected=True, binary_attr=False):
    """Generate adversarial perturbations using the LR-BCD attack."""
    adversary = create_attack(attack='LRBCD', attr=attr_orig, adj=adj_orig, labels=labels, model=model, idx_attack=idx_train, device=device, data_device=device, make_undirected=make_undirected, binary_attr=binary_attr)
    
    adversary.attack(n_perturbations=50)
    
    adj_perturbed = adversary.get_modified_adj().to(device)  # Move perturbed adj to GPU
    
    return adj_perturbed

def evaluate_transferability(source_model_name, target_model_name, dataset='cora'):
    torch.cuda.empty_cache()

    # Load the Cora dataset using the newly integrated load_cora_data function
    cora_data = load_cora_data('./data/cora/cora.content', './data/cora/cora.cites')

    # Use cora_data to get the node features (attr_orig), edge information (adj_orig), and labels
    attr_orig = cora_data.x
    adj_orig = (cora_data.edge_index, None)  # No edge weights in Cora by default
    labels = cora_data.y

    # Assuming idx_train, idx_val, idx_test are predefined splits, if not, define them
    # Here you can use any split logic, for simplicity let's assume you have them as:
    idx_train = torch.arange(0, int(0.6 * len(labels)))
    idx_val = torch.arange(int(0.6 * len(labels)), int(0.8 * len(labels)))
    idx_test = torch.arange(int(0.8 * len(labels)), len(labels))

    # Step 2: Train the source model (e.g., GCN)
    source_model = train_gnn_model(source_model_name, attr_orig, adj_orig, labels, idx_train, idx_val, idx_test)

    # Step 3: Generate adversarial attack on the source model
    adj_perturbed = generate_adversarial_attack(source_model, attr_orig, adj_orig, labels, idx_train)

    # Step 4: Evaluate source model on clean and adversarial graphs
    source_accuracy_clean, source_accuracy_adversarial = evaluate_gnn_model(source_model, attr_orig, adj_orig, adj_perturbed, labels, idx_test)

    # Step 5: Train the target model (e.g., GAT)
    target_model = train_gnn_model(target_model_name, attr_orig, adj_orig, labels, idx_train, idx_val, idx_test)

    # Step 6: Evaluate target model on the adversarial graph (perturbed by GCN attack)
    target_accuracy_clean, target_accuracy_adversarial = evaluate_gnn_model(target_model, attr_orig, adj_orig, adj_perturbed, labels, idx_test)

    print(f"Target Model {target_model_name} Clean Accuracy: {target_accuracy_clean}")
    print(f"Target Model {target_model_name} Adversarial Accuracy: {target_accuracy_adversarial}")

    degradation_source = source_accuracy_clean - source_accuracy_adversarial
    degradation_target = target_accuracy_clean - target_accuracy_adversarial

    transferability_rate = (degradation_target / degradation_source) * 100
    print(f"Transferability Rate from {source_model_name} to {target_model_name}: {transferability_rate:.2f}%")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    evaluate_transferability(source_model_name='GCN', target_model_name='GAT', dataset='cora')
