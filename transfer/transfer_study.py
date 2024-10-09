import torch
import os
from robust_diffusion.data import prep_graph
from robust_diffusion.models import create_model
from robust_diffusion.attacks import create_attack
from robust_diffusion.train import train_inductive
from robust_diffusion.helper.utils import accuracy, calculate_loss

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_dataset(dataset_name='cora', data_device=0):
    """Load and preprocess the dataset (Cora or PubMed)."""
    graph = prep_graph(dataset=dataset_name, data_device=data_device, dataset_root='./data', make_undirected=True)
    attr_orig, adj_orig, labels = graph[:3]
    return attr_orig, adj_orig, labels, graph

def train_gnn_model(model_name, attr_orig, adj_orig, labels, idx_train, idx_val, idx_test):
    """Train a GNN model (GCN, GAT, GPRGNN) on clean data."""
    model_params = {
        'label': model_name,
        'model': model_name,
        'n_filters': 64,
        'dropout': 0.5
    }
    
    model = create_model(model_params).to(device)
    
    # Train cleanly (no adversarial attack)
    train_inductive(model=model, 
                    attr_training=attr_orig, 
                    adj_training=adj_orig, 
                    labels_training=labels, 
                    idx_train=idx_train, 
                    idx_val=idx_val,
                    max_epochs=100)
    
    return model

def generate_adversarial_attack(model, attr_orig, adj_orig, labels, idx_train):
    """Generate adversarial perturbations using the LR-BCD attack."""
    adversary = create_attack(attack='LRBCD', attr=attr_orig, adj=adj_orig, labels=labels, model=model, idx_attack=idx_train,
                              device=device, data_device=device)
    
    # Perform the attack (perturb the graph)
    adversary.attack(n_perturbations=50)
    
    # Get the perturbed adjacency matrix (adversarial graph)
    adj_perturbed = adversary.get_modified_adj()
    
    return adj_perturbed

def evaluate_gnn_model(model, attr_orig, adj_orig, adj_perturbed, labels, idx_test):
    """Evaluate the GNN model on both clean and adversarial graphs."""
    model.eval()
    logits_clean = model(attr_orig, adj_orig)  # Clean graph accuracy
    logits_adversarial = model(attr_orig, adj_perturbed)  # Adversarial graph accuracy
    
    accuracy_clean = accuracy(logits_clean.cpu(), labels.cpu(), idx_test)
    accuracy_adversarial = accuracy(logits_adversarial.cpu(), labels.cpu(), idx_test)
    
    return accuracy_clean, accuracy_adversarial

def log_results(file_path, content):
    """Log the results into the specified file."""
    with open(file_path, 'a') as f:
        f.write(content + "\n")

def evaluate_transferability(source_model_name, target_model_name, dataset='cora'):
    """Evaluate the transferability of adversarial attacks between different GNN models."""
    
    # Ensure log directory exists
    log_dir = './results/performance/transferability/'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'transferability_log.txt')

    # Step 1: Load the dataset
    attr_orig, adj_orig, labels, graph = load_dataset(dataset)
    idx_train, idx_val, idx_test = graph[3], graph[4], graph[5]
    
    # Step 2: Train the source model (e.g., GCN)
    print(f"Training {source_model_name} on clean data...")
    source_model = train_gnn_model(source_model_name, attr_orig, adj_orig, labels, idx_train, idx_val, idx_test)
    
    # Step 3: Generate the adversarial attack on the source model (GCN)
    print(f"Generating adversarial attack using {source_model_name}...")
    adj_perturbed = generate_adversarial_attack(source_model, attr_orig, adj_orig, labels, idx_train)
    
    # Step 4: Evaluate the source model on both clean and adversarial graphs
    print(f"Evaluating {source_model_name} on both clean and adversarial graphs...")
    source_accuracy_clean, source_accuracy_adversarial = evaluate_gnn_model(source_model, attr_orig, adj_orig, adj_perturbed, labels, idx_test)
    
    print(f"Source Model {source_model_name} Clean Accuracy: {source_accuracy_clean}")
    print(f"Source Model {source_model_name} Adversarial Accuracy: {source_accuracy_adversarial}")
    
    # Step 5: Train the target model (e.g., GAT) cleanly
    print(f"Training {target_model_name} on clean data...")
    target_model = train_gnn_model(target_model_name, attr_orig, adj_orig, labels, idx_train, idx_val, idx_test)
    
    # Step 6: Evaluate the target model on the adversarial graph (perturbed by GCN attack)
    print(f"Evaluating {target_model_name} on the adversarial graph (perturbed by {source_model_name})...")
    target_accuracy_clean, target_accuracy_adversarial = evaluate_gnn_model(target_model, attr_orig, adj_orig, adj_perturbed, labels, idx_test)
    
    print(f"Clean accuracy on {target_model_name}: {target_accuracy_clean}")
    print(f"Adversarial accuracy on {target_model_name}: {target_accuracy_adversarial}")
    
    # Calculate transferability rate
    degradation_source = source_accuracy_clean - source_accuracy_adversarial
    degradation_target = target_accuracy_clean - target_accuracy_adversarial
    
    transferability_rate = (degradation_target / degradation_source) * 100
    print(f"Transferability Rate from {source_model_name} to {target_model_name}: {transferability_rate:.2f}%")

    # Log the results
    log_content = (
        f"Source Model: {source_model_name}\n"
        f"Target Model: {target_model_name}\n"
        f"Source Clean Accuracy: {source_accuracy_clean}\n"
        f"Source Adversarial Accuracy: {source_accuracy_adversarial}\n"
        f"Target Clean Accuracy: {target_accuracy_clean}\n"
        f"Target Adversarial Accuracy: {target_accuracy_adversarial}\n"
        f"Transferability Rate: {transferability_rate:.2f}%\n"
        "-------------------------------------------------------------"
    )
    log_results(log_file, log_content)

if __name__ == "__main__":
    # Example usage: Transfer attack from GCN to GAT
    evaluate_transferability(source_model_name='GCN', target_model_name='GAT', dataset='cora')
