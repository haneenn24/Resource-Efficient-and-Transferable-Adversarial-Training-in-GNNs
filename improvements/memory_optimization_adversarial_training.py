import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import numpy as np
import time
import sys
from tqdm import tqdm
import os
import psutil
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from robust_diffusion.data import prep_graph
from robust_diffusion.train import train_inductive
from robust_diffusion.attacks import create_attack
from robust_diffusion.models import create_model
from robust_diffusion.helper.utils import accuracy


# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_dataset(dataset_name='cora', data_device=0):
    """Load and preprocess the dataset."""
    graph = prep_graph(dataset=dataset_name, data_device=data_device, dataset_root='./data', make_undirected=True)
    attr_orig, adj_orig, labels = graph[:3]
    return attr_orig, adj_orig, labels, graph

def gradient_checkpointing(model, inputs):
    """Applies gradient checkpointing to save memory by recomputing gradients only when needed."""
    return checkpoint(model, inputs)

def apply_pruning(model, amount=0.2):
    """Prunes the model by removing a certain percentage of weights."""
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    
    # Apply pruning with the specified amount
    nn.utils.prune.global_unstructured(
        parameters_to_prune,
        pruning_method=nn.utils.prune.L1Unstructured,
        amount=amount,
    )

def adaptive_adversarial_budgeting(node_degrees, max_budget):
    """Dynamically adjust adversarial perturbation budget based on node degrees."""
    degree_sum = np.sum(node_degrees)
    budgets = node_degrees / degree_sum * max_budget
    return budgets

def measure_memory():
    """Measures current memory usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # Return in MB

def memory_efficient_adversarial_training(model_name, dataset='cora', max_budget=50, pruning_amount=0.2):
    """Train a GNN with memory optimizations such as gradient checkpointing and pruning."""
    
    # Load dataset
    attr_orig, adj_orig, labels, graph = load_dataset(dataset)
    idx_train, idx_val, idx_test = graph[3], graph[4], graph[5]
    
    # Create model
    model_params = {
        'label': model_name,
        'model': model_name,
        'n_filters': 64,
        'dropout': 0.5
    }
    model = create_model(model_params).to(device)
    
    # Apply model pruning (resource efficiency)
    apply_pruning(model, amount=pruning_amount)
    
    # Training loop with gradient checkpointing and adaptive adversarial budgeting
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    # Dynamic adversarial budgeting
    node_degrees = adj_orig.sum(1).cpu().numpy()
    budgets = adaptive_adversarial_budgeting(node_degrees, max_budget)
    
    mem_usage_start = measure_memory()
    time_start = time.time()
    operations_count = 0  # Placeholder for tracking operations (can use FLOPs or other measures)

    for epoch in tqdm(range(100), desc="Training with Memory Optimization"):
        
        # Apply gradient checkpointing during forward pass
        attr_checkpointed = gradient_checkpointing(model, attr_orig)
        
        # Adversarial attack (LR-BCD)
        adversary = create_attack(attack='LRBCD', attr=attr_checkpointed, adj=adj_orig, labels=labels, model=model, idx_attack=idx_train,
                                  device=device, data_device=device)
        adversary.attack(n_perturbations=int(budgets[idx_train].sum()))
        adj_perturbed = adversary.get_modified_adj()

        # Forward pass with perturbed adjacency matrix
        logits = model(attr_checkpointed, adj_perturbed)
        loss = loss_fn(logits[idx_train], labels[idx_train])
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluation on validation set
        model.eval()
        with torch.no_grad():
            val_logits = model(attr_checkpointed, adj_perturbed)
            val_accuracy = accuracy(val_logits.cpu(), labels.cpu(), idx_val)
            print(f"Epoch {epoch}, Validation Accuracy: {val_accuracy}")
        
        # Track operations (approximation)
        operations_count += logits.numel()  # Example: count number of elements processed

    time_end = time.time()
    mem_usage_end = measure_memory()
    
    # Calculate results
    mem_usage_diff = mem_usage_end - mem_usage_start
    time_taken = time_end - time_start
    print(f"Memory Usage: {mem_usage_diff} MB, Time Taken: {time_taken} seconds, Operations: {operations_count}")

    return val_accuracy, mem_usage_diff, time_taken, operations_count

def evaluate_resource_efficiency(model_name, dataset='cora'):
    """Evaluate model performance with and without optimizations."""
    # Memory-Efficient Training
    print("Running memory-efficient adversarial training...")
    accuracy_opt, mem_usage_opt, time_opt, operations_opt = memory_efficient_adversarial_training(model_name, dataset)

    # Standard Training (without optimizations)
    print("Running standard adversarial training for comparison...")
    accuracy_standard, mem_usage_standard, time_standard, operations_standard = memory_efficient_adversarial_training(model_name, dataset, pruning_amount=0)

    # Log and print results
    log_dir = './results/performance/improvements/'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'memory_optimization_results.txt')

    with open(log_file, 'a') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Accuracy (Optimized): {accuracy_opt}, Memory Usage (Optimized): {mem_usage_opt} MB, Time (Optimized): {time_opt} s, Operations (Optimized): {operations_opt}\n")
        f.write(f"Accuracy (Standard): {accuracy_standard}, Memory Usage (Standard): {mem_usage_standard} MB, Time (Standard): {time_standard} s, Operations (Standard): {operations_standard}\n")
        f.write("-------------------------------------------------------------\n")

    print(f"Results logged to {log_file}")

if __name__ == "__main__":
    evaluate_resource_efficiency(model_name='GCN', dataset='cora')
