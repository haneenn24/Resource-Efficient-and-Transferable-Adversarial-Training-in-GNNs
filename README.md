# **Resource-Efficient and Transferable Adversarial Training in GNNs**

This project builds on the foundational work from the paper *Adversarial Training for Graph Neural Networks: Pitfalls, Solutions, and New Directions*. It focuses on implementing resource-efficient adversarial training and studying the transferability of attacks across different Graph Neural Network (GNN) architectures.

---

## **Project Structure**

- **`data/`**: Contains datasets used for training and testing GNNs (e.g., Cora, PubMed).
- **`config/`**: YAML configuration files for different experiments (e.g., adversarial training and attack evasion).
  - **`attack_evasion_global_direct/`**: Configuration files for global attack evasion experiments.
  - **`train/`**: Configuration files for adversarial training using LR-BCD and PGD.
- **`experiments/`**: Scripts for running various experiments such as inductive and transductive adversarial training, and global attack evaluations.
  - **`experiment_adv_train_inductive.py`**: Code for inductive adversarial training.
  - **`experiment_adv_train_transductive.py`**: Code for transductive adversarial training.
  - **`experiment_global_attack_direct.py`**: Global attack code to evaluate robustness.
- **`robust_diffusion/`**: Contains all code related to robust diffusion methods applied to GNNs.
  - **`attacks/`**: Implementation of adversarial attack methods such as LR-BCD.
  - **`helper/`**: Helper functions required for robust diffusion or attack methods.
  - **`models/`**: GNN models incorporating robust diffusion techniques.
  - **`projections/`**: Methods for projecting node features in diffusion processes.
  - **`aggregation.py`**: Aggregation methods for node features during robust diffusion.
  - **`data.py`**: Handles dataset loading and preprocessing.
  - **`train.py`**: Main training script for GNN models using robust diffusion.
- **`transfer/`**: Script to study the transferability of adversarial attacks between different GNN architectures.
  - **`transfer_study.py`**: Script for analyzing attack transferability across models.
- **`improvements/`**: Contains code for implementing resource-efficient improvements (e.g., memory-saving techniques).
  - **`memory_optimizations.py`**: Implements memory-efficient techniques for adversarial training.
- **`results/`**: Stores experiment logs and performance metrics.
  - **`log.txt`**: General log file for experiments.
  - **`performance/`**: Directory for saving performance evaluation results.
- **`requirements.txt`**: Lists the dependencies required for running the project.

---

## **How to Run the Project**

### 1. **Install Dependencies**
To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

### 2. **Running Adversarial Training**
To perform inductive adversarial training using LR-BCD, execute the following command:

```bash
seml [mongodb-collection-name] add config/train/adv_ind_lrbcd.yaml start --local
```

This command runs adversarial training locally. The results will be stored in the **`results/`** directory.

### 3. **Studying Attack Transferability**
To test the transferability of adversarial attacks between GNN architectures, run the script in the **`transfer/`** folder:

```bash
python transfer/transfer_study.py --source-model GCN --target-model GAT --dataset Cora
```

This command tests whether an attack generated on a GCN model can transfer successfully to a GAT model using the Cora dataset.

### 4. **Resource-Efficient Adversarial Training**
For memory-optimized adversarial training, execute the script in the **`improvements/`** folder:

```bash
python improvements/memory_optimizations.py --config config/train/adv_ind_lrbcd.yaml
```

This script applies techniques like gradient checkpointing to reduce memory usage during the adversarial training process.

---

## **Reference to Previous Work**
This project builds upon the foundational work from the paper *Adversarial Training for Graph Neural Networks: Pitfalls, Solutions, and New Directions*. The methods, including adversarial training with LR-BCD and robust diffusion techniques, have been adapted and extended for resource-efficient implementation and to explore the transferability of attacks across different GNN architectures.

The code from the original repository has been referenced and modified where necessary to suit the objectives of this project. We acknowledge the contributions of the authors and encourage referencing their work in any further developments or publications based on this project.

```
@inproceedings{
    gosch2023adversarial,
    title={Adversarial Training for Graph Neural Networks: Pitfalls, Solutions, and New Directions},
    author={Lukas Gosch and Simon Geisler and Daniel Sturm and Bertrand Charpentier and Daniel Z{\"u}gner and Stephan G{\"u}nnemann},
    booktitle={NeurIPS},
    year={2023}
}
```
