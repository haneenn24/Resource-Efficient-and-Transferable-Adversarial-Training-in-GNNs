# Resource-Efficient Adversarial Training and Transferability of Adversarial Attacks in Graph Neural Networks

## Overview

Welcome to the GitHub repository for our project on **Resource-Efficient Adversarial Training and Transferability of Adversarial Attacks in Graph Neural Networks (GNNs)**! This project builds upon the insights from the paper *“Adversarial Training for Graph Neural Networks: Pitfalls, Solutions, and New Directions.”* By enhancing the efficiency of adversarial training techniques and exploring cross-model and cross-task attack transferability, this research advances both the robustness and adaptability of GNNs across different architectures and learning settings.

Our project introduces optimizations that maintain high robustness while reducing computational costs, with a special focus on improving the **Local Robustness Bounded by Consensus Distance (LR-BCD)** method. Additionally, we analyze the **transferability** of adversarial attacks across various GNN models and tasks, shedding light on how vulnerabilities in one model may affect others.

## Key Contributions

### 1. Resource-Efficient Adversarial Training
Our work enhances the efficiency of adversarial training in GNNs by:
- **Reducing Perturbation Frequency**: Perturbations are applied every alternate epoch, leading to significant reductions in memory usage and training time without compromising robustness.
- **Monte Carlo Sampling**: Implemented as an approximation technique to lower computational overhead in perturbation calculations, while preserving robustness through controlled randomness.

**Baseline Results**: We establish performance benchmarks using the **Cora, Citeseer, and arXiv datasets** based on previous findings:
- **Cora**: LR-BCD achieves a clean accuracy of 83.3% and an adversarial accuracy of 76.8%.
- **Citeseer**: LR-BCD outperforms PR-BCD in both clean and adversarial accuracy.
- **arXiv**: Original LR-BCD reports a memory usage of 20 GB and 10 seconds per epoch, which serves as our baseline for efficiency improvements.

### 2. Transferability of Adversarial Attacks Across GNN Architectures and Tasks
We extend our study to assess the **transferability** of adversarial attacks in GNNs by exploring:
- **Cross-Model Transferability**: Adversarial examples generated for one GNN model (e.g., GCN) are tested on others (e.g., GAT and GPRGNN) to evaluate shared vulnerabilities.
- **Cross-Task Transferability**: We test attacks across tasks like **node classification, edge prediction, and graph-level classification** to understand task-specific susceptibilities.
- **Transductive vs. Inductive Learning**: Both learning modes are analyzed for attack transferability to reveal how access to complete graph structures impacts robustness.

### 3. Transferability Rate Improvements with Regularization
Our findings reveal that **batch normalization** and other regularization techniques improve robustness by:
- **Reducing Overfitting**: Generalizes model representations, making adversarial perturbations more transferable.
- **Creating Stable Representations**: Perturbations crafted on regularized models tend to generalize better across models, which improves cross-model transferability.

## Project Structure

- **`lrbcd.py`**: Contains the implementation of the **LR-BCD** method with our modifications, including reduced perturbation frequency and Monte Carlo sampling.
- **`transferability_tests.py`**: Implements cross-model and cross-task transferability tests, covering both transductive and inductive learning modes.
- **`regularization_effects.py`**: Analyzes the effects of regularization on transferability rates and overall robustness.
- **Datasets**: Scripts for loading and processing **Cora, Citeseer, and arXiv** datasets.
- **Results**: Contains empirical results on memory usage, time per epoch, and adversarial accuracy to evaluate the effectiveness of our resource-efficient modifications.

## Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/haneenn24/Resource-Efficient-and-Transferable-Adversarial-Training-in-GNNs.git
   cd Resource-Efficient-and-Transferable-Adversarial-Training-in-GNNs
   ```

2. **Install Dependencies**:
   Ensure you have Python and necessary libraries installed. Dependencies are listed in `requirements.txt`.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Adversarial Training**:
   To run adversarial training using optimized LR-BCD:
   ```bash
   python lrbcd.py
   ```

4. **Transferability Testing**:
   Evaluate the transferability of adversarial attacks across GNN architectures and tasks:
   ```bash
   python transferability_tests.py
   ```

5. **Analyze Regularization Effects**:
   To explore regularization impacts on transferability:
   ```bash
   python regularization_effects.py
   ```

## Results Summary

Our optimizations demonstrate significant resource savings with minimal trade-offs in robustness:
- **Time per Epoch**: Decreased time per epoch
- **Adversarial Robustness**: Robustness levels remain consistent with the baseline, validating the effectiveness of reduced perturbation frequency and Monte Carlo sampling.

Transferability results indicate:
- **Enhanced Cross-Model Attack Success Rates**: Models with similar architectures (e.g., GCN and GAT) show higher vulnerability to shared attacks.
- **Improved Generalization in Regularized Models**: Regularized models exhibit higher transferability rates, suggesting that regularization enhances the attack’s effectiveness across models.

## Future Work

This project opens doors for further exploration in:
- **Adaptive Regularization**: Experimenting with adaptive regularization techniques to dynamically adjust robustness and transferability based on task needs.
- **Advanced Approximation Methods**: Testing other approximation techniques for adversarial perturbations to find the most computationally efficient and robust approach.

## Contributing

Contributions are welcome! Feel free to open issues, submit pull requests, or discuss improvements. Let's work together to push the boundaries of adversarial robustness in GNNs.

## Acknowledgments

Special thanks to the authors of *“Adversarial Training for Graph Neural Networks: Pitfalls, Solutions, and New Directions”* for inspiring this project. This work also leverages libraries such as **PyTorch Geometric** and **DGL** for efficient GNN implementations.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

Explore the boundaries of adversarial robustness in GNNs with us. Start cloning, testing, and contributing today!
