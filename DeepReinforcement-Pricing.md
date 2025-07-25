# Deep Reinforcement for Exotic Pricing - Summary

presentation : [DeepReinforcement-Pricing.pdf](./pdf/DeepReinforcement-Pricing.pdf)

This document presents Natixis' approach to pricing exotic derivatives using deep reinforcement learning, drawing analogies from computational biology and advanced neural network techniques.

## Key Concepts
1. **Protein Folding Analogy**  
   - Energy minimization principles from protein folding (e.g., AlphaFold) inspire optimization methods for pricing models.  
   - Protein misfolding diseases (Alzheimer’s, Parkinson’s) parallel the challenges of local minima in optimization.  

2. **Neural Network Foundations**  
   - **Architecture**: Deep Neural Networks (DNNs) with input, hidden, and output layers.  
   - **Neuron Mechanics**: Weighted inputs → Sum → Activation Function → Output.  
   - **Residual Networks (ResNets)** solve degradation issues in deep networks via identity mapping.  

3. **Reinforcement Learning (RL) Framework**  
   - **MDP Components**: Agent, Environment, Actions, Rewards.  
   - **Algorithms**:  
     - Q-Learning: Off-policy value iteration.  
     - SARSA: On-policy TD learning with deep targets.  
     - **Multi-Armed Bandit**: Balances exploration vs. exploitation using ε-greedy/UCB strategies.  

4. **Pricing Challenge**  
   - **Goal**: Price exotic derivatives (e.g., Yeti-Phoenix) with 23 parameters.  
   - **Data**: 1M training samples, 400K test samples.  
   - **Risk Metric**: MSE between predicted and benchmark prices.  

---

## Methodologies
### 1. **Dynamic Network Construction**
- **Genetic Algorithm**: Dynamically adds layers/neurons to escape local minima.  
- **Neural Increments**: Asymptotic approximation of network growth:  
  \[Q\{N + \sum b_i\} ≈ Q\{N\} + \sum (Q\{N + b_i\} - Q\{N\})\]

### 2. **Reinforcement Optimizations**
- **Layer Selection**: Treats each layer as a "bandit arm" using:  
  - **ε-Greedy**: Random exploration (prob = ε) + greedy exploitation (prob = 1-ε).  
  - **UCB (Upper Confidence Bound)**: Optimistic exploration with Hoeffding’s inequality:  
    \[a_t = \text{argmax}_a \left( \hat{Q}_t(a) + \sqrt{\frac{2\log t}{N_t(a)}} \right)\]

### 3. **Mathematical Guarantees**
- **Asymptotic Convexity**: For λ-regularized ReLU networks, ∃ unique global optimum.  
- **Monotonic Upgrades**: Local optima can always be escaped by adding neurons to specific layers.  

---

## Results
| **Benchmark** | Error ≤1 BP (97% cases) | Challenge Score |
|---------------|-------------------------|----------------|
| Benchmark 1   | 96%                     | 0.1146         |
| Benchmark 2   | 97%                     | 0.053          |

**Key Insight**: UCB outperforms ε-greedy in asymptotic regret (logarithmic bound).  

---

## Advanced Insights
1. **Information Theory**  
   - **Mutual Information**: Measures layer interactions via Kullback-Leibler divergence:  
     \[I(X;Y) = D_{KL}[p(x,y) \| p(x)p(y)]\]  
   - **Information Bottleneck**: Balances compression (min \(I(X;T)\)) and prediction (max \(I(T;Y)\)).  

2. **ResNet Success**  
   - Dominates ILSVRC 2015 (top-5 error: 3.57%).  
   - Enables stable training of 1000+ layer networks.  

3. **Hyperparameter Optimization**  
   - **SMBO (Sequential Model-Based Optimization)**: Uses Gaussian process surrogates.  
   - **TPE (Tree-structured Parzen Estimator)**: Models \(p(x|y)\) for efficient EI (Expected Improvement) optimization.  

---

## Conclusion
1. Pricing requires **high-dimensional interpolation** of topological structures.  
2. **Reinforcement-driven dynamic learning** handles local optima efficiently.  
3. **Mutual information** and **energy analysis** guide layer design and generalization.  
4. **Exploration strategies** (UCB > ε-greedy) ensure asymptotic optimality.  

> "Failure is an option here. If things are not failing, you are not innovating enough."  
> — Olivier Croissant, Natixis  
