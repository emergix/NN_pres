# Pricing Exotics with Neural Networks - Summary

presentation : [Exotics_NN_2021.pdf](./pdf/Exotics_NN_2021.pdf)

## Overview
This document from Natixis explores using neural networks (NN) to price complex exotic financial derivatives ("AutoCall Yeti-Phoenix"), addressing challenges in traditional Monte Carlo pricing techniques.

## Key Concepts

### AutoCall Yeti-Phoenix
- **Complex payoff structure** combining:
  - Bonus coupons (BO)
  - Yeti coupons (YE) 
  - Phoenix coupons (PH)
  - Down-and-in put options
- Requires Monte Carlo simulation for pricing
- Underlying: `P_l = Min(S₁(t_i), S₂(t_i), S₃(t_i))`

### Challenges in Pricing
1. **High dimensionality**: >200 input parameters
2. **Strong non-linearity**: Highly non-convex optimization landscape
3. **Accuracy requirements**: 0.01% precision
4. **Local minima**: Many suboptimal solutions during training

### Fast-Pricing Techniques
| Technique | Purpose | Key Implementation |
|-----------|---------|---------------------|
| **Convolutional First Layer** | Feature extraction | 1D Conv layers with ReLU activation |
| **Stochastic Data Phasing** | Prevent overfitting | `train_test_split()` with test_size=0.3 |
| **Learning Rate Oscillation** | Escape local minima | `ReduceLROnPlateau` with dynamic adjustments |
| **BestChecking Oscillation** | Maintain progress | Restart training from best checkpoint |
| **Final Debiasing** | Improve accuracy | Post-training calibration |
| **Genetic Programming** | Architecture optimization | Network mutations + selection |

### Genetic Programming Details
- **Mutation types**:
  - Type 0: Add neurons to layers
  - Type 1/2: Add skip-connected layers
- **Selection strategy**:
  - Maintain candidate population
  - Remove low performers aggressively
- **Exploration vs Exploitation**:
  - ε-greedy policy selection (e.g., ε=0.1)
  - Multi-armed bandit for mutation prioritization

## Results
### Performance Metrics
| Model | L1 Error | 95% Quantile (bp) | 99% Quantile (bp) |
|-------|----------|-------------------|-------------------|
| Initial | 0.0105 | 3.46 | 6.81 |
| Genetic Optimized | 0.0104 | 3.39 | 6.43 |
| CatBoost | 0.0465 | 13.03 | 21.7 |

### Scaling Performance
| Dimensions | Dataset Size | L2 Error | Compute Resources |
|------------|--------------|----------|-------------------|
| 23 | 1M | 6.8e⁻⁸ | 4×Maxwell GPUs (3 days) |
| 191 | 7M | 3.5e⁻⁷ | 16×V100 GPUs (3 days) |
| 61 | 1M | 7.6e⁻⁷ | 4×Maxwell GPUs (3 days) |

## Conclusion
Neural networks enable high-accuracy pricing (0.05% error) for exotic derivatives with:
1. Hybrid exploration/exploitation strategies
2. Genetic architecture optimization
3. Efficient handling of high-dimensional inputs (>190 dimensions)
4. GPU-accelerated training (3 days for 7M samples)

> **Key Insight**: Combining convolutional layers, dynamic learning rates, and genetic algorithms overcomes traditional limitations in financial derivative pricing.
