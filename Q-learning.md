# Q-Learning and Policy Optimization Summary

## 1. Bellman Equation to Q-Learning
### Core Equations
- **Bellman Equation**:  
  `V^π(s) ← ∑ₛ' P(s'|s, π(s))[R(s, π(s), s') + γV^π(s')]`
- **Optimal Q-Function**:  
  `Q*(s,a) ← ∑ₛ' P(s'|s,a)[R(s,a,s') + γ maxₐ' Q*(s',a')]`
- **Value-Policy Relationship**:  
  `V*(s) = maxₐ Q*(s,a)`

## 2. Value and Policy Iteration
### Methods
- **Value Iteration** (Non-linear):  
  `Vₖ*(s) ← maxₐ ∑ₛ' P(s'|s,a)[R(s,a,s') + γVₖ₋₁*(s')]`
- **Policy Iteration**:  
  `Vₖ^π(s) ← ∑ₛ' P(s'|s,π(s))[R(s,π(s),s') + γVₖ₋₁^π(s')]`
- **Stochastic Policy**:  
  `Vₖ^π(s) ← ∑ₛ' ∑ₐ π(a|s)P(s'|s,a)[R(s,a,s') + γVₖ₋₁^π(s')]`

## 3. Q-Learning Fundamentals
### Algorithm
`Qₖ(s,a) ← ∑ₛ' P(s'|s,a)[R(s,a,s') + γ maxₐ' Qₖ₋₁(s',a')]`

### Convergence Conditions
1. Explore states/actions sufficiently
2. Learning rate α must satisfy:  
   `∑αₜ(s,a) = ∞` and `∑αₜ²(s,a) < ∞`

### On-Policy vs Off-Policy
| **Off-Policy (Q-Learning)** | **On-Policy (SARSA)** |
|-----------------------------|------------------------|
| Uses `max Q` for next state | Follows current policy |
| More exploratory | More conservative |

## 4. Q-Learning Improvements
### Key Enhancements
1. **Double DQN**:  
   - Reduces overestimation bias using two networks  
   - θ selects action, θ* evaluates it
2. **Prioritized Experience Replay**:  
   - Replays transitions with high Bellman error:  
     `|r + γ maxₐ' Q(s',a';θ⁻) - Q(s,a;θ)|`
3. **Dueling DQN**:  
   - Separates value and advantage:  
     `Q(s,a) = V(s) + A(s,a) - 1/|A| ∑ₐ' A(s,a')`
4. **Noisy Nets**:  
   - Adds parameter noise:  
     `y = (μʷ + σʷ⊙εʷ)x + μᵇ + σᵇ⊙εᵇ`

### Performance Gains
| Method          | Median Score Improvement |
|-----------------|--------------------------|
| Double DQN      | 115% → 117%              |
| Prioritized REP | 111% → 128%              |
| Dueling DQN     | Up to 591.9%             |

## 5. Policy Optimization
### Policy Gradient Theorem
`∇θU(θ) ≈ 1/m ∑ᵢ ∇θ logP(τ⁽ⁱ⁾;θ) R(τ⁽ⁱ⁾)`

### Policy Types
| **Softmax Policy**               | **Gaussian Policy**             |
|----------------------------------|---------------------------------|
| `πθ(s,a) ∝ e^{φ(s,a)ᵀθ}`        | `a ∼ N(φ(s)ᵀθ, σ²)`            |
| Score: `φ(s,a) - 𝔼[φ]`           | Score: `(a-μ(s))φ(s)/σ²`       |

### Natural Policy Gradient
`∇θⁿᵃᵗJ(θ) = Gθ⁻¹ ∇θJ(θ)`  
Where `Gθ` is the Fisher information matrix.

## 6. Actor-Critic Methods
### Algorithm
```python
Initialize s, θ, w
for each step:
    Sample a ∼ πθ(s)
    Observe r, s'
    Sample a' ∼ πθ(s')
    δ = r + γQw(s',a') - Qw(s,a)
    θ ← θ + α∇θ logπθ(s,a) Qw(s,a)
    w ← w + βδφ(s,a)
    s, a ← s', a'
