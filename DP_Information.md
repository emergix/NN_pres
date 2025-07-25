# Deep Learning: The End of the Black Box Era - Summary

This document presents a comprehensive framework for understanding deep neural networks (DNNs) through theoretical foundations and practical applications. Below is a structured summary of key concepts:

[AGI_annexes.pdf](./pdf/AGI_Natixis-annexes.pdf)

## 1. Introduction & Core Questions
- **Fundamental challenges**: 
  - Why do DNNs generalize well despite overparameterization?
  - How to determine optimal architecture (layers, width, connectivity)?
  - How to interpret neurons/layers and monitor learning?
- **Stochastic dynamics**: 
  - SGD converges to Gibbs distribution (max entropy)
  - Information compression occurs during learning
  - Hidden layers diffuse noise across hierarchies
  - Local optima suffice; global optimum unnecessary

## 2. Theoretical Foundations
### Universal Approximation
- Shallow (1 hidden layer) and deep networks are universal approximators
- **Curse of dimensionality**: Parameter count scales exponentially with input dimension
- **Hierarchical Compositionality (HLC)**:
  - Deep networks break the curse for compositional functions
  - Parameter complexity: `O(d·ε⁻²)` vs `O(ε⁻ᵈ)` for shallow nets

### Information Theory
- **Mutual Information (MI)**:
  `I(X;Y) = Dₖₗ[p(x,y)||p(x)p(y)]`
- **Properties**: 
  - Data Processing Inequality
  - Reparametrization Invariance
- **Measurement Challenges**:
  - MI estimation sensitive to binning/noise/parametric assumptions
  - Not robust to invertible transformations

### Variational Inference
- Evidence Lower Bound (ELBO) links likelihood and KL divergence:
  `ELBO = E_q[log p(z,x)] - E_q[log q(z)]`

## 3. Information Bottleneck (IB) Framework
- **Objective**: 
  `min[ I(X;T) - β·I(T;Y) ]`  
  (Compress input `X` while preserving information about `Y`)
- **Key Insights**:
  - Layers converge to critical points on IB bound
  - Training stages visualized in *Information Plane*:
    ```mermaid
    graph LR
    A[Undertrained] --> B[Compression] --> C[Generalization]
    ```
  - Hidden layers exponentially accelerate convergence by:
    - Breaking entropy gaps `ΔSₖ = I(X;Tₖ) - I(X;Tₖ₋₁)`
    - Avoiding critical slowing at phase transitions

## 4. Wavelets & Invariant Representations
- **Scattering Transform**:
  - Multilayer wavelet modulus averaging:
    ```
    Sx = [ |x∗ψλ₁|∗ϕ, ||x∗ψλ₁|∗ψλ₂|∗ϕ, ... ]
    ```
  - **Properties**:
    - Translation/rotation/scaling invariant
    - Deformation-stable: `‖Sx - Sxₜ‖ ≤ C·sup|∇τ|`
    - Preserves norms: `‖Sx‖ = ‖x‖`
- **Applications**:
  | Task          | Performance       |
  |---------------|-------------------|
  | MNIST         | 0.4% error       |
  | Texture (CUREt)| 0.2% error      |

## 5. Renormalization Group (RG) Connections
- **Tensor Networks**:
  - Efficient representation of high-dim states (e.g., MERA)
  - Implement RG flow: `|Ψ⟩ → |Ψ'⟩ → |Ψ''⟩`
- **Key Principles**:
  - Remove short-range entanglement at each scale
  - Preserve locality of operators
  - Explain power-law correlations `C(L) ∼ L⁻²Δ`
- **Ising Model**: MERA achieves 0.003% error in scaling dimensions

## 6. Biological & Financial Applications
### Neuroscience
- Hippocampal neurons exhibit:
  - Scale-invariant activity distributions
  - Correlation spectra: `λ ∼ (size/rank)⁰·⁶⁴`
  - Slower dynamics in larger clusters

### Finance (Fast Exotic Pricing)
- **Approach**:
  ```mermaid
    graph LR
    A[High-dim pricing] --> B[DNN compression] --> C[23-param manifold]
    ```
- **Results**:
  - 100 prices + sensitivities in 10 mins
  - Leverages *Manifold Tangent Hypothesis*

## 7. Design Principles
- **Architecture**:
  - Layer sizes match information compression needs
  - Noise variance `β⁻¹ ∝ 1/layer_size`
- **Future Directions**:
  - Solvable DNN models via symmetry/group theory
  - Quantum circuit analogs for RG
  - Optimal algorithms beyond SGD

> "Failure is an option here. If things are not failing, you are not innovating."  
> – Guiding principle for theoretical exploration
