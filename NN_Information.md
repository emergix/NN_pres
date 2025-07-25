# Information Theory in Deep Learning: Summary

 presentation : [NN_Information.pdf](./pdf/NN_Information.pdf)

## Core Concepts
### Kullback-Leibler (KL) Divergence
- **Definition**:  
  \( D_{KL}[p \parallel q] = \int p(x) \log \frac{p(x)}{q(x)}  dx \)
- **Properties**:
  - Asymmetric (\( D_{KL}[p \parallel q] \neq D_{KL}[q \parallel p] \))
  - \( D_{KL}[p \parallel p] = 0 \)
- **Key Insight**:  
  Measures information gain when conditioning on subset \( F \):  
  \( D_{KL}[q \parallel q_F] = \log \frac{1}{\text{Prob}[F]} \)

### Mutual Information (MI)
- **Definition**:  
  \( I(X; Y) = D_{KL}[p(x,y) \parallel p(x)p(y)] \)
- **Properties**:
  - Data Processing Inequality: \( X \to Y \to Z \implies I(X;Y) \geq I(X;Z) \)
  - Invariant under invertible reparameterization

### Variational Inference
- **Evidence Lower Bound (ELBO)**:  
  \( \log p(x) \geq \mathbb{E}_q[\log p(x,z)] - \mathbb{E}_q[\log q(z)] \)
- **Information Decomposition**:  
  \( \log p(x) = \text{ELBO} + D_{KL}[q(z) \parallel p(z|x)] \)

---

## Deep Neural Networks & Information Theory
### Information Bottleneck (IB) Framework
- **Objective**:  
  \( \min_{p(\hat{x}|x)} \left\{ I(\hat{X}; X) - \beta I(\hat{X}; Y) \right\} \)
- **Optimal Encoder**:  
  \( p(x|\hat{x}) \propto p(x) \exp\left(-\beta D_{KL}[p(y|x) \parallel p(y|\hat{x})]\right) \)

### Layer-Wise Information Dynamics
| Concept | Description |
|---------|-------------|
| **Information Plane** | Encoder MI (\( I(X;T) \)) vs Decoder MI (\( I(T;Y) \)) |
| **DPI in DNNs** | \( I(X;h_i) \geq I(X;h_{i+1}) \) and \( I(h_i;Y) \geq I(h_{i+1};Y) \) |
| **Phase Transitions** | Critical points where layer representations bifurcate (5% undertrained → 80% well-trained) |

### Training Dynamics
```mermaid
graph LR
A[SGD Noise] --> B[Gibbs Distribution]
B --> C[IB-Optimal Representation]
C --> D[Maximal Compression]
