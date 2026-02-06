# Capacity Dynamics in Neural Optimization

## Core

This project implements an information-geometric perspective on learning. It compares two distinct optimization regimes navigating a non-convex loss surface toward an irrational target:

1.  **High-Capacity (Full Precision):** Uses Nesterov-style momentum and floating-point gradients to traverse the smooth geodesics of the manifold.
2.  **Restricted (Quantized):** Simulates Edge-AI and low-rank constraints using a digital bottleneck ($n$-bit quantization) and stochastic gradient noise.

## Conclusion

Optimization is not just about the objective function; the resolution of the weights defines the very topology the optimizer "sees."
The optimizer’s capacity defines the geometry it inhabits. You cannot separate the objective (where you want to go) from the precision (how clearly you can see the path). For Edge-AI, this means the "best possible" solution is not a single point, but a region of irreducible uncertainty dictated by the hardware bottleneck.
