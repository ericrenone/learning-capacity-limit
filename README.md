# Capacity Dynamics in Neural Optimization

## Core

This project implements an information-geometric perspective on learning. It compares two distinct optimization regimes navigating a non-convex loss surface toward an irrational target ($2^{1/3}$):

1.  **High-Capacity (Full Precision):** Uses Nesterov-style momentum and floating-point gradients to traverse the smooth geodesics of the manifold.
2.  **Restricted (Quantized):** Simulates Edge-AI and low-rank constraints using a digital bottleneck ($n$-bit quantization) and stochastic gradient noise.

