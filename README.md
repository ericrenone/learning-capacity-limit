# Capacity Dynamics in Neural Optimization

## Core

This project implements an information-geometric perspective on learning. It compares two distinct optimization regimes navigating a non-convex loss surface toward an irrational target:

1.  **High-Capacity (Full Precision):** Uses Nesterov-style momentum and floating-point gradients to traverse the smooth geodesics of the manifold.
2.  **Restricted (Quantized):** Simulates Edge-AI and low-rank constraints using a digital bottleneck ($n$-bit quantization) and stochastic gradient noise.

## Conclusion

In a controlled convex stochastic regression setting, increased parameter precision and optimizer memory accelerate convergence and reduce variance, while quantization and noise impose fundamental limits on attainable estimator certainty, producing qualitatively distinct optimization geometries despite identical objectives.
