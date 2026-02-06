# Capacity Dynamics in Neural Optimization

## Core

This project implements an information-geometric perspective on learning. It compares two distinct optimization regimes navigating a non-convex loss surface toward an irrational target:

1.  **High-Capacity (Full Precision):** Uses Nesterov-style momentum and floating-point gradients to traverse the smooth geodesics of the manifold.
2.  **Restricted (Quantized):** Simulates Edge-AI and low-rank constraints using a digital bottleneck ($n$-bit quantization) and stochastic gradient noise.

## Conclusion

Optimization isn’t just about the destination; it’s about the quality of the tools you use to get there. The precision of your data acts like a lens that defines the "map" your AI sees—if the resolution is low, a smooth path turns into a jagged, pixelated landscape. Because Edge-AI hardware has limited power, the optimizer acts like a hiker with blurry goggles, unable to see or reach a microscopic "perfect" point. Ultimately, you cannot separate the goal from the hardware’s ability to see the path, meaning success on small devices is about finding a stable "good enough neighborhood" rather than a single, perfect bullseye.
