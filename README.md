# Capacity Dynamics in Neural Optimization

## Core

This project implements an information-geometric perspective on learning. It compares two distinct optimization regimes navigating a non-convex loss surface toward an irrational target:

1.  **High-Capacity (Full Precision):** Uses Nesterov-style momentum and floating-point gradients to traverse the smooth geodesics of the manifold.
2.  **Restricted (Quantized):** Simulates Edge-AI and low-rank constraints using a digital bottleneck ($n$-bit quantization) and stochastic gradient noise.

## Conclusion

- The Map Changes: The precision of your numbers acts like a lens; if the lens is blurry, a smooth mountain path turns into a jagged flight of stairs.

- The Traveler is Limited: An optimizer can only navigate the shapes it is smart enough to "see" and flexible enough to move through.

- The Goal and the View are One: You cannot reach a perfect destination if your tools aren't sharp enough to distinguish it from the surrounding ground.

- Hardware Sets the Rules: In Edge-AI, small chips force you to use "low-detail" goggles, making the world look pixelated and simplified.

- Aim for the Zone, Not the Point: Success isn't about hitting a microscopic bullseye, but finding a "good enough" neighborhood that your hardware can actually recognize.
