#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Capacity-Constrained Learning Dynamics
======================================

Demonstrates learning under restricted hypothesis updates versus
full gradient-based optimization.

Core result:
    Finite-loss-distance optima may be unreachable under
    limited representation transformations.

This models:
    - expressivity limits
    - information bottlenecks
    - quantization / low-rank update constraints

Standard library only.

Author: Eric Ren
License: MIT
"""

import math
import random
import tkinter as tk
import time

# ------------------ Problem Definition ------------------

TARGET = 2 ** (1/3)     # high-complexity optimum
STEPS = 180
LR = 0.10

WIDTH = 920
HEIGHT = 520
MARGIN = 70

# ------------------ Metrics ------------------

def loss(x):
    return (x - TARGET) ** 2

def fisher_distance(x):
    return abs(3 * math.log(x))

# ------------------ Update Classes ------------------

class LowCapacityUpdater:
    """
    Restricted algebraic updates mimicking limited representational power.
    """

    def step(self, x):
        a = random.uniform(-0.02, 0.02)
        b = random.uniform(0.7, 1.3)
        c = random.uniform(-0.02, 0.02)
        v = b * x + c
        if v <= 0:
            return x
        return abs(a + math.sqrt(v))


class HighCapacityUpdater:
    """
    Full gradient descent with no representational constraint.
    """

    def step(self, x):
        grad = 2 * (x - TARGET)
        return x - LR * grad


# ------------------ Visualization Engine ------------------

class LearningExperiment:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Capacity-Constrained Learning Dynamics")

        self.canvas = tk.Canvas(self.root, width=WIDTH, height=HEIGHT, bg="white")
        self.canvas.pack()

        self.low = 1.0
        self.high = 1.0

        self.low_updater = LowCapacityUpdater()
        self.high_updater = HighCapacityUpdater()

        self.step_count = 0

        self.draw_axes()
        self.root.after(40, self.iterate)

    def draw_axes(self):
        self.canvas.create_line(MARGIN, HEIGHT-MARGIN, WIDTH-MARGIN, HEIGHT-MARGIN)
        self.canvas.create_line(MARGIN, MARGIN, MARGIN, HEIGHT-MARGIN)

        self.canvas.create_text(
            WIDTH/2, 25,
            text="Learning under Representation Capacity Constraints",
            font=("Arial", 14, "bold")
        )

        self.canvas.create_text(WIDTH-120, HEIGHT-30, text="Training steps →")
        self.canvas.create_text(25, 90, text="Loss", angle=90)

    def sx(self, t):
        return MARGIN + t / STEPS * (WIDTH - 2 * MARGIN)

    def sy(self, v):
        v = min(v, 0.5)
        return HEIGHT - MARGIN - v * (HEIGHT - 2 * MARGIN) * 2

    def plot(self, t, val, color):
        x = self.sx(t)
        y = self.sy(val)
        self.canvas.create_oval(x-2, y-2, x+2, y+2, fill=color, outline="")

    def iterate(self):
        if self.step_count >= STEPS:
            self.show_summary()
            return

        self.low = self.low_updater.step(self.low)
        self.high = self.high_updater.step(self.high)

        low_l = loss(self.low)
        high_l = loss(self.high)

        self.plot(self.step_count, low_l, "red")
        self.plot(self.step_count, high_l, "blue")

        self.canvas.delete("info")

        info = (
            f"step: {self.step_count}\n"
            f"low x : {self.low:.6f}   loss: {low_l:.6f}\n"
            f"high x: {self.high:.6f}  loss: {high_l:.6f}\n"
            f"Fisher distance(high): {fisher_distance(self.high):.4f}"
        )

        self.canvas.create_text(
            WIDTH-300, 100,
            anchor="nw",
            text=info,
            tag="info",
            font=("Courier", 10)
        )

        self.step_count += 1
        self.root.after(40, self.iterate)

    def show_summary(self):
        self.canvas.create_rectangle(
            160, 150, WIDTH-160, HEIGHT-150,
            fill="white", outline="black"
        )

        summary = f"""
FINAL MEASUREMENTS

Target optimum: {TARGET:.8f}

Low-capacity hypothesis:
    x = {self.low:.8f}
    loss = {loss(self.low):.8f}

High-capacity hypothesis:
    x = {self.high:.8f}
    loss = {loss(self.high):.8f}

CONCLUSION:
The minimizer exists at finite loss distance.
However, restricted update algebra cannot reach it.
Learning failure arises from representation capacity,
not optimization instability.
"""

        self.canvas.create_text(
            WIDTH/2, HEIGHT/2,
            text=summary,
            justify="center",
            font=("Arial", 12)
        )


# ------------------ Execution ------------------

if __name__ == "__main__":
    LearningExperiment().root.mainloop()

