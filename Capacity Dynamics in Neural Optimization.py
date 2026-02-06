

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import Tuple
import warnings

# ===================== Production Setup =====================
warnings.filterwarnings('ignore')
plt.style.use('dark_background')

@dataclass
class MLFlowConfig:
    """Centralized Configuration with Master Seed."""
    target: float = 2 ** (1/3)
    n_steps: int = 500
    lr: float = 0.08
    momentum: float = 0.92
    random_seed: int = 99       # The Root of Reproducibility
    noise_scale: float = 0.012   # Magnitude of stochasticity
    quantization_bits: int = 2
    batch_size: int = 32

# ===================== AI/ML Dynamics Engine =====================
class AISimulator:
    def __init__(self, config: MLFlowConfig):
        self.config = config
        # All randomness is piped through this single seeded generator
        self.rng = np.random.default_rng(config.random_seed)
        
        self.x_low = 1.0   
        self.x_high = 1.0  
        self.v_high = 0.0  
        
        self._generate_seeded_data()
        
    def _generate_seeded_data(self):
        """Generates a deterministic data pool based on the master seed."""
        self.data_pool = self.config.target + self.rng.normal(
            0, self.config.noise_scale, size=5000
        )
        self.data_idx = 0
    
    def _sample_batch(self) -> np.ndarray:
        """Deterministic sampling sequence."""
        batch = self.data_pool[self.data_idx : self.data_idx + self.config.batch_size]
        self.data_idx = (self.data_idx + self.config.batch_size) % len(self.data_pool)
        return batch
    
    def get_metrics(self, x: float) -> Tuple[float, float]:
        batch = self._sample_batch()
        mse = 0.5 * np.mean((x - batch) ** 2)
        # Empirical Fisher Information: (Grad_L)^2 / Sigma^2
        fisher = np.mean(((x - batch) / (self.config.noise_scale ** 2)) ** 2)
        return mse, fisher / (np.abs(x) ** 2 + 1e-6)

    def step(self):
        batch = self._sample_batch()
        batch_mean = np.mean(batch)

        # 1. Restricted (Quantized) Update with seeded noise injection
        intrinsic_noise = self.rng.normal(0, self.config.noise_scale)
        self.x_low = self.x_low - 0.08 * (self.x_low - batch_mean) + intrinsic_noise
        self.x_low = np.round(self.x_low, self.config.quantization_bits)

        # 2. High-Capacity (Momentum SGD) Update
        grad_high = (self.x_high - batch_mean)
        self.v_high = self.config.momentum * self.v_high - self.config.lr * grad_high
        self.x_high += self.v_high
        
        l_low, f_low = self.get_metrics(self.x_low)
        l_high, f_high = self.get_metrics(self.x_high)
        
        return (self.x_low, l_low, f_low), (self.x_high, l_high, f_high)

    def animate(self):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7))
        c_low, c_high, c_target = '#ff0055', '#00d4ff', '#00ff88'

        ax1.set_title(r"I. Parameter Manifold ($\Theta$)", fontsize=13, pad=15)
        ax1.axvline(self.config.target, color=c_target, ls='--', alpha=0.5, label=r'Target $\theta^*$')
        dot_low, = ax1.plot([], [], 'o', color=c_low, ms=12, label='Restricted (Red)')
        dot_high, = ax1.plot([], [], 'o', color=c_high, ms=12, label='High-Cap (Blue)')
        ax1.set_xlim(0.9, 1.4); ax1.set_ylim(-0.1, 0.1)
        ax1.get_yaxis().set_visible(False)
        ax1.legend(frameon=False)

        ax2.set_title("II. Information Topology", fontsize=13, pad=15)
        ax2.set_xlabel("MSE Loss", color='white')
        ax2.set_ylabel("Fisher Info (Density)", color='white')
        path_low, = ax2.plot([], [], color=c_low, lw=1.5, alpha=0.4)
        path_high, = ax2.plot([], [], color=c_high, lw=1.5, alpha=0.4)
        head_low = ax2.scatter([], [], color='white', edgecolors=c_low, s=60, zorder=5)
        head_high = ax2.scatter([], [], color='white', edgecolors=c_high, s=60, zorder=5)

        ax3.set_title("III. Convergence Timeline", fontsize=13, pad=15)
        ax3.set_xlabel("Steps")
        ax3.set_ylabel("Log-Scale Loss")
        line_low, = ax3.semilogy([], [], color=c_low, lw=2, label='Red: Restricted')
        line_high, = ax3.semilogy([], [], color=c_high, lw=2, label='Blue: High-Cap')
        ax3.set_xlim(0, self.config.n_steps); ax3.set_ylim(1e-7, 1.0)
        ax3.legend(frameon=False)

        h_l_low, h_f_low, h_l_high, h_f_high, steps = [], [], [], [], []

        def update(frame):
            (x_l, l_l, f_l), (x_h, l_h, f_h) = self.step()
            dot_low.set_data([x_l], [0]); dot_high.set_data([x_h], [0])
            
            h_l_low.append(l_l); h_f_low.append(f_l)
            h_l_high.append(l_h); h_f_high.append(f_h)
            path_low.set_data(h_l_low, h_f_low); path_high.set_data(h_l_high, h_f_high)
            head_low.set_offsets([[l_l, f_l]]); head_high.set_offsets([[l_h, f_h]])
            
            # Perfect Scaling
            all_l = h_l_high + h_l_low
            ax2.set_xlim(max(min(all_l)*0.9, 1e-8), max(all_l)*1.1)
            ax2.set_ylim(min(h_f_high + h_f_low)*0.98, max(h_f_high + h_f_low)*1.02)

            steps.append(frame)
            line_low.set_data(steps, h_l_low); line_high.set_data(steps, h_l_high)
            
            fig.suptitle(fr"Capacity Dynamics | Seed: {self.config.random_seed} | Step {frame}", 
                         color='white', fontsize=15, y=0.98)
            return dot_low, dot_high, path_low, path_high, line_low, line_high

        ani = FuncAnimation(fig, update, frames=self.config.n_steps, interval=15, blit=False)
        plt.tight_layout(rect=[0, 0.05, 1, 0.92])
        plt.show()

if __name__ == "__main__":
    AISimulator(MLFlowConfig()).animate()
