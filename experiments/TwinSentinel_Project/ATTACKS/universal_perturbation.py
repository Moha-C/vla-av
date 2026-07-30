"""
UniversalPerturbation Attack (Moosavi-Dezfooli et al., CVPR 2017)

Attack Model:
    1. Find a single universal perturbation δ_u that fools the model for MOST inputs
    2. Broadcast this perturbation to ALL vehicles in simulation
    3. Observe degraded vehicle trajectories / safety decisions
    
Threat Model (VANET context):
    - Scenario A: Man-in-the-middle on V2X broadcast channel
    - Scenario B: Compromised infrastructure broadcasting malicious position updates
    - Result: All vehicles receive slightly perturbed sensor readings
    
Detection Difficulty:
    - Medium: Constant bias across all vehicles is detectable via statistical analysis
    - Signature: Coordinated fleet-wide degradation (vs random sensor noise)
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime
from pathlib import Path
import logging

from .threat_models import VehicleThreatModel, VehicleTrajectoryModel, get_default_threat_model

# Setup logging
logger = logging.getLogger(__name__)


class UniversalPerturbationAttack:
    """
    UniversalPerturbation attack for VANET simulation.
    
    Computes a single perturbation δ_u that is applied to ALL vehicles.
    Uses iterative gradient descent on a threat model to find δ_u.
    """
    
    def __init__(
        self,
        threat_model: Optional[VehicleThreatModel] = None,
        epsilon: float = 0.3,
        learning_rate: float = 0.01,
        num_iterations: int = 100,
        num_samples_per_iter: int = 10,
        name: str = "universal_perturbation"
    ):
        """
        Args:
            threat_model: Model to compute gradients from (if None, uses default)
            epsilon: Max perturbation magnitude (L∞ norm)
            learning_rate: Step size for gradient descent
            num_iterations: Number of gradient descent iterations
            num_samples_per_iter: Training samples used per iteration
            name: Attack name for logging
        """
        self.threat_model = threat_model or get_default_threat_model()
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.num_samples_per_iter = num_samples_per_iter
        self.name = name
        
        # Attack state
        self.perturbation = None
        self.perturbation_history = []
        self.loss_history = []
        self.query_count = 0
        self.start_time = None
        self.end_time = None
        
        logger.info(f"UniversalPerturbation initialized:")
        logger.info(f"  Epsilon: {self.epsilon}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Iterations: {self.num_iterations}")
        logger.info(f"  Threat model: {self.threat_model.config.name}")
    
    def compute_perturbation(
        self,
        sample_generator: Callable[[], np.ndarray],
        loss_func: Callable[[np.ndarray, np.ndarray], float],
        input_shape: Tuple[int, ...] = (5,)
    ) -> np.ndarray:
        """
        Compute universal perturbation using iterative gradient descent.
        
        Algorithm (Moosavi-Dezfooli et al.):
            1. Initialize δ_u = 0
            2. For each iteration:
               a. Sample random inputs X
               b. Compute gradient of loss(X + δ_u) w.r.t. δ_u
               c. δ_u -= lr * ∇ δ_u J(X + δ_u)
               d. Clip δ_u to [-ε, ε]
        
        Args:
            sample_generator: Function that generates random training samples
            loss_func: Function that computes loss: loss = loss_func(x, perturbation)
            input_shape: Shape of input vectors (e.g., (5,) for vehicle state)
            
        Returns:
            Computed universal perturbation δ_u
        """
        self.start_time = datetime.now()
        logger.info(f"Starting perturbation computation...")
        
        # Initialize perturbation
        self.perturbation = np.zeros(input_shape, dtype=np.float32)
        
        # Iterative gradient descent
        for iteration in range(self.num_iterations):
            # Sample training batch
            samples = [sample_generator() for _ in range(self.num_samples_per_iter)]
            samples = np.array(samples)  # Shape: (num_samples, *input_shape)
            
            # Compute loss for current perturbation
            losses = np.array([
                loss_func(sample, self.perturbation) 
                for sample in samples
            ])
            avg_loss = np.mean(losses)
            self.loss_history.append(float(avg_loss))
            
            # Compute gradient via finite differences
            grad = np.zeros_like(self.perturbation)
            delta = 0.001
            
            for i in range(len(self.perturbation)):
                perturbation_plus = self.perturbation.copy()
                perturbation_minus = self.perturbation.copy()
                
                perturbation_plus[i] += delta
                perturbation_minus[i] -= delta
                
                loss_plus = np.mean([
                    loss_func(sample, perturbation_plus)
                    for sample in samples
                ])
                loss_minus = np.mean([
                    loss_func(sample, perturbation_minus)
                    for sample in samples
                ])
                
                grad[i] = (loss_plus - loss_minus) / (2 * delta)
                self.query_count += 2 * len(samples)
            
            # Gradient descent step
            self.perturbation -= self.learning_rate * grad
            
            # Clip to epsilon ball
            self.perturbation = np.clip(self.perturbation, -self.epsilon, self.epsilon)
            
            # Log progress
            grad_norm = np.linalg.norm(grad)
            if (iteration + 1) % max(1, self.num_iterations // 10) == 0:
                logger.info(
                    f"Iteration {iteration + 1}/{self.num_iterations} | "
                    f"Loss: {avg_loss:.4f} | Grad norm: {grad_norm:.4f} | "
                    f"Pert norm: {np.linalg.norm(self.perturbation):.4f}"
                )
            
            self.perturbation_history.append(self.perturbation.copy())
        
        self.end_time = datetime.now()
        elapsed = (self.end_time - self.start_time).total_seconds()
        logger.info(
            f"Perturbation computed! "
            f"Final loss: {self.loss_history[-1]:.4f} | "
            f"Total queries: {self.query_count} | "
            f"Time: {elapsed:.2f}s"
        )
        
        return self.perturbation
    
    def apply_to_vehicle_state(self, vehicle_state: np.ndarray) -> np.ndarray:
        """
        Apply universal perturbation to vehicle state.
        
        Args:
            vehicle_state: Current vehicle state (e.g., [x, y, vx, vy, heading])
            
        Returns:
            Perturbed vehicle state
        """
        if self.perturbation is None:
            raise RuntimeError("Perturbation not computed yet. Call compute_perturbation() first.")
        
        # Add perturbation
        perturbed_state = vehicle_state + self.perturbation
        
        return perturbed_state
    
    def get_statistics(self) -> Dict:
        """Get attack statistics and metadata."""
        elapsed = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        
        return {
            "attack_name": self.name,
            "threat_model": self.threat_model.config.name,
            "perturbation_shape": tuple(self.perturbation.shape) if self.perturbation is not None else None,
            "perturbation_norm": float(np.linalg.norm(self.perturbation)) if self.perturbation is not None else None,
            "perturbation_magnitude": float(np.max(np.abs(self.perturbation))) if self.perturbation is not None else None,
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
            "num_iterations": self.num_iterations,
            "num_samples_per_iter": self.num_samples_per_iter,
            "total_queries": self.query_count,
            "final_loss": float(self.loss_history[-1]) if self.loss_history else None,
            "initial_loss": float(self.loss_history[0]) if self.loss_history else None,
            "loss_improvement": (
                float(self.loss_history[0] - self.loss_history[-1])
                if self.loss_history else None
            ),
            "elapsed_seconds": elapsed,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "loss_history": [float(x) for x in self.loss_history],
        }
    
    def to_dict(self) -> Dict:
        """Serialize attack to dictionary."""
        return {
            **self.get_statistics(),
            "perturbation": self.perturbation.tolist() if self.perturbation is not None else None,
            "threat_model_config": {
                "name": self.threat_model.config.name,
                "model_type": self.threat_model.config.model_type,
                "input_dim": self.threat_model.config.input_dim,
                "output_dim": self.threat_model.config.output_dim,
                "description": self.threat_model.config.description,
            }
        }
    
    def save_to_file(self, filepath: Path) -> None:
        """Save attack metadata and perturbation to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Attack saved to {filepath}")
    
    @staticmethod
    def simple_loss_function(x: np.ndarray, perturbation: np.ndarray) -> float:
        """
        Simple loss function for demonstration.
        
        Loss = sum((x + perturbation)^2) - attempts to maximize coordinates
        """
        perturbed = x + perturbation
        loss = -np.sum(perturbed ** 2)  # Negative to maximize
        return loss


# Convenience functions for SUMO integration

def create_universal_perturbation_attack(
    epsilon: float = 0.3,
    iterations: int = 100
) -> UniversalPerturbationAttack:
    """Factory function to create attack with common parameters."""
    return UniversalPerturbationAttack(
        epsilon=epsilon,
        num_iterations=iterations,
        threat_model=VehicleTrajectoryModel()
    )


if __name__ == "__main__":
    # Demo: Simple standalone test
    logging.basicConfig(level=logging.INFO)
    
    attack = UniversalPerturbationAttack(epsilon=0.2, num_iterations=50)
    
    # Sample generator
    def sample_gen():
        return np.random.randn(5)  # Random vehicle state
    
    # Loss function
    def loss_fn(x, pert):
        return UniversalPerturbationAttack.simple_loss_function(x, pert)
    
    # Compute perturbation
    print("Computing perturbation...")
    delta_u = attack.compute_perturbation(sample_gen, loss_fn, input_shape=(5,))
    
    print(f"\nComputed perturbation: {delta_u}")
    print(f"Perturbation norm: {np.linalg.norm(delta_u):.4f}")
    print(f"\nStatistics: {json.dumps(attack.get_statistics(), indent=2)}")
