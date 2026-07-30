"""
Targeted Adversarial Sensor Spoofing Attack (Hirano & Takemoto, 2019)

Attack Model:
    1. Compute a targeted adversarial perturbation δ that forces vehicles to perceive
       a fake obstacle at a specific location
    2. The perturbation is calculated using iterative gradient descent (FGSM-style)
    3. Applied to spoofed sensor readings (LIDAR/Radar) or V2X messages
    
Threat Model (VANET context):
    - Scenario: Man-in-the-middle on V2X broadcast OR compromised sensor/ECU
    - Target: Vehicle obstacle detection/avoidance logic
    - Result: Coordinated emergency braking → cascade of traffic disruption
    
Realism:
    - High: Sensor spoofing is a known vulnerability in autonomous systems
    - Signature: Localized cascade of emergency braking events at specific position
    - Detection: Cluster of false positives from different vehicles at same location

Reference:
    Hirano, H., & Takemoto, K. (2019). Simple iterative method for generating 
    targeted universal adversarial perturbations. arXiv:1911.06502
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime
from pathlib import Path
import logging
import math

from .threat_models import VehicleThreatModel, get_default_threat_model

logger = logging.getLogger(__name__)


class TargetedAdversarialSpoofingAttack:
    """
    Targeted Adversarial Sensor Spoofing Attack for VANET.
    
    Computes a 2D spatial perturbation δ = [δ_x, δ_y] that is applied to
    inject a fake obstacle. The perturbation is calculated to maximize the
    detection probability across the fleet (targeted adversarial).
    """
    
    def __init__(
        self,
        threat_model: Optional[VehicleThreatModel] = None,
        epsilon: float = 50.0,  # Max spatial perturbation (meters)
        learning_rate: float = 5.0,  # Gradient step size (meters)
        num_iterations: int = 50,  # Optimization iterations
        num_samples_per_iter: int = 20,  # Training samples per iteration
        detection_radius: float = 100.0,  # How far vehicles can detect the obstacle
        name: str = "adversarial_sensor_spoofing"
    ):
        """
        Args:
            threat_model: Model to compute gradients from
            epsilon: Max perturbation magnitude (L∞ norm, in meters)
            learning_rate: Step size for gradient descent
            num_iterations: Number of optimization iterations
            num_samples_per_iter: Training samples per iteration
            detection_radius: Maximum distance for obstacle detection
            name: Attack name for logging
        """
        self.threat_model = threat_model or get_default_threat_model()
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.num_samples_per_iter = num_samples_per_iter
        self.detection_radius = detection_radius
        self.name = name
        
        # Attack state
        self.perturbation = None  # Shape: (2,) for [δ_x, δ_y]
        self.perturbation_history = []
        self.loss_history = []
        self.gradient_norms = []
        self.query_count = 0
        self.start_time = None
        self.end_time = None
        
        logger.info(f"TargetedAdversarialSpoofing initialized:")
        logger.info(f"  Epsilon: {self.epsilon} meters")
        logger.info(f"  Learning rate: {self.learning_rate} m/iter")
        logger.info(f"  Iterations: {self.num_iterations}")
        logger.info(f"  Detection radius: {self.detection_radius} meters")
        logger.info(f"  Threat model: {self.threat_model.config.name}")
    
    def compute_perturbation(
        self,
        target_position: Tuple[float, float],
        sample_generator: Callable[[], Tuple[float, float]],
        detection_model: Callable[[np.ndarray, np.ndarray], float],
        verbose: bool = True
    ) -> np.ndarray:
        """
        Compute targeted adversarial perturbation using iterative gradient descent.
        
        Algorithm (Hirano & Takemoto, 2019):
            1. Initialize δ = [0, 0]
            2. For each iteration:
               a. Sample random vehicle positions X
               b. For each position x ∈ X:
                  - Compute detection probability: p = detection_model(x, target + δ)
                  - Compute loss: L = -log(p) (maximize detection)
               c. Compute gradient: ∇_δ L via finite differences
               d. Update: δ -= lr * sign(∇_δ L)  [FGSM-style]
               e. Clip δ to [-ε, ε]
        
        Args:
            target_position: (x_t, y_t) where to inject the fake obstacle
            sample_generator: Function that returns random (x, y) vehicle positions
            detection_model: Function that returns detection prob: p = detection_model(vehicle_pos, obstacle_pos)
            verbose: Whether to log progress
            
        Returns:
            Computed perturbation δ = [δ_x, δ_y]
        """
        self.start_time = datetime.now()
        target_pos = np.array(target_position, dtype=np.float32)
        
        if verbose:
            logger.info(f"Computing targeted adversarial perturbation for target {target_pos}...")
        
        # Initialize perturbation
        self.perturbation = np.zeros((2,), dtype=np.float32)
        
        # Iterative optimization
        for iteration in range(self.num_iterations):
            # Sample training batch: random vehicle positions
            samples = np.array([
                sample_generator() for _ in range(self.num_samples_per_iter)
            ], dtype=np.float32)  # Shape: (num_samples, 2)
            
            # Compute loss for current perturbation
            current_obstacle_pos = target_pos + self.perturbation
            losses = []
            
            for vehicle_pos in samples:
                # Detection probability (higher = more likely to detect the fake obstacle)
                detection_prob = detection_model(vehicle_pos, current_obstacle_pos)
                # Loss = -log(detection_prob) to maximize detection across vehicles
                loss = -np.log(np.clip(detection_prob, 1e-6, 1.0))
                losses.append(loss)
            
            avg_loss = np.mean(losses)
            self.loss_history.append(float(avg_loss))
            
            # Compute gradient via finite differences (for 2D perturbation)
            grad = np.zeros_like(self.perturbation)
            delta = 1.0  # Small spatial perturbation for finite difference
            
            for dim in range(2):
                # Perturbation + delta in dimension dim
                pert_plus = self.perturbation.copy()
                pert_plus[dim] += delta
                obstacle_pos_plus = target_pos + pert_plus
                
                # Perturbation - delta in dimension dim
                pert_minus = self.perturbation.copy()
                pert_minus[dim] -= delta
                obstacle_pos_minus = target_pos + pert_minus
                
                # Compute losses
                loss_plus = np.mean([
                    -np.log(np.clip(detection_model(v_pos, obstacle_pos_plus), 1e-6, 1.0))
                    for v_pos in samples
                ])
                loss_minus = np.mean([
                    -np.log(np.clip(detection_model(v_pos, obstacle_pos_minus), 1e-6, 1.0))
                    for v_pos in samples
                ])
                
                # Gradient approximation
                grad[dim] = (loss_plus - loss_minus) / (2.0 * delta)
                self.query_count += 2 * len(samples)
            
            # Gradient descent step (with FGSM-style sign thresholding for efficiency)
            grad_sign = np.sign(grad)
            self.perturbation -= self.learning_rate * grad_sign
            
            # Clip to epsilon ball (L∞ norm)
            self.perturbation = np.clip(self.perturbation, -self.epsilon, self.epsilon)
            
            # Track metrics
            grad_norm = np.linalg.norm(grad)
            self.gradient_norms.append(float(grad_norm))
            self.perturbation_history.append(self.perturbation.copy())
            
            # Log progress
            if verbose and (iteration + 1) % max(1, self.num_iterations // 5) == 0:
                logger.info(
                    f"  Iteration {iteration + 1}/{self.num_iterations} | "
                    f"Loss: {avg_loss:.4f} | Grad norm: {grad_norm:.4f} | "
                    f"Pert: δ=[{self.perturbation[0]:+.2f}, {self.perturbation[1]:+.2f}] "
                    f"| Effective pos: [{target_pos[0] + self.perturbation[0]:.1f}, "
                    f"{target_pos[1] + self.perturbation[1]:.1f}]"
                )
        
        self.end_time = datetime.now()
        elapsed = (self.end_time - self.start_time).total_seconds()
        
        if verbose:
            logger.info(
                f"✓ Perturbation computed! "
                f"Final loss: {self.loss_history[-1]:.4f} | "
                f"Final perturbation: δ=[{self.perturbation[0]:+.2f}, {self.perturbation[1]:+.2f}] | "
                f"Total queries: {self.query_count} | Time: {elapsed:.2f}s"
            )
        
        return self.perturbation
    
    def get_obstacle_position(self, target_position: Tuple[float, float]) -> Tuple[float, float]:
        """
        Get the actual obstacle position after applying perturbation.
        
        Args:
            target_position: Original target position
            
        Returns:
            Effective obstacle position = target + perturbation
        """
        if self.perturbation is None:
            raise RuntimeError("Perturbation not computed yet. Call compute_perturbation() first.")
        
        target = np.array(target_position)
        effective_pos = target + self.perturbation
        return tuple(effective_pos)
    
    def compute_detection_probability(
        self, 
        vehicle_pos: Tuple[float, float],
        obstacle_pos: Tuple[float, float]
    ) -> float:
        """
        Compute probability that a vehicle detects the fake obstacle.
        
        Uses a sigmoid model for realistic detection: high near obstacle,
        decays gradually with distance.
        p = 1 / (1 + exp(-k*(detection_radius - distance)))
        where k controls the steepness of the curve.
        
        Args:
            vehicle_pos: (x_v, y_v) vehicle position
            obstacle_pos: (x_o, y_o) obstacle position
            
        Returns:
            Detection probability [0, 1]
        """
        v_pos = np.array(vehicle_pos)
        o_pos = np.array(obstacle_pos)
        
        distance = np.linalg.norm(v_pos - o_pos)
        
        # Sigmoid detection model: steeper falloff for more realistic behavior
        # At detection_radius, probability is 50%
        # Within detection_radius/2, probability is >90%
        # Beyond 1.5*detection_radius, probability is <10%
        k = 0.04  # Steepness parameter (larger = sharper transition)
        prob = 1.0 / (1.0 + np.exp(-k * (self.detection_radius - distance)))
        
        return float(np.clip(prob, 0.0, 1.0))
    
    def compute_detection_impact(
        self,
        vehicle_pos: Tuple[float, float],
        obstacle_pos: Tuple[float, float]
    ) -> Dict[str, float]:
        """
        Compute the impact of detection on vehicle behavior.
        
        Returns metrics about how vehicle should react.
        """
        detection_prob = self.compute_detection_probability(vehicle_pos, obstacle_pos)
        
        v_pos = np.array(vehicle_pos)
        o_pos = np.array(obstacle_pos)
        distance = np.linalg.norm(v_pos - o_pos)
        
        # Direction from vehicle to obstacle
        direction = (o_pos - v_pos) / (distance + 1e-6)
        
        return {
            "detection_probability": float(detection_prob),
            "distance_to_obstacle": float(distance),
            "direction_x": float(direction[0]),
            "direction_y": float(direction[1]),
            # Braking intensity based on distance (emergency if very close)
            "braking_intensity": float(np.clip(1.0 - (distance / self.detection_radius), 0.0, 1.0)),
        }
    
    def get_statistics(self) -> Dict:
        """Get attack statistics and metadata."""
        elapsed = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        
        return {
            "attack_name": self.name,
            "attack_type": "targeted_adversarial_sensor_spoofing",
            "threat_model": self.threat_model.config.name,
            "perturbation_shape": tuple(self.perturbation.shape) if self.perturbation is not None else None,
            "perturbation": [float(p) for p in self.perturbation] if self.perturbation is not None else None,
            "perturbation_norm": float(np.linalg.norm(self.perturbation)) if self.perturbation is not None else None,
            "perturbation_magnitude": float(np.max(np.abs(self.perturbation))) if self.perturbation is not None else None,
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
            "num_iterations": self.num_iterations,
            "num_samples_per_iter": self.num_samples_per_iter,
            "detection_radius": self.detection_radius,
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
            "gradient_norms": [float(x) for x in self.gradient_norms],
        }
    
    def to_dict(self) -> Dict:
        """Serialize attack to dictionary."""
        return {
            **self.get_statistics(),
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


def create_adversarial_spoofing_attack(
    epsilon: float = 50.0,
    iterations: int = 50,
    detection_radius: float = 100.0
) -> TargetedAdversarialSpoofingAttack:
    """Factory function to create attack with common parameters."""
    return TargetedAdversarialSpoofingAttack(
        epsilon=epsilon,
        num_iterations=iterations,
        detection_radius=detection_radius,
        threat_model=get_default_threat_model()
    )


if __name__ == "__main__":
    # Demo: Simple standalone test
    logging.basicConfig(level=logging.INFO)
    
    attack = create_adversarial_spoofing_attack(epsilon=50.0, iterations=30)
    
    # Target position (center of a road segment)
    target_pos = (100.0, 50.0)
    
    # Sample generator: random vehicle positions
    def sample_gen():
        return (
            np.random.uniform(0, 200),  # x: 0-200m
            np.random.uniform(0, 100)   # y: 0-100m
        )
    
    # Detection model
    def detection_fn(v_pos, o_pos):
        return attack.compute_detection_probability(v_pos, o_pos)
    
    # Compute perturbation
    print("Computing targeted adversarial perturbation...")
    delta = attack.compute_perturbation(target_pos, sample_gen, detection_fn, verbose=True)
    
    print(f"\n✓ Attack computed successfully!")
    print(f"  Original target: {target_pos}")
    print(f"  Perturbation: δ = {delta}")
    print(f"  Effective obstacle position: {target_pos + delta}")
    print(f"\nFull statistics:")
    print(json.dumps(attack.get_statistics(), indent=2))
