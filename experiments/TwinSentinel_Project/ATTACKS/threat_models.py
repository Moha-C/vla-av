"""
Threat Models for VANET ML Attacks

Defines models used to compute perturbations/gradients.
Each threat model represents an assumption about the vehicle's ML system.
"""

import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass
import json
from datetime import datetime


@dataclass
class ThreatModelConfig:
    """Configuration for a threat model"""
    name: str
    model_type: str  # "classification", "regression"
    input_dim: int
    output_dim: int
    description: str


class VehicleThreatModel:
    """
    Abstract threat model for vehicle ML systems.
    
    In VANET context, the "model" is a black box that takes vehicle state
    (position, velocity, sensor data) and outputs decisions (route, speed, alerts).
    
    This class approximates gradients without access to internal model.
    """
    
    def __init__(self, config: ThreatModelConfig):
        self.config = config
        self.gradient_queries = 0
        self.query_log = []
        
    def compute_gradient_blackbox(
        self, 
        query_func,
        x: np.ndarray, 
        delta: float = 0.001,
        batch_size: int = 5
    ) -> np.ndarray:
        """
        Finite differences gradient approximation (black-box).
        
        Args:
            query_func: Function that queries the model: loss = query_func(x)
            x: Input vector (vehicle state)
            delta: Perturbation magnitude for finite differences
            batch_size: Number of random directions to test
            
        Returns:
            Estimated gradient ∇_x J(x)
        """
        grad = np.zeros_like(x)
        
        for _ in range(batch_size):
            # Random direction
            u = np.random.randn(*x.shape)
            u = u / np.linalg.norm(u)
            
            # Finite difference
            loss_pos = query_func(x + delta * u)
            loss_neg = query_func(x - delta * u)
            
            grad += (loss_pos - loss_neg) * u / (2 * delta)
            self.gradient_queries += 2
        
        return grad / batch_size
    
    def compute_gradient_whitebox(
        self,
        loss_func,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Analytical gradient computation (white-box).
        
        Args:
            loss_func: Function that returns (loss, gradients)
            x: Input vector
            
        Returns:
            Exact gradient ∇_x J(x)
        """
        loss, grads = loss_func(x)
        self.gradient_queries += 1
        return grads
    
    def log_gradient_query(self, x_shape: Tuple, gradient_norm: float, loss: float):
        """Log gradient queries for detection analysis."""
        self.query_log.append({
            "timestamp": datetime.now().isoformat(),
            "x_shape": x_shape,
            "gradient_norm": float(gradient_norm),
            "loss": float(loss),
            "total_queries": self.gradient_queries
        })


class VehicleTrajectoryModel(VehicleThreatModel):
    """
    Trajectory prediction model for vehicles.
    
    Input: (position_x, position_y, velocity_x, velocity_y, heading)
    Output: (predicted_x, predicted_y) at next timestep
    """
    
    def __init__(self):
        config = ThreatModelConfig(
            name="vehicle_trajectory",
            model_type="regression",
            input_dim=5,  # x, y, vx, vy, heading
            output_dim=2,  # predicted_x, predicted_y
            description="Trajectory prediction model for vehicle path planning"
        )
        super().__init__(config)
        
    def simple_linear_model(self, state: np.ndarray) -> np.ndarray:
        """
        Simple linear trajectory model: next_pos = current_pos + velocity
        
        This is a placeholder threat model. Real VANET models are more complex.
        
        Args:
            state: [x, y, vx, vy, heading]
            
        Returns:
            [predicted_x, predicted_y]
        """
        x, y, vx, vy, heading = state
        return np.array([x + vx * 0.1, y + vy * 0.1])


class VehicleSafetyModel(VehicleThreatModel):
    """
    Safety classification model for vehicles.
    
    Input: (speed, acceleration, time_to_collision, road_type, weather)
    Output: Binary decision [safe=1, unsafe=0]
    """
    
    def __init__(self):
        config = ThreatModelConfig(
            name="vehicle_safety",
            model_type="classification",
            input_dim=5,  # speed, accel, ttc, road_type, weather
            output_dim=1,  # binary safe/unsafe
            description="Safety classification model for autonomous vehicle"
        )
        super().__init__(config)
    
    def simple_linear_classifier(self, state: np.ndarray) -> float:
        """
        Simple linear safety classifier: confidence = w^T * state
        
        Args:
            state: [speed, acceleration, ttc, road_type, weather]
            
        Returns:
            Confidence score [0, 1]
        """
        # Learned weights (arbitrary for this model)
        weights = np.array([-0.5, 0.2, 1.0, -0.1, 0.3])
        logits = np.dot(weights, state)
        
        # Sigmoid to get probability
        confidence = 1.0 / (1.0 + np.exp(-logits))
        return confidence


class VehicleDecisionModel(VehicleThreatModel):
    """
    Decision model for vehicle action selection.
    
    Input: (current_state, nearby_vehicles, traffic_light_state, etc.)
    Output: Multi-class decision (ACCELERATE, BRAKE, TURN_LEFT, TURN_RIGHT, etc.)
    """
    
    def __init__(self):
        config = ThreatModelConfig(
            name="vehicle_decision",
            model_type="classification",
            input_dim=10,  # aggregated state
            output_dim=5,  # action classes
            description="Decision model for vehicle action selection"
        )
        super().__init__(config)


# Utility functions for threat model analysis

def get_default_threat_model() -> VehicleThreatModel:
    """Returns the default threat model for UniversalPerturbation."""
    return VehicleTrajectoryModel()


def threat_model_to_dict(model: VehicleThreatModel) -> Dict:
    """Serialize threat model to dictionary."""
    return {
        "name": model.config.name,
        "model_type": model.config.model_type,
        "input_dim": model.config.input_dim,
        "output_dim": model.config.output_dim,
        "description": model.config.description,
        "gradient_queries": model.gradient_queries,
        "query_log": model.query_log
    }
