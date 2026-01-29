"""
Kalman Filter for Motion Prediction
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class KalmanFilter:
    """Kalman Filter for tracking object motion"""
    
    def __init__(self, ndim: int = 4):
        """
        Initialize Kalman filter
        
        Args:
            ndim: Dimensionality of state (4 for 2D bbox)
        """
        self.ndim = ndim
        
        # State transition matrix
        self.F = np.eye(ndim, ndim)
        self.F[:ndim//2, ndim//2:] = np.eye(ndim//2, ndim//2)
        
        # Measurement matrix
        self.H = np.eye(ndim//2, ndim)
        
        # Process noise covariance
        self.Q = np.eye(ndim, ndim) * 0.01
        
        # Measurement noise covariance
        self.R = np.eye(ndim//2, ndim//2) * 0.1
        
        # State estimate covariance
        self.P = np.eye(ndim, ndim)
        
        # State
        self.x = None
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict next state
        
        Args:
            x: Current state
            
        Returns:
            Predicted state
        """
        self.x = x
        
        # Predict state
        x_pred = self.F @ self.x
        
        # Update covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return x_pred
    
    def update(self, z: np.ndarray) -> np.ndarray:
        """
        Update state with measurement
        
        Args:
            z: Measurement
            
        Returns:
            Updated state
        """
        if self.x is None:
            return z
        
        # Innovation
        y = z - (self.H @ self.x)
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        self.P = (np.eye(self.ndim) - K @ self.H) @ self.P
        
        return self.x
