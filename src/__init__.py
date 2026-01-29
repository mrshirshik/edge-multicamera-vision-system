"""
Edge Multicamera Vision System
Real-time multi-camera intelligent vision system for NVIDIA edge devices
"""

__version__ = "1.0.0"
__author__ = "Vision Team"

from .vision import CameraManager, CameraConfig, ImageProcessor, ImageProcessingConfig
from .inference import InferenceEngine, InferenceConfig, ModelType, ObjectDetector
from .tracking import MultiObjectTracker, Track
from .clustering import BehaviorClusterer

__all__ = [
    "CameraManager",
    "CameraConfig",
    "ImageProcessor",
    "ImageProcessingConfig",
    "InferenceEngine",
    "InferenceConfig",
    "ModelType",
    "ObjectDetector",
    "MultiObjectTracker",
    "Track",
    "BehaviorClusterer",
]
