"""
Image Processing Module
Handles image preprocessing, enhancement, and stitching
"""

import numpy as np
import logging
from typing import Tuple, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ImageProcessingConfig:
    """Configuration for image processing"""
    resize_enabled: bool = True
    target_size: Tuple[int, int] = (640, 480)
    normalize: bool = True
    brightness_adjust: bool = False
    contrast_adjust: bool = False


class ImageProcessor:
    """Processes and enhances images for inference"""
    
    def __init__(self, config: Optional[ImageProcessingConfig] = None):
        """
        Initialize image processor
        
        Args:
            config: Processing configuration
        """
        self.config = config or ImageProcessingConfig()
    
    def preprocess(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Preprocess image for inference
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Processed image
        """
        if image is None or image.size == 0:
            logger.error("Invalid image")
            return None
        
        processed = image.copy()
        
        # Resize
        if self.config.resize_enabled:
            processed = self._resize(processed, self.config.target_size)
        
        # Brightness/Contrast adjustment
        if self.config.brightness_adjust:
            processed = self._adjust_brightness(processed)
        
        if self.config.contrast_adjust:
            processed = self._adjust_contrast(processed)
        
        # Normalize
        if self.config.normalize:
            processed = processed.astype(np.float32) / 255.0
        
        return processed
    
    def _resize(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image"""
        import cv2
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    def _adjust_brightness(self, image: np.ndarray, alpha: float = 0.9) -> np.ndarray:
        """Adjust brightness"""
        return np.clip(image * alpha, 0, 255).astype(np.uint8)
    
    def _adjust_contrast(self, image: np.ndarray, alpha: float = 1.1) -> np.ndarray:
        """Adjust contrast"""
        return np.clip(image * alpha, 0, 255).astype(np.uint8)
    
    def stitch_images(self, images: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Stitch multiple images together
        
        Args:
            images: List of images to stitch
            
        Returns:
            Stitched image or None
        """
        if not images or len(images) < 2:
            logger.warning("Need at least 2 images for stitching")
            return None
        
        try:
            import cv2
            stitcher = cv2.Stitcher.create() if hasattr(cv2, 'Stitcher') else None
            if stitcher is None:
                logger.warning("Stitcher not available in this OpenCV version")
                return None
            status, result = stitcher.stitch(images)
            
            if status == cv2.Stitcher_OK:
                return result
            else:
                logger.error(f"Stitching failed: {status}")
                return None
        except Exception as e:
            logger.error(f"Image stitching error: {e}")
            return None
    
    def extract_roi(self, image: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract region of interest
        
        Args:
            image: Input image
            roi: (x, y, width, height)
            
        Returns:
            ROI image
        """
        x, y, w, h = roi
        return image[y:y+h, x:x+w]
    
    def draw_detections(self, image: np.ndarray, detections: List[dict]) -> np.ndarray:
        """
        Draw detection boxes on image
        
        Args:
            image: Input image
            detections: List of detection dicts with 'bbox', 'class', 'conf'
            
        Returns:
            Image with drawn detections
        """
        import cv2
        
        result = image.copy()
        
        for det in detections:
            bbox = det.get('bbox')
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                label = f"{det.get('class', 'obj')} {det.get('conf', 0):.2f}"
                cv2.putText(result, label, (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return result
