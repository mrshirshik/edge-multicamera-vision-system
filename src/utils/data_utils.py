"""
Data Utilities
Common data processing functions
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load image from file
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image array (BGR) or None
    """
    try:
        import cv2
        image = cv2.imread(image_path)
        
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        
        return image
        
    except Exception as e:
        logger.error(f"Image loading error: {e}")
        return None


def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    Save image to file
    
    Args:
        image: Image array
        output_path: Path to save image
        
    Returns:
        Success status
    """
    try:
        import cv2
        from pathlib import Path
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(output_path, image)
        
        if success:
            logger.info(f"Image saved: {output_path}")
        else:
            logger.error(f"Failed to save image: {output_path}")
        
        return success
        
    except Exception as e:
        logger.error(f"Image saving error: {e}")
        return False


def format_detections(detections: List[Any]) -> List[Dict[str, Any]]:
    """
    Format detection objects to dictionaries
    
    Args:
        detections: List of detection objects
        
    Returns:
        List of detection dicts
    """
    formatted = []
    
    for det in detections:
        formatted.append({
            'bbox': det.bbox,
            'class_id': det.class_id,
            'class_name': det.class_name,
            'confidence': det.confidence,
            'mask': det.mask
        })
    
    return formatted


def calculate_iou(bbox1: Tuple[int, int, int, int], 
                 bbox2: Tuple[int, int, int, int]) -> float:
    """
    Calculate IoU between two bounding boxes
    
    Args:
        bbox1: (x, y, w, h)
        bbox2: (x, y, w, h)
        
    Returns:
        IoU value
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # Union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def nms(detections: List[Dict], threshold: float = 0.4) -> List[Dict]:
    """
    Non-Maximum Suppression
    
    Args:
        detections: List of detection dicts
        threshold: NMS threshold
        
    Returns:
        Filtered detections
    """
    if not detections:
        return []
    
    # Sort by confidence
    sorted_dets = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)
    
    keep = []
    
    while sorted_dets:
        current = sorted_dets.pop(0)
        keep.append(current)
        
        remaining = []
        for det in sorted_dets:
            iou = calculate_iou(current['bbox'], det['bbox'])
            if iou < threshold:
                remaining.append(det)
        
        sorted_dets = remaining
    
    return keep


def get_frame_size(image: np.ndarray) -> Tuple[int, int]:
    """Get image dimensions (width, height)"""
    if image is None or image.size == 0:
        return 0, 0
    
    h, w = image.shape[:2]
    return w, h
