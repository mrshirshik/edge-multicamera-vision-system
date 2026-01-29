"""
Example 1: Basic Object Detection Pipeline
Simple object detection with visualization
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from src.inference import ObjectDetector, InferenceConfig, ModelType
from src.vision import ImageProcessor, ImageProcessingConfig
from src.utils import setup_logger, get_logger

logger = setup_logger("example_detection", log_file="logs/example_detection.log")


def main():
    """Run basic detection example"""
    
    # Create image processor
    proc_config = ImageProcessingConfig(
        resize_enabled=True,
        target_size=(640, 480),
        normalize=True
    )
    processor = ImageProcessor(proc_config)
    
    # Create detector
    inf_config = InferenceConfig(
        model_path="models/yolov8n.onnx",
        model_type=ModelType.YOLOV8,
        confidence_threshold=0.5,
        use_tensorrt=False
    )
    detector = ObjectDetector(inf_config)
    
    logger.info("Detection pipeline initialized")
    
    # Create sample image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Preprocess
    processed = processor.preprocess(image)
    if processed is None:
        logger.error("Preprocessing failed")
        return
    logger.info(f"Image preprocessed: {processed.shape}")
    
    # Detect
    detections = detector.detect(image)
    logger.info(f"Found {len(detections)} detections")
    
    for det in detections:
        logger.info(f"  - {det.class_name}: {det.confidence:.2f} at {det.bbox}")
    
    # Visualize
    if detections:
        vis_image = processor.draw_detections(image, [
            {
                'bbox': d.bbox,
                'class': d.class_name,
                'conf': d.confidence
            }
            for d in detections
        ])
        
        cv2.imwrite("output/detection_result.jpg", vis_image)
        logger.info("Result saved to output/detection_result.jpg")


if __name__ == "__main__":
    main()
