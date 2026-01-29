"""
Main Pipeline Entry Point
Central entry point for the vision system
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logger, ConfigHandler
from src.vision import CameraManager, CameraConfig
from src.inference import ObjectDetector, InferenceConfig, ModelType
from src.tracking import MultiObjectTracker
from src.clustering import BehaviorClusterer
import numpy as np


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Edge Multicamera Vision System"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/system_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--example",
        type=str,
        choices=["detection", "tracking", "clustering", "full"],
        default="full",
        help="Example to run"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger(
        name="edge-vision",
        log_level=getattr(__import__('logging'), args.log_level),
        log_file=f"{args.output}/vision_system.log"
    )
    
    logger.info("=" * 60)
    logger.info("Edge Multicamera Vision System Started")
    logger.info("=" * 60)
    
    # Load config
    config = ConfigHandler(args.config)
    logger.info(f"Configuration loaded from {args.config}")
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Run example
    if args.example == "detection":
        run_detection_example(logger)
    elif args.example == "tracking":
        run_tracking_example(logger)
    elif args.example == "clustering":
        run_clustering_example(logger)
    else:  # full
        run_full_pipeline(logger, config)


def run_detection_example(logger):
    """Run object detection example"""
    logger.info("Running object detection example...")
    
    from src.vision import ImageProcessor, ImageProcessingConfig
    from src.inference import ObjectDetector, InferenceConfig, ModelType
    
    # Create processor and detector
    processor = ImageProcessor(ImageProcessingConfig())
    config = InferenceConfig(
        model_path="models/yolov8n.onnx",
        model_type=ModelType.YOLOV8
    )
    detector = ObjectDetector(config)
    
    # Mock frame
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Detect
    detections = detector.detect(image)
    logger.info(f"Found {len(detections)} detections")


def run_tracking_example(logger):
    """Run tracking example"""
    logger.info("Running tracking example...")
    
    from src.tracking import MultiObjectTracker
    
    tracker = MultiObjectTracker()
    
    for frame_idx in range(50):
        # Mock detections
        detections = [
            {'bbox': (100 + frame_idx, 100, 50, 50), 'class': 0, 'conf': 0.9},
            {'bbox': (200, 150 + frame_idx, 60, 60), 'class': 0, 'conf': 0.85}
        ]
        
        tracks = tracker.update(detections, frame_idx)
        
        if frame_idx % 10 == 0:
            logger.info(f"Frame {frame_idx}: {len(tracks)} active tracks")


def run_clustering_example(logger):
    """Run clustering example"""
    logger.info("Running clustering example...")
    
    from src.clustering import BehaviorClusterer
    
    clusterer = BehaviorClusterer(n_clusters=3)
    
    # Create mock tracks
    tracks = []
    for i in range(15):
        velocity = i // 5  # 0-2
        bbox_hist = [
            (100 + i*30, 100 + velocity*j, 40, 40)
            for j in range(5)
        ]
        tracks.append({
            'track_id': i,
            'bbox_history': bbox_hist
        })
    
    # Cluster
    clusters = clusterer.cluster_tracks(tracks)
    logger.info(f"Generated {len(clusters)} clusters")
    
    for cluster in clusters:
        logger.info(f"  Cluster {cluster.cluster_id}: {cluster.characteristics}")


def run_full_pipeline(logger, config):
    """Run complete pipeline"""
    logger.info("Running complete vision pipeline...")
    
    # Initialize components
    camera_manager = CameraManager(max_streams=2)
    detector = ObjectDetector(InferenceConfig(
        model_path="models/yolov8n.onnx",
        model_type=ModelType.YOLOV8
    ))
    tracker = MultiObjectTracker()
    clusterer = BehaviorClusterer(n_clusters=3)
    
    logger.info("All components initialized")
    
    # Process frames
    num_frames = 100
    for frame_idx in range(num_frames):
        # Mock frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Detect
        detections = detector.detect(frame)
        
        # Track
        det_dicts = [
            {'bbox': d.bbox, 'class': d.class_id, 'conf': d.confidence}
            for d in detections
        ]
        tracks = tracker.update(det_dicts, frame_idx)
        
        if frame_idx % 20 == 0:
            logger.info(f"Frame {frame_idx}: {len(detections)} detections, {len(tracks)} tracks")
    
    # Final clustering
    all_tracks = tracker.get_all_tracks()
    track_data = [
        {'track_id': t.track_id, 'bbox_history': t.bbox_history}
        for t in all_tracks
    ]
    clusters = clusterer.cluster_tracks(track_data)
    
    logger.info(f"Pipeline complete: {len(clusters)} behavior clusters identified")


if __name__ == "__main__":
    main()
