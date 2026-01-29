"""
Example 4: Complete Vision Pipeline
Integrated example with all components
"""

import sys
from pathlib import Path
import time
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vision import CameraManager, CameraConfig, ImageProcessor, ImageProcessingConfig
from src.inference import ObjectDetector, InferenceConfig, ModelType
from src.tracking import MultiObjectTracker
from src.clustering import BehaviorClusterer
from src.utils import setup_logger, ConfigHandler

logger = setup_logger("example_full_pipeline", log_file="logs/example_pipeline.log")


class VisionPipeline:
    """Complete vision pipeline"""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize pipeline"""
        
        # Load config
        self.config = ConfigHandler(config_file)
        logger.info("Configuration loaded")
        
        # Initialize components
        self.camera_manager = CameraManager(max_streams=2)
        self.image_processor = ImageProcessor()
        self.detector = ObjectDetector(InferenceConfig(
            model_path="models/yolov8n.onnx",
            model_type=ModelType.YOLOV8
        ))
        self.tracker = MultiObjectTracker()
        self.clusterer = BehaviorClusterer(n_clusters=3)
        
        logger.info("Pipeline components initialized")
    
    def register_cameras(self):
        """Register camera streams"""
        cameras = [
            CameraConfig(camera_id="cam0", source=0),
            CameraConfig(camera_id="cam1", source=1)
        ]
        
        for cam in cameras:
            if self.camera_manager.register_camera(cam):
                logger.info(f"Registered {cam.camera_id}")
    
    def process_frame(self, frame, frame_idx: int):
        """Process single frame"""
        
        # Preprocess
        processed = self.image_processor.preprocess(frame)
        
        # Detect
        detections = self.detector.detect(frame)
        
        # Track
        det_dicts = [
            {'bbox': d.bbox, 'class': d.class_id, 'conf': d.confidence}
            for d in detections
        ]
        tracks = self.tracker.update(det_dicts, frame_idx)
        
        return {
            'frame': frame,
            'detections': detections,
            'tracks': tracks
        }
    
    def cluster_and_analyze(self):
        """Cluster tracks and analyze behavior"""
        
        all_tracks = self.tracker.get_all_tracks()
        
        # Convert tracks to cluster input format
        track_data = [
            {
                'track_id': t.track_id,
                'bbox_history': t.bbox_history
            }
            for t in all_tracks
        ]
        
        clusters = self.clusterer.cluster_tracks(track_data)
        
        logger.info(f"Clustered {len(all_tracks)} tracks into {len(clusters)} clusters")
        
        return clusters
    
    def run(self, num_frames: int = 100):
        """Run pipeline"""
        
        logger.info("Starting pipeline...")
        
        # Register cameras (use mock for demo)
        self.register_cameras()
        
        # Process frames
        import numpy as np
        
        for frame_idx in range(num_frames):
            # Mock frame (in real scenario, get from camera)
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Process
            result = self.process_frame(frame, frame_idx)
            
            if frame_idx % 10 == 0:
                logger.info(f"Frame {frame_idx}: {len(result['detections'])} detections")
        
        # Final clustering
        clusters = self.cluster_and_analyze()
        
        logger.info(f"Pipeline execution complete")
        
        return clusters


def main():
    """Run full pipeline example"""
    
    pipeline = VisionPipeline(config_file="config/system_config.yaml")
    
    clusters = pipeline.run(num_frames=50)
    
    for cluster in clusters:
        logger.info(f"Cluster {cluster.cluster_id}: {cluster.characteristics}")


if __name__ == "__main__":
    main()
