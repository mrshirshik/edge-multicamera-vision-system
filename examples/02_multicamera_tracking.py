"""
Example 2: Multi-Camera Stream Processing
Process multiple camera streams with tracking
"""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vision import CameraManager, CameraConfig
from src.tracking import MultiObjectTracker
from src.utils import setup_logger

logger = setup_logger("example_multicam", log_file="logs/example_multicam.log")


def main():
    """Run multi-camera processing example"""
    
    # Create camera manager
    camera_manager = CameraManager(max_streams=2)
    
    # Register cameras
    cameras = [
        CameraConfig(
            camera_id="cam0",
            source=0,  # Webcam
            resolution=(1280, 720),
            fps=30
        ),
        CameraConfig(
            camera_id="cam1",
            source="rtsp://example.com/stream",
            resolution=(1280, 720),
            fps=30
        )
    ]
    
    for cam_config in cameras:
        success = camera_manager.register_camera(cam_config)
        if success:
            logger.info(f"Camera {cam_config.camera_id} registered")
    
    # Start streaming
    camera_manager.start_streaming()
    
    # Create tracker
    tracker = MultiObjectTracker(max_age=70, min_hits=3)
    
    # Process frames
    logger.info("Processing camera streams...")
    
    try:
        for frame_idx in range(100):
            # Get frames from all cameras
            frames = camera_manager.get_all_frames()
            
            if not frames:
                logger.warning("No frames received")
                continue
            
            logger.info(f"Frame {frame_idx}: Got {len(frames)} camera(s)")
            
            for camera_id, frame_data in frames.items():
                logger.debug(f"  {camera_id}: {frame_data['frame'].shape}")
                
                # Mock detection data
                mock_detections = [
                    {'bbox': (100, 100, 50, 50), 'class': 0, 'conf': 0.9},
                    {'bbox': (200, 150, 60, 60), 'class': 0, 'conf': 0.85}
                ]
                
                # Update tracker
                tracks = tracker.update(mock_detections, frame_idx)
                logger.info(f"  Active tracks: {len(tracks)}")
            
            time.sleep(0.033)  # ~30 FPS
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        camera_manager.stop_streaming()
        logger.info("Camera streams stopped")


if __name__ == "__main__":
    main()
