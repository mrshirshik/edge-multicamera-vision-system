"""
Camera Manager Module
Handles multi-camera stream ingestion from RTSP/USB sources
"""

import logging
from typing import List, Optional, Dict, Any, Union
import threading
import queue
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Configuration for a single camera"""
    camera_id: str
    source: Union[str, int]  # RTSP URL or device index
    resolution: tuple = (1920, 1080)
    fps: int = 30
    buffer_size: int = 1


class CameraManager:
    """Manages multiple camera streams"""
    
    def __init__(self, max_streams: int = 4):
        """
        Initialize camera manager
        
        Args:
            max_streams: Maximum concurrent camera streams
        """
        self.max_streams = max_streams
        self.cameras: Dict[str, CameraConfig] = {}
        self.streams: Dict[str, queue.Queue] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self.running = False
        
    def register_camera(self, config: CameraConfig) -> bool:
        """
        Register a new camera
        
        Args:
            config: Camera configuration
            
        Returns:
            Success status
        """
        if len(self.cameras) >= self.max_streams:
            logger.error(f"Maximum streams ({self.max_streams}) reached")
            return False
            
        self.cameras[config.camera_id] = config
        self.streams[config.camera_id] = queue.Queue(maxsize=config.buffer_size)
        logger.info(f"Camera {config.camera_id} registered: {config.source}")
        return True
    
    def start_streaming(self) -> bool:
        """Start all registered camera streams"""
        if self.running:
            logger.warning("Streaming already running")
            return False
            
        self.running = True
        
        for camera_id, config in self.cameras.items():
            thread = threading.Thread(
                target=self._stream_worker,
                args=(camera_id, config),
                daemon=True,
                name=f"camera-{camera_id}"
            )
            thread.start()
            self.threads[camera_id] = thread
            
        logger.info(f"Started {len(self.cameras)} camera streams")
        return True
    
    def _stream_worker(self, camera_id: str, config: CameraConfig) -> None:
        """Worker thread for individual camera stream"""
        import cv2
        
        cap = None
        try:
            cap = cv2.VideoCapture(config.source)
            if not cap.isOpened():
                logger.error(f"Failed to open camera {camera_id}: {config.source}")
                return
            
            # Set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.resolution[1])
            cap.set(cv2.CAP_PROP_FPS, config.fps)
            
            logger.info(f"Camera {camera_id} stream started")
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame from {camera_id}")
                    continue
                
                try:
                    self.streams[camera_id].put_nowait({
                        'camera_id': camera_id,
                        'frame': frame,
                        'timestamp': None
                    })
                except queue.Full:
                    pass  # Drop oldest frame
                    
        except Exception as e:
            logger.error(f"Camera stream error {camera_id}: {e}")
        finally:
            if cap is not None:
                cap.release()
    
    def get_frame(self, camera_id: str, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        Get latest frame from camera
        
        Args:
            camera_id: Camera identifier
            timeout: Maximum wait time
            
        Returns:
            Frame dict or None
        """
        if camera_id not in self.streams:
            return None
            
        try:
            return self.streams[camera_id].get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_all_frames(self) -> Dict[str, Dict[str, Any]]:
        """Get latest frames from all cameras"""
        frames = {}
        for camera_id in self.cameras:
            frame = self.get_frame(camera_id, timeout=0.1)
            if frame:
                frames[camera_id] = frame
        return frames
    
    def stop_streaming(self) -> None:
        """Stop all camera streams"""
        self.running = False
        for thread in self.threads.values():
            thread.join(timeout=5)
        logger.info("All camera streams stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all cameras"""
        return {
            'running': self.running,
            'total_cameras': len(self.cameras),
            'cameras': list(self.cameras.keys())
        }
