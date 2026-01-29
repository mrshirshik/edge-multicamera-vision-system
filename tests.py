"""
Unit Tests for Edge Vision System
"""

import sys
from pathlib import Path
import unittest
import numpy as np
import tempfile

sys.path.insert(0, str(Path(__file__).parent))

from src.vision import CameraConfig, ImageProcessor, ImageProcessingConfig
from src.inference import InferenceConfig, ObjectDetector, ModelType
from src.tracking import MultiObjectTracker, Track
from src.clustering import BehaviorClusterer
from src.utils import ConfigHandler, setup_logger, calculate_iou


class TestVisionModule(unittest.TestCase):
    """Test vision module"""
    
    def test_image_processor_creation(self):
        """Test image processor initialization"""
        processor = ImageProcessor()
        self.assertIsNotNone(processor)
    
    def test_image_preprocessing(self):
        """Test image preprocessing"""
        processor = ImageProcessor(ImageProcessingConfig(
            resize_enabled=True,
            target_size=(640, 480)
        ))
        
        image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        processed = processor.preprocess(image)
        
        self.assertIsNotNone(processed)
        if processed is not None:
            self.assertEqual(processed.shape, (480, 640, 3))


class TestInferenceModule(unittest.TestCase):
    """Test inference module"""
    
    def test_detector_config(self):
        """Test detector configuration"""
        config = InferenceConfig(
            model_path="dummy.onnx",
            model_type=ModelType.YOLOV8,
            confidence_threshold=0.5
        )
        
        self.assertEqual(config.confidence_threshold, 0.5)
        self.assertEqual(config.model_type, ModelType.YOLOV8)


class TestTrackingModule(unittest.TestCase):
    """Test tracking module"""
    
    def test_tracker_initialization(self):
        """Test tracker creation"""
        tracker = MultiObjectTracker(max_age=70, min_hits=3)
        self.assertEqual(tracker.max_age, 70)
        self.assertEqual(tracker.min_hits, 3)
    
    def test_tracker_update(self):
        """Test tracker update"""
        tracker = MultiObjectTracker()
        
        # First detection
        detections = [
            {'bbox': (100, 100, 50, 50), 'class': 0, 'conf': 0.9}
        ]
        tracks = tracker.update(detections, frame_idx=0)
        
        # Should not be confirmed yet (min_hits=3)
        self.assertEqual(len(tracker.get_confirmed_tracks()), 0)
    
    def test_iou_calculation(self):
        """Test IoU calculation"""
        bbox1 = (0, 0, 100, 100)
        bbox2 = (50, 50, 100, 100)
        
        iou = calculate_iou(bbox1, bbox2)
        
        # Overlapping boxes should have IoU > 0
        self.assertGreater(iou, 0)
        self.assertLess(iou, 1)


class TestClusteringModule(unittest.TestCase):
    """Test clustering module"""
    
    def test_clusterer_initialization(self):
        """Test clusterer creation"""
        clusterer = BehaviorClusterer(n_clusters=3)
        self.assertEqual(clusterer.n_clusters, 3)
    
    def test_clustering(self):
        """Test track clustering"""
        clusterer = BehaviorClusterer(n_clusters=2, method='kmeans')
        
        # Create mock tracks
        tracks = [
            {
                'track_id': i,
                'bbox_history': [
                    (100 + i*10, 100, 40, 40),
                    (102 + i*10, 100, 40, 40),
                    (104 + i*10, 100, 40, 40)
                ]
            }
            for i in range(10)
        ]
        
        clusters = clusterer.cluster_tracks(tracks)
        
        self.assertGreaterEqual(len(clusters), 0)
        self.assertLessEqual(len(clusters), 2)


class TestUtilityModule(unittest.TestCase):
    """Test utility module"""
    
    def test_config_handler(self):
        """Test configuration handler"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
system:
  name: test
  device: gpu
inference:
  threshold: 0.5
""")
            config_file = f.name
        
        try:
            config = ConfigHandler(config_file)
            
            self.assertEqual(config.get('system.name'), 'test')
            self.assertEqual(config.get('inference.threshold'), 0.5)
            
        finally:
            Path(config_file).unlink()
    
    def test_logger_setup(self):
        """Test logger setup"""
        logger = setup_logger("test_logger")
        self.assertIsNotNone(logger)
        
        # Should not raise exception
        logger.info("Test message")


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_pipeline_flow(self):
        """Test basic pipeline flow"""
        
        # Initialize components
        tracker = MultiObjectTracker()
        clusterer = BehaviorClusterer(n_clusters=3)
        
        # Simulate frames
        all_detections = []
        
        for frame_idx in range(50):
            # Mock detections
            detections = [
                {'bbox': (100 + frame_idx, 100, 50, 50), 'class': 0, 'conf': 0.9},
                {'bbox': (200, 150 + frame_idx, 60, 60), 'class': 1, 'conf': 0.85}
            ]
            
            # Update tracker
            tracks = tracker.update(detections, frame_idx)
            all_detections.extend(detections)
        
        # Get final tracks
        final_tracks = tracker.get_all_tracks()
        
        self.assertGreater(len(final_tracks), 0)


def run_tests():
    """Run all tests"""
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestVisionModule))
    suite.addTests(loader.loadTestsFromTestCase(TestInferenceModule))
    suite.addTests(loader.loadTestsFromTestCase(TestTrackingModule))
    suite.addTests(loader.loadTestsFromTestCase(TestClusteringModule))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityModule))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
