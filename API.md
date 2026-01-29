# API Reference

## Vision Module

### CameraManager

```python
class CameraManager:
    def __init__(self, max_streams: int = 4)
    def register_camera(self, config: CameraConfig) -> bool
    def start_streaming() -> bool
    def stop_streaming() -> None
    def get_frame(self, camera_id: str, timeout: float = 1.0) -> Optional[Dict]
    def get_all_frames(self) -> Dict[str, Dict]
    def get_status(self) -> Dict
```

### ImageProcessor

```python
class ImageProcessor:
    def __init__(self, config: ImageProcessingConfig = None)
    def preprocess(self, image: np.ndarray) -> np.ndarray
    def stitch_images(self, images: List[np.ndarray]) -> Optional[np.ndarray]
    def extract_roi(self, image: np.ndarray, roi: Tuple) -> np.ndarray
    def draw_detections(self, image: np.ndarray, detections: List[dict]) -> np.ndarray
```

## Inference Module

### ObjectDetector

```python
class ObjectDetector:
    def __init__(self, config: InferenceConfig)
    def detect(self, image: np.ndarray) -> List[Detection]
    def detect_batch(self, images: List[np.ndarray]) -> List[List[Detection]]
    def filter_detections(self, detections: List[Detection], 
                         class_ids: Optional[List[int]] = None,
                         confidence_min: float = 0.5) -> List[Detection]
```

### SemanticSegmenter

```python
class SemanticSegmenter:
    def __init__(self, config: InferenceConfig)
    def segment(self, image: np.ndarray) -> Optional[SegmentationResult]
    def get_class_mask(self, segmentation: SegmentationResult, 
                      class_id: int) -> np.ndarray
    def colorize_segmentation(self, segmentation: SegmentationResult) -> np.ndarray
```

## Tracking Module

### MultiObjectTracker

```python
class MultiObjectTracker:
    def __init__(self, max_age: int = 70, min_hits: int = 3)
    def update(self, detections: List[Dict], frame_idx: int = None) -> List[Track]
    def get_confirmed_tracks(self) -> List[Track]
    def get_all_tracks(self) -> List[Track]
    def reset() -> None
```

### Track

```python
@dataclass
class Track:
    track_id: int
    detections: List[Dict]
    first_frame: int
    last_frame: int
    bbox_history: List[Tuple]
    velocity: Tuple[float, float]
    age: int
    hits: int
    misses: int
    confirmed: bool
```

## Clustering Module

### BehaviorClusterer

```python
class BehaviorClusterer:
    def __init__(self, n_clusters: int = 3, method: str = 'kmeans')
    def cluster_tracks(self, tracks: List[Dict]) -> List[BehaviorCluster]
    def get_cluster_stats(self) -> Dict[str, Any]
```

### BehaviorCluster

```python
@dataclass
class BehaviorCluster:
    cluster_id: int
    members: List[int]
    centroid: np.ndarray
    characteristics: Dict[str, Any]
```

## Utilities

### ConfigHandler

```python
class ConfigHandler:
    def __init__(self, config_file: Optional[str] = None)
    def load(self, config_file: str) -> bool
    def save(self, config_file: str, fmt: str = 'yaml') -> bool
    def get(self, key: str, default: Any = None) -> Any
    def set(self, key: str, value: Any) -> None
```

### Data Utils

```python
def load_image(image_path: str) -> Optional[np.ndarray]
def save_image(image: np.ndarray, output_path: str) -> bool
def calculate_iou(bbox1: Tuple, bbox2: Tuple) -> float
def nms(detections: List[Dict], threshold: float = 0.4) -> List[Dict]
def format_detections(detections: List) -> List[Dict]
```

### Logging

```python
def setup_logger(name: str = "edge-vision", 
                log_level: int = logging.INFO,
                log_file: Optional[str] = None) -> logging.Logger

def get_logger(name: str = "edge-vision") -> logging.Logger
```

## Data Classes

### Detection

```python
@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    class_id: int
    class_name: str
    confidence: float
    mask: Optional[np.ndarray] = None
```

### SegmentationResult

```python
@dataclass
class SegmentationResult:
    mask: np.ndarray
    class_ids: np.ndarray
    class_names: Dict[int, str]
```

## Configuration Classes

### CameraConfig

```python
@dataclass
class CameraConfig:
    camera_id: str
    source: str
    resolution: tuple = (1920, 1080)
    fps: int = 30
    buffer_size: int = 1
```

### InferenceConfig

```python
@dataclass
class InferenceConfig:
    model_path: str
    model_type: ModelType
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    input_size: Tuple[int, int] = (640, 640)
    device: str = "cuda"
    use_tensorrt: bool = True
```

### ImageProcessingConfig

```python
@dataclass
class ImageProcessingConfig:
    resize_enabled: bool = True
    target_size: Tuple[int, int] = (640, 480)
    normalize: bool = True
    brightness_adjust: bool = False
    contrast_adjust: bool = False
```

## Enums

### ModelType

```python
class ModelType(Enum):
    YOLOV8 = "yolov8"
    YOLOV5 = "yolov5"
    EFFICIENTDET = "efficientdet"
    DEEPLAB = "deeplab"
    TENSORRT = "tensorrt"
```

## Constants

```python
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    # ... 80 classes total
]

PASCAL_VOC_CLASSES = {
    0: 'background', 1: 'aeroplane', 2: 'bicycle',
    # ... 20 classes total
}
```
