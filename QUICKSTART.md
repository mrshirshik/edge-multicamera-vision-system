# Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download a Model

```bash
mkdir -p models
cd models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx
cd ..
```

### 3. Run an Example

```bash
# Object detection
python examples/01_object_detection.py

# Or run full pipeline
python main.py --example full
```

## Next Steps

- Read [DOCUMENTATION.md](DOCUMENTATION.md) for complete guide
- Check [SETUP.md](SETUP.md) for detailed installation
- Review [API.md](API.md) for API reference
- Run tests: `python tests.py`

## Common Commands

```bash
# Run specific example
python main.py --example detection
python main.py --example tracking
python main.py --example clustering
python main.py --example full

# With custom config
python main.py --config config/system_config.yaml --example full

# Change output directory
python main.py --output results/ --example full

# Debug logging
python main.py --log-level DEBUG --example full

# Run tests
python tests.py

# Build Docker image
docker build -t edge-vision:latest -f docker/Dockerfile .

# Run with Docker
docker run --gpus all \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/output:/app/output \
  edge-vision:latest
```

## File Structure

```
├── src/                    # Main library
│   ├── vision/            # Camera & image processing
│   ├── inference/         # Deep learning models
│   ├── tracking/          # Object tracking
│   ├── clustering/        # Behavior analysis
│   └── utils/             # Utilities
├── config/                # Configuration files
├── examples/              # Example scripts
├── docker/                # Docker setup
├── models/                # Model storage
├── main.py               # Entry point
├── tests.py              # Unit tests
├── requirements.txt      # Dependencies
├── setup.py             # Package setup
└── docs/                # Documentation
    ├── DOCUMENTATION.md  # Full guide
    ├── SETUP.md         # Installation
    ├── API.md           # API reference
    └── QUICKSTART.md    # This file
```

## Troubleshooting

### "Module not found" error
```bash
# Make sure src is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### "Model not found" error
```bash
# Download model
mkdir -p models
wget -O models/yolov8n.onnx https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx
```

### CUDA/GPU issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU if GPU unavailable
# Edit config to use device: "cpu"
```

## Features Overview

| Component | Status | Features |
|-----------|--------|----------|
| **Vision** | ✓ | Multi-camera streams, Image processing |
| **Detection** | ✓ | YOLOv8, YOLOv5, EfficientDet |
| **Tracking** | ✓ | DeepSORT, IoU matching |
| **Clustering** | ✓ | K-Means, DBSCAN behavior analysis |
| **Segmentation** | ✓ | DeepLabV3+ semantic segmentation |
| **Utilities** | ✓ | Logging, Config, Data processing |
| **Docker** | ✓ | GPU support, Compose orchestration |

## Performance Tips

1. **Use TensorRT** for 2-3x speed improvement
   - Convert models: `trtexec --onnx=model.onnx --saveEngine=model.trt`
   - Enable in config: `use_tensorrt: true`

2. **Reduce input resolution** for faster processing
   - Change `target_size: [640, 480]` to `[448, 336]`

3. **Use smaller models**
   - `yolov8n.onnx` (fastest) vs `yolov8l.onnx` (most accurate)

4. **Process every Nth frame**
   - Skip frames: `if frame_idx % 3 == 0: detect()`

5. **Batch processing**
   - Process multiple frames at once: `detect_batch(images)`

## API Example

```python
from src.vision import CameraManager, CameraConfig
from src.inference import ObjectDetector, InferenceConfig, ModelType
from src.tracking import MultiObjectTracker

# Initialize
manager = CameraManager()
detector = ObjectDetector(InferenceConfig(
    model_path="models/yolov8n.onnx",
    model_type=ModelType.YOLOV8
))
tracker = MultiObjectTracker()

# Register camera
manager.register_camera(CameraConfig(
    camera_id="cam0",
    source=0  # Webcam
))

# Start streaming
manager.start_streaming()

# Process frame
frame = manager.get_frame("cam0")
if frame:
    detections = detector.detect(frame['frame'])
    tracks = tracker.update([
        {'bbox': d.bbox, 'class': d.class_id, 'conf': d.confidence}
        for d in detections
    ])
    
    print(f"Found {len(detections)} objects, tracking {len(tracks)} targets")

# Cleanup
manager.stop_streaming()
```

## Support & Docs

- **Full Documentation**: [DOCUMENTATION.md](DOCUMENTATION.md)
- **API Reference**: [API.md](API.md)
- **Setup Guide**: [SETUP.md](SETUP.md)
- **Examples**: `examples/` folder
- **Tests**: Run `python tests.py`

## License

MIT License - See LICENSE file for details
