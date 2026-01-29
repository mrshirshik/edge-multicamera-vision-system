# Setup and Installation Guide

## Prerequisites

- Python 3.8 or higher
- NVIDIA GPU or NVIDIA Jetson device (Orin/Xavier recommended)
- CUDA 11.8+ (for GPU acceleration)
- 4GB+ RAM, 2GB+ VRAM

## Installation Methods

### Method 1: Pip Installation (Development)

```bash
# Clone repository
git clone https://github.com/your-org/edge-multicamera-vision-system.git
cd edge-multicamera-vision-system

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Or install directly
pip install -r requirements.txt
```

### Method 2: Docker Installation

```bash
# Build Docker image
docker build -t edge-vision:latest -f docker/Dockerfile .

# Run with Docker
docker run --gpus all \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/output:/app/output \
  edge-vision:latest

# Or use Docker Compose
docker-compose -f docker/docker-compose.yml up
```

### Method 3: Docker on Jetson

```bash
# Enable NGC registry
docker login nvcr.io

# Build with Jetson base image
docker build \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-cuda:11.4.14-cudnn8-runtime \
  -t edge-vision-jetson:latest \
  -f docker/Dockerfile.jetson .

# Run
docker run --runtime nvidia \
  --env NVIDIA_VISIBLE_DEVICES=all \
  --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v $(pwd)/models:/app/models \
  edge-vision-jetson:latest
```

## Post-Installation Setup

### 1. Download Models

```bash
# Create models directory
mkdir -p models

# Download YOLOv8 nano (ONNX)
cd models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.onnx

# Download YOLOv8 Seg (Segmentation)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.onnx

cd ..
```

### 2. Configure System

Edit `config/system_config.yaml`:

```yaml
system:
  device: "nvidia_jetson"  # or "gpu" for desktop GPU

camera:
  cameras:
    - id: "cam0"
      source: 0  # or "rtsp://192.168.1.100/stream"
      resolution: [1920, 1080]
      fps: 30

inference:
  object_detection:
    model_path: "models/yolov8n.onnx"
    confidence_threshold: 0.5
```

### 3. Test Installation

```bash
# Test basic imports
python3 -c "
import sys
sys.path.insert(0, '.')
from src.vision import CameraManager
from src.inference import ObjectDetector
from src.tracking import MultiObjectTracker
print('✓ All modules imported successfully')
"

# Run quick test
python3 examples/01_object_detection.py
```

## Environment Setup

### NVIDIA Jetson Setup

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install CUDA dev tools
sudo apt-get install -y cuda-toolkit-11-4
sudo apt-get install -y libcudnn8 libcudnn8-dev

# Install cuDNN
# Download from NVIDIA (requires account)
sudo dpkg -i cudnn*.deb
```

### GPU Setup (Desktop)

```bash
# Install NVIDIA drivers
ubuntu-drivers install nvidia

# Install CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo bash cuda_11.8.0_520.61.05_linux.run

# Add to PATH
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

## Verification

### Check GPU/CUDA Availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Verify Model Loading

```python
import sys
sys.path.insert(0, '.')

from src.inference import ObjectDetector, InferenceConfig, ModelType

config = InferenceConfig(
    model_path="models/yolov8n.onnx",
    model_type=ModelType.YOLOV8,
    use_tensorrt=False
)

detector = ObjectDetector(config)
print(f"Model loaded: {detector.initialized}")
print(f"Model info: {detector.get_info()}")
```

### Test Camera Access

```python
import sys
sys.path.insert(0, '.')

from src.vision import CameraManager, CameraConfig

manager = CameraManager()
manager.register_camera(CameraConfig(
    camera_id="test_cam",
    source=0  # Webcam
))
manager.start_streaming()

frame = manager.get_frame("test_cam")
print(f"Frame captured: {frame is not None}")

manager.stop_streaming()
```

## Troubleshooting

### CUDA Not Found

```bash
# Check CUDA installation
nvcc --version
ldconfig -p | grep libcuda

# Add to PATH if missing
export CUDA_PATH=/usr/local/cuda-11.8
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
```

### Model Loading Errors

```bash
# Verify ONNX model
python3 -c "
import onnx
model = onnx.load('models/yolov8n.onnx')
onnx.checker.check_model(model)
print('✓ Model is valid')
"

# Check ORT providers
python3 -c "
import onnxruntime as ort
print('Available providers:', ort.get_available_providers())
"
```

### Camera Connection Issues

```bash
# List available cameras
v4l2-ctl --list-devices

# Test RTSP stream
ffprobe rtsp://your-camera-url/stream

# Check ports
netstat -tlnp | grep 8554  # RTSP default port
```

### Memory Issues

```bash
# Check available memory
free -h

# Check GPU memory (if using NVIDIA)
nvidia-smi

# Reduce batch size in config
# Reduce model size (use yolov8n instead of yolov8l)
# Enable frame skipping
```

## Performance Tuning

### Enable TensorRT (Fastest)

```yaml
inference:
  object_detection:
    use_tensorrt: true
    model_path: "models/yolov8n.trt"
```

Convert ONNX to TensorRT:

```bash
# If TensorRT installed
trtexec --onnx=models/yolov8n.onnx --saveEngine=models/yolov8n.trt
```

### Optimize for Edge Devices

```yaml
processing:
  target_size: [640, 480]  # Reduce from 1920x1080

inference:
  object_detection:
    model_path: "models/yolov8n.onnx"  # Use nano model
    confidence_threshold: 0.5

camera:
  fps: 15  # Reduce from 30
```

### Batch Processing

```python
# Process multiple frames at once
detector = ObjectDetector(config)
results = detector.detect_batch([frame1, frame2, frame3])
```

## Next Steps

1. Review [DOCUMENTATION.md](DOCUMENTATION.md) for complete guide
2. Check [API.md](API.md) for API reference
3. Run examples in `examples/` directory
4. Customize configuration for your use case
5. Deploy to production

## Support

- **Issues**: Open GitHub issue with:
  - System info (`nvidia-smi`, `python --version`)
  - Error messages
  - Reproduction steps

- **Documentation**: See [DOCUMENTATION.md](DOCUMENTATION.md)
- **Examples**: See `examples/` directory
