#!/usr/bin/env python
"""
Setup script for Edge Multicamera Vision System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="edge-multicamera-vision",
    version="1.0.0",
    author="Vision Team",
    description="Real-time multi-camera intelligent vision system for NVIDIA edge devices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/edge-multicamera-vision-system",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "torch>=1.10.0",
        "onnxruntime>=1.10.0",
        "scikit-learn>=0.24.0",
        "PyYAML>=5.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0",
            "pylint>=2.10.0",
        ],
        "tensorrt": [
            "tensorrt>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "edge-vision=src.cli:main",
        ],
    },
)
