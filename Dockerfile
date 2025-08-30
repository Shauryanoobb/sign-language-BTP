# Use official Python base image (slim to keep it light)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install required Python packages
RUN pip install --no-cache-dir \
    tensorflow==2.19.0 \
    opencv-python-headless==4.9.0.80 \
    numpy==1.26.4 \
    mediapipe==0.10.18 \
    pillow

# Copy project files into the container
COPY . .

# Default command
CMD ["python", "main.py"]
