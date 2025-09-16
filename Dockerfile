# Use official Python base image (slim to keep it light)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN pip install --no-cache-dir \
    tensorflow==2.19.0 \
    opencv-python-headless==4.9.0.80 \
    numpy==1.26.4

# Copy model and code into container
COPY main.py .
COPY z.png .

# Run the script
CMD ["python", "main.py"]

