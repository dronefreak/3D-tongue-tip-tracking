FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for dlib and OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    libboost-all-dev \
    libx11-dev \
    libgtk-3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY *.py ./
COPY *.m ./
COPY CODE_ANALYSIS.md README.md LICENSE ./

# Create directories for data
RUN mkdir -p /data/input /data/output /data/models

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Volume mounts for data
VOLUME ["/data/input", "/data/output", "/data/models"]

# Default command shows help
CMD ["python", "facial_landmarks_video.py", "-h"]
