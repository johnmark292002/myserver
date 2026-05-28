FROM python:3.10-slim

# Install system dependencies: FFmpeg + build tools (for Real-ESRGAN)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Download Real-ESRGAN-ncnn-vulkan (pre-built binary)
WORKDIR /opt
RUN wget https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/v0.2.0/realesrgan-ncnn-vulkan-20220410-ubuntu.zip \
    && unzip realesrgan-ncnn-vulkan-20220410-ubuntu.zip \
    && mv realesrgan-ncnn-vulkan-20220410-ubuntu /opt/realesrgan \
    && rm realesrgan-ncnn-vulkan-20220410-ubuntu.zip \
    && chmod +x /opt/realesrgan/realesrgan-ncnn-vulkan

# Set working directory
WORKDIR /app

# Copy Python dependencies and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the server code
COPY . .

# Environment variables (can be overridden on Render)
ENV PORT=10000
ENV WORK_DIR=/tmp/aive_work
ENV OUTPUT_DIR=/tmp/aive_out
ENV TTL_SECONDS=3600
ENV REALESRGAN_BIN=/opt/realesrgan/realesrgan-ncnn-vulkan

# Expose the port
EXPOSE $PORT

# Start Gunicorn (or directly Flask for testing)
CMD gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --workers 2
