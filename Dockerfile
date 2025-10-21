# Use a CUDA-enabled base image for GPU support
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Set up environment paths for CUDA
ENV PATH="/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

# Prevent interactive prompts during installs
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, FFmpeg (for audio), and other system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ffmpeg git \
 && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose Streamlit port
EXPOSE 8080

# Run Streamlit app
ENTRYPOINT ["streamlit", "run", "new_app.py", "--server.port=8080", "--server.address=0.0.0.0"]

