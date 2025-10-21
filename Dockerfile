# ====================================================
# 1️⃣ Base image with CUDA and Python
# ====================================================
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

# Accept Hugging Face token at build time
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Install system dependencies (Python + FFmpeg + Git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ffmpeg git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ====================================================
# 2️⃣ Copy dependencies and install
# ====================================================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ====================================================
# 3️⃣ Pre-download models during build
# ====================================================
COPY preload_models.py .
RUN python3 preload_models.py

# ====================================================
# 4️⃣ Copy application code
# ====================================================
COPY . .

# Expose Streamlit port
EXPOSE 8080

# ====================================================
# 5️⃣ Launch Streamlit app
# ====================================================
ENTRYPOINT ["streamlit", "run", "new_app.py", "--server.port=8080", "--server.address=0.0.0.0"]

