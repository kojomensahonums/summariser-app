# ====================================================
# 1️⃣ Base image with CUDA and Python
# ====================================================
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

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
# 3️⃣ Pre-download models during build (to avoid slow startup)
# ====================================================
RUN python3 - <<'EOF'
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer, AutoModelForCausalLM
print("Downloading models to cache...")
# Whisper ASR
whisper_id = "distil-whisper/distil-medium.en"
AutoModelForSpeechSeq2Seq.from_pretrained(whisper_id)
AutoProcessor.from_pretrained(whisper_id)
# Cohere LLM
llm_id = "CohereLabs/c4ai-command-r7b-12-2024"
AutoTokenizer.from_pretrained(llm_id)
AutoModelForCausalLM.from_pretrained(llm_id)
print("✅ Model pre-download complete.")
EOF

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
