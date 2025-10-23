🎧 Audio Summariser & Repurposer
---
A lightweight Streamlit app that turns audio into content — automatically.
Upload or paste a downloadable audio link, and the app will:

🔊 Transcribe the audio using Whisper

🧠 Summarise the transcript for quick insights

💬 Repurpose it into social media posts, headlines, and tweet threads

All models — including distil-Whisper and Microsoft/phi-2 for text generation — are loaded at runtime on Google Cloud Run, so everything runs smoothly in one containerized environment.

🚀 Features
---
Audio upload or URL download

Speech-to-text transcription (Whisper-based)

Summarisation & content generation powered by phi-2

Streamlit interface for easy interaction

Fully containerized app — no external frameworks or servers needed

Deployed via Google Cloud Run with simple Dockerfile and cloudbuild.yaml configuration

🧩 Tech Stack
---
Component |	Description
----------|-------------
Frontend	| Streamlit
Backend	  | Python (single container)
LLM	      | Microsoft/phi-2 for summarisation & repurposing
ASR Model	| Whisper
Cloud Infrastructure	| Google Cloud Run + Cloud Build
Containerization	| Docker

⚙️ Architecture Overview
---
```
User
 │
 ▼
Streamlit UI ──► Audio Processor (ffmpeg + pydub)
                 │
                 ▼
           Whisper ASR Model
                 │
                 ▼
        LLM (phi-2) Summariser & Repurposer
                 │
                 ▼
          Formatted Outputs (text)
```


🪜 Setup (Developer Notes)
---
To replicate the setup, you’ll need:

A Google Cloud Project with:

Cloud Run

Cloud Build

Artifact Registry enabled


❤️ Acknowledgements
---
Built by Andrew Mensah-Onumah — powered by curiosity, and a few sleepless nights ☕💻.<br>
**Live demo**: [Try the App out here](https://summariser-app-269347135162.europe-west1.run.app/)
