import streamlit as st
import torch
import requests
from io import BytesIO
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoTokenizer, AutoModelForCausalLM
from pydub import AudioSegment, effects
from tempfile import NamedTemporaryFile
import os
from preload_models import load_models_from_gcs  # Import the new function

# ------------------------------
# Page setup
# ------------------------------
st.set_page_config(page_title="Audio Summarizer & Repurposer", layout="wide")
st.markdown("<style> .block-container {max-width: 1200px; font-size: 16px;} </style>", unsafe_allow_html=True)
st.title("🎙️ Audio Summarizer & Repurposer")
st.markdown(
    """
Upload an audio file **or** paste a direct audio link.  
This app will transcribe, summarize, or generate promo content — all in one place.
"""
)

# ------------------------------
# Load Models (cached)
# ------------------------------
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Whisper
    whisper_id = "distil-whisper/distil-small.en"
    st.info("Loading Whisper model... (first time only)")
    asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(whisper_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
    asr_model.to(device)
    processor = AutoProcessor.from_pretrained(whisper_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=asr_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=25,
        batch_size=16,
        torch_dtype=torch_dtype,
        device=device,
    )

    # LLM
    st.info("Loading Microsoft Phi-2 model... (first time only)")
    llm_id = "microsoft/phi-2"
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_id)
    llm_model = AutoModelForCausalLM.from_pretrained(llm_id, torch_dtype="float16", device_map="auto")

    return pipe, llm_tokenizer, llm_model

# Call the function to load models
pipe, llm_tokenizer, llm_model = load_models()


# ------------------------------
# Helper Functions
# ------------------------------
def transcribe_audio(audio_path):
    result = pipe(audio_path)
    return result["text"]


def download_audio_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            st.error("Failed to download audio from the link.")
            return None
        
        audio_data = BytesIO(response.content)
        audio = AudioSegment.from_file(audio_data)
        temp_path = "temp.wav"
        audio.export(temp_path, format="wav")
        return temp_path
    except Exception as e:
        st.error(f"Error downloading audio: {e}")
        return None

def generate_with_llm(prompt, context):
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": prompt}
    ]
    input_ids = llm_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    gen_tokens = llm_model.generate(
        input_ids,
        max_new_tokens=250,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    enhanced = llm_tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
    return enhanced.strip()

def summarise_output(transcript):
    context = "This is a transcript of a podcast or conversation. Summarize it clearly and concisely."
    instructions = """
    - Extract the main ideas and key takeaways.
    - Write in smooth, natural language.
    - Avoid unnecessary repetition or filler.
    """
    return generate_with_llm(f"{instructions}\n\nTranscript:\n{transcript}", context)

def repurpose_output(transcript):
    context = "This transcript will be repurposed for social media and SEO-friendly content."
    instructions = """
    Create:
    1. A short tweet thread (2–4 tweets).
    2. 2–3 catchy captions for social media.
    3. 2–3 SEO-friendly titles or headlines.
    Write engaging, natural language text.
    """
    return generate_with_llm(f"{instructions}\n\nTranscript:\n{transcript}", context)


# ------------------------------
# Input Section
# ------------------------------
input_mode = st.radio("Choose input method", ("Upload audio file", "Use web link"))

audio_file_path = None
audio_filename = None
audio_url = None

if input_mode == "Upload audio file":
    audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a", "ogg"])
    if audio_file:
        tmp = NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1])
        tmp.write(audio_file.read())
        tmp.flush()
        tmp.close()
        audio_file_path = tmp.name
        audio_filename = audio_file.name
        st.success(f"✅ File ready: {audio_file.name}")
else:
    audio_url = st.text_input("Paste a direct audio URL:")
    if audio_url:
        # st.warning("⚠️ Remote URLs are not yet supported in this version.")
        st.info("Downloading audio from the provided link...")
        audio_path = download_audio_from_url(audio_url)
        if audio_path:
            st.success("Audio downloaded successfully!")


# ------------------------------
# Action Buttons
# ------------------------------
st.markdown("### Actions")
col1, col2, col3 = st.columns(3)

if audio_file_path:
    with col1:
        if st.button("🎧 Transcribe"):
            transcript = transcribe_audio(audio_file_path)
            st.text_area("📝 Transcript", transcript, height=350)
            st.download_button("Download Transcript", transcript, file_name="transcript.txt")

    with col2:
        if st.button("🧾 Summarize"):
            transcript = transcribe_audio(audio_file_path)
            summary = summarise_output(transcript)
            st.text_area("📄 Summary", summary, height=350)
            st.download_button("Download Summary", summary, file_name="summary.txt")

    with col3:
        if st.button("💡 Promo Content"):
            transcript = transcribe_audio(audio_file_path)
            promo = repurpose_output(transcript)
            st.text_area("💡 Promo Output", promo, height=350)
            st.download_button("Download Promo", promo, file_name="promo.txt")
else:
    st.info("Please upload an audio file to begin.")

# ------------------------------
# Cleanup temp files
# ------------------------------
if audio_file_path and os.path.exists(audio_file_path):
    os.remove(audio_file_path)







