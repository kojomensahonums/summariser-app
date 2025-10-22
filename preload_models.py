import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer, AutoModelForCausalLM

hf_token = os.environ.get("HF_TOKEN")

print("Downloading models to cache...")
# Whisper ASR
whisper_id = "distil-whisper/distil-small.en"
AutoModelForSpeechSeq2Seq.from_pretrained(whisper_id)
AutoProcessor.from_pretrained(whisper_id)

# Llama 2
llm_id = "unsloth/llama-2-7b-chat"
AutoTokenizer.from_pretrained(llm_id)
AutoModelForCausalLM.from_pretrained(llm_id)
print("✅ Model pre-download complete.")

# import os
# from google.cloud import storage
# import torch
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer, AutoModelForCausalLM
# import streamlit as st

# # --- Function to download models from GCS ---
# def download_model_from_gcs(bucket_name, source_blob_name, destination_file_name):
#     print(f"Downloading {source_blob_name} from bucket {bucket_name}...")
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)
#     blob.download_to_filename(destination_file_name)
#     print(f"Downloaded {source_blob_name} to {destination_file_name}")

# # --- Load models with caching and GCS download logic ---
# @st.cache_resource
# def load_models_from_gcs():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
#     # --- Model paths and bucket ---
#     bucket_name = "summariser_app_bucket"  # <-- REPLACE WITH YOUR BUCKET NAME
#     whisper_dir = "/tmp/whisper_model"
#     llm_dir = "/tmp/llm_model"
#     os.makedirs(whisper_dir, exist_ok=True)
#     os.makedirs(llm_dir, exist_ok=True)
    
#     # --- Whisper ASR ---
#     whisper_id = "distil-whisper/distil-medium.en"
#     st.info("Loading Whisper model from cache or GCS...")
#     download_model_from_gcs(bucket_name, f"{whisper_id}/model.bin", os.path.join(whisper_dir, "model.bin"))
#     download_model_from_gcs(bucket_name, f"{whisper_id}/config.json", os.path.join(whisper_dir, "config.json"))
#     # ... Add other Whisper files if needed

#     asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(whisper_dir, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
#     asr_model.to(device)
#     processor = AutoProcessor.from_pretrained(whisper_dir)
#     pipe = pipeline(
#         "automatic-speech-recognition",
#         model=asr_model,
#         tokenizer=processor.tokenizer,
#         feature_extractor=processor.feature_extractor,
#         chunk_length_s=25,
#         batch_size=16,
#         torch_dtype=torch_dtype,
#         device=device,
#     )

#     # --- Mixtral LLM ---
#     llm_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
#     st.info("Loading Mixtral model from cache or GCS...")
#     # This is complex. For large models like Mixtral, you need to download all the files.
#     # It's easier to load the model from Hugging Face's cache directly, which will
#     # handle authentication for you (if the token is correctly set).
#     llm_tokenizer = AutoTokenizer.from_pretrained(llm_id)
#     llm_model = AutoModelForCausalLM.from_pretrained(llm_id, torch_dtype="float16", device_map="auto")

#     return pipe, llm_tokenizer, llm_model

