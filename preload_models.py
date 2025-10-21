import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer, AutoModelForCausalLM

hf_token = os.environ.get("HF_TOKEN")

print("Downloading models to cache...")
# Whisper ASR
whisper_id = "distil-whisper/distil-medium.en"
AutoModelForSpeechSeq2Seq.from_pretrained(whisper_id)
AutoProcessor.from_pretrained(whisper_id)

# Cohere LLM
llm_id = "CohereLabs/c4ai-command-r7b-12-2024"
AutoTokenizer.from_pretrained(llm_id, use_auth_token=hf_token)
AutoModelForCausalLM.from_pretrained(llm_id, use_auth_token=hf_token)
print("✅ Model pre-download complete.")
