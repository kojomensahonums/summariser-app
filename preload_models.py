import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer, AutoModelForCausalLM

hf_token = os.environ.get("HF_TOKEN")

print("Downloading models to cache...")
# Whisper ASR
whisper_id = "distil-whisper/distil-small.en"
AutoModelForSpeechSeq2Seq.from_pretrained(whisper_id)
AutoProcessor.from_pretrained(whisper_id)

# Phi-2
llm_id = "microsoft/phi-2"
AutoTokenizer.from_pretrained(llm_id)
AutoModelForCausalLM.from_pretrained(llm_id)
print("✅ Model pre-download complete.")
