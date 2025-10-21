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
