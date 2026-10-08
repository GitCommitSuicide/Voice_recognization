from huggingface_hub import hf_hub_download
from datasets import load_dataset
from transformers import pipeline
import evaluate
import json
import os
import pandas as pd
import soundfile as sf

REPO = "HAFSATA/english-asr-monolingual"
OUTPUT_DIR = "speech_samples"
JSON_FILE = "metadata.json"
EXCEL_FILE = "evaluation_results.xlsx"
MODEL_NAME = "distil-whisper/distil-large-v3"
NUM_SAMPLES = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading {MODEL_NAME}...")

asr = pipeline("automatic-speech-recognition", model=MODEL_NAME)

print("Loading evaluation metrics (WER, CER)...")
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

print("Downloading dataset file...")
parquet_file = hf_hub_download(
    repo_id=REPO,
    filename="data/validation-00000-of-00001.parquet",
    repo_type="dataset"
)

from datasets import load_dataset, Audio

# Read the dataset
dataset = load_dataset("parquet", data_files=parquet_file, split="train")
dataset = dataset.cast_column("audio", Audio(decode=False))

import io

metadata = {}
results = []

print(f"Processing {NUM_SAMPLES} samples...")
for i in range(min(NUM_SAMPLES, len(dataset))):
    sample = dataset[i]
    audio_data = sample["audio"]
    original_text = sample["transcript"]
    
    
    if "bytes" in audio_data and audio_data["bytes"]:
        audio_array, sampling_rate = sf.read(io.BytesIO(audio_data["bytes"]))
    elif "path" in audio_data and audio_data["path"]:
        audio_array, sampling_rate = sf.read(audio_data["path"])
    else:
        raise ValueError("Audio data is missing bytes and path.")
        
    filename = f"sample_{i+1}.wav"
    audio_path = os.path.join(OUTPUT_DIR, filename)
    
    # Save audio if it doesn't exist
    if not os.path.exists(audio_path):
        sf.write(audio_path, audio_array, sampling_rate)
        
    # Transcribe using Whisper
    print(f"\nTranscribing {filename}...")
    # Pass dict containing array and sampling rate to the pipeline
    prediction = asr({"array": audio_array, "sampling_rate": sampling_rate})
    predicted_text = prediction["text"].strip()
    
    print(f"Original : {original_text}")
    print(f"Predicted: {predicted_text}")
    
    # Save metadata
    metadata[filename] = original_text
    
    # Compute metrics (requires a list of strings)
    # convert text to lowercase to get fair evaluation if the dataset or model differ in casing
    orig_lower = original_text.lower()
    pred_lower = predicted_text.lower()
    
    wer = wer_metric.compute(predictions=[pred_lower], references=[orig_lower])
    cer = cer_metric.compute(predictions=[pred_lower], references=[orig_lower])
    
    # Store result
    results.append({
        "model_name": MODEL_NAME,
        "filename": filename,
        "original_text": original_text,
        "predicted_text": predicted_text,
        "wer": wer,
        "cer": cer
    })

# Save metadata to JSON
with open(JSON_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)
print(f"\nSaved original text metadata to {JSON_FILE}")

# Save results to Excel
df_new = pd.DataFrame(results)

if os.path.exists(EXCEL_FILE):
    # Append to existing Excel file so you can accumulate multiple models
    df_existing = pd.read_excel(EXCEL_FILE)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
else:
    df_combined = df_new

df_combined.to_excel(EXCEL_FILE, index=False)
print(f"Saved evaluation results to {EXCEL_FILE}")
print("DONE!")
