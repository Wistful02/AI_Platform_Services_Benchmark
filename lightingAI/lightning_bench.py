import requests
import json
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
import time

# --- Config ---
file_path = "clip_1_2sec.wav"
url = "https://8000-dep-01jzw54tn95a60rzgktz8g73pq-d.cloudspaces.litng.ai/asr"  # Endpoint should be /asr not /predict
headers = {
    'Authorization': 'Bearer b403cd00-f1eb-4d16-b4e5-7632129c861d',
    'Content-Type': 'application/json',
}

# --- Load and normalize audio ---
samples, sampling_rate = sf.read(file_path)

# If stereo, convert to mono
if samples.ndim > 1:
    samples = samples.mean(axis=1)

# Normalize to float32 in [-1, 1]
samples = samples.astype(np.float32)
if np.max(np.abs(samples)) > 1.0:
    samples = samples / np.max(np.abs(samples))

audio_payload = {
    "array": samples.tolist(),
    "sampling_rate": int(sampling_rate)
}

# --- Run the cold call ---
print("Cold run...")
start_time = time.time()
cold_result = requests.post(url, headers=headers, data=json.dumps(audio_payload))
try:
    cold_json = cold_result.json()
except Exception:
    cold_json = {}
cold_latency = cold_json.get("latency_s", -1)
cold_transcription = cold_json.get("transcription", "")
cold_time = time.time() - start_time
print(f"COLD run latency: {cold_time:.3f} s, transcription: {cold_transcription}")

# --- 100 warm runs ---
warm_latencies = []
warm_transcriptions = []
for idx in tqdm(range(100), desc="Warm runs"):
    start_time = time.time()
    response = requests.post(url, headers=headers, data=json.dumps(audio_payload))
    try:
        json_result = response.json()
    except Exception:
        json_result = {}
    print(f"[warm_{idx+1}] {json_result}")
    warm_latencies.append(time.time() - start_time)
    warm_transcriptions.append(json_result.get("transcription", ""))

warm_avg = np.mean(warm_latencies)
warm_std = np.std(warm_latencies)
print(f"\nCOLD run latency: {cold_time:.3f} s")
print(f"WARM run avg: {warm_avg:.3f} s, std: {warm_std:.3f} s")
