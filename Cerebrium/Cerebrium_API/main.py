import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
import time
import numpy as np

device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = 0 if device_str == "cuda" else -1
model_id = "facebook/wav2vec2-base-960h"

model = Wav2Vec2ForCTC.from_pretrained(model_id)
model.to(device_str)
processor = Wav2Vec2Processor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device,
)

# THIS FUNCTION SIGNATURE MATCHES YOUR CLIENT CODE
def run(array, sampling_rate, run_id=None):
    """
    array: list of floats [-1, 1]
    sampling_rate: int
    run_id: optional
    """
    try:
        samples = np.array(array, dtype=np.float32)
        sampling_rate = int(sampling_rate)

        start = time.time()
        result = pipe({"array": samples, "sampling_rate": sampling_rate})
        end = time.time()
        latency = end - start

        return {
            "transcription": result["text"],
            "latency_s": latency,
            "status_code": 200
        }
    except Exception as e:
        return {
            "transcription": "",
            "latency_s": -1,
            "status_code": 500,
            "error": str(e)
        }
    
if __name__ == "__main__":
    # Example usage
    import soundfile as sf

    file_path = "clip_1_2sec.wav"
    samples, sampling_rate = sf.read(file_path)
    
    if samples.ndim > 1:
        samples = samples.mean(axis=1)  # Convert to mono if stereo

    samples = samples.astype(np.float32)
    if np.max(np.abs(samples)) > 1.0:
        samples = samples / np.max(np.abs(samples))

    result = run(samples.tolist(), sampling_rate)
    print(result)
