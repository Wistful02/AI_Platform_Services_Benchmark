import io
import time
import numpy as np
import tempfile
from typing import List, Optional, Tuple

import torch
from fastapi.responses import JSONResponse
from litserve import LitAPI, LitServer
from pydantic import BaseModel, Field, field_validator
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
import requests
import base64
import re

class ASRRequest(BaseModel):
    array: Optional[List[float]] = Field(
        None, description="Audio waveform array (float32, -1.0 to 1.0)"
    )
    sampling_rate: Optional[int] = Field(
        None, description="Sampling rate of audio"
    )
    audio: Optional[str] = Field(
        None, description="(Optional) Base64, URL, or file path to audio file"
    )

    @field_validator("audio")
    def validate_audio(cls, v):
        if v is None:
            return v
        is_url = re.match(r"^https?://", v)
        is_base64 = (re.match(r"^[A-Za-z0-9+/=]+\Z", v) and len(v) > 100)
        # Otherwise, treat as file path
        is_file_path = not is_url and not is_base64
        if is_url or is_base64 or is_file_path:
            return v
        raise ValueError("audio must be a base64 string, URL, or file path")

    def get_audio_array(self) -> Tuple[np.ndarray, int]:
        if self.array is not None and self.sampling_rate is not None:
            return np.array(self.array, dtype=np.float32), int(self.sampling_rate)

        if self.audio is not None:
            # URL
            if re.match(r"^https?://", self.audio):
                resp = requests.get(self.audio)
                resp.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(resp.content)
                    tmp_file.close()
                    path = tmp_file.name
                array, sr = sf.read(path)
            # Base64
            elif re.match(r"^[A-Za-z0-9+/=]+\Z", self.audio) and len(self.audio) > 100:
                padded = self.audio + "=" * (-len(self.audio) % 4)
                decoded = base64.b64decode(padded)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(decoded)
                    tmp_file.close()
                    path = tmp_file.name
                array, sr = sf.read(path)
            # File path
            else:
                array, sr = sf.read(self.audio)
            if array.ndim > 1:
                array = array.mean(axis=1)
            array = array.astype(np.float32)
            if np.max(np.abs(array)) > 1.0:
                array = array / np.max(np.abs(array))
            return array, sr

        raise ValueError("Provide either (array & sampling_rate) or audio.")

class Wav2Vec2API(LitAPI):
    """
    LitServe API for Hugging Face Wav2Vec2 speech recognition.
    Accepts waveform or audio file input.
    """

    def setup(self, device):
        model_id = "facebook/wav2vec2-base-960h"
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id)
        self.model.to(device)
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            device=0 if device == "cuda" else -1,
        )

    def decode_request(self, request: ASRRequest) -> Tuple[np.ndarray, int]:
        array, sr = request.get_audio_array()
        return (array, sr)

    def predict(self, inputs: Tuple[np.ndarray, int]) -> dict:
        array, sampling_rate = inputs
        start = time.time()
        result = self.pipe({"array": array, "sampling_rate": sampling_rate})
        end = time.time()
        return {
            "transcription": result["text"],
            "latency_s": end - start,
            "status_code": 200
        }

    def encode_response(self, output: dict) -> JSONResponse:
        return JSONResponse(content=output)

if __name__ == "__main__":
    api = Wav2Vec2API()
    server = LitServer(api, accelerator="auto", api_path="/asr", timeout=100)
    server.run(port=8000)
