"""
The `Model` class is an interface between the ML model that you're packaging and the model
server that you're running it on.

The main methods to implement here are:
* `load`: runs exactly once when the model server is spun up or patched and loads the
   model onto the model server. Include any logic for initializing your model, such
   as downloading model weights and loading the model into memory.
* `predict`: runs every time the model server is called. Include any logic for model
  inference and return the model output.

See https://truss.baseten.co/quickstart for more.
"""

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
import time
import numpy as np
import logging

class Model:
    def __init__(self, **kwargs):
        # Uncomment the following to get access
        # to various parts of the Truss config.

        # self._data_dir = kwargs["data_dir"]
        # self._config = kwargs["config"]
        # self._secrets = kwargs["secrets"]
        self._model = None

    def load(self):
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = 0 if device_str == "cuda" else -1
        model_id = "facebook/wav2vec2-base-960h"

        model = Wav2Vec2ForCTC.from_pretrained(model_id)
        model.to(device_str)
        processor = Wav2Vec2Processor.from_pretrained(model_id)

        self.model = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
        )

    def predict(self, model_input):

        logging.basicConfig(level=logging.INFO)
        logging.info("Model input: %s", model_input)

        array = model_input['array']
        sampling_rate = model_input['sampling_rate']

        try:
            samples = np.array(array, dtype=np.float32)
            sampling_rate = int(sampling_rate)

            start = time.time()
            result = self.model({"array": samples, "sampling_rate": sampling_rate})
            end = time.time()
            latency = end - start

            return {
                "transcription": result["text"],
                "latency_s": latency,
                "status_code": 200
            }
        except Exception as e:
            return {
                "transcription": "ERROR: check logs",
                "latency_s": -500,
                "status_code": 500,
                "error": str(e)
            }