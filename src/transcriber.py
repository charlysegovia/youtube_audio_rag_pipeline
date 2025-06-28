# src/transcriber.py

import os
from openai import OpenAI

# instantiate the new client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def transcribe_audio(audio_path: str, language: str = "en") -> str:
    """
    Uses OpenAI’s Whisper via the new v1 client interface to transcribe an audio file.
    Returns the transcript as plain text.
    """
    try:
        with open(audio_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                file=f,
                model="whisper-1",
                response_format="text",
                language=language
            )
        return resp  # transcript string

    except Exception as e:
        # wrap any error so your main.py can handle it uniformly
        raise RuntimeError(f"Whisper transcription failed: {e}")
