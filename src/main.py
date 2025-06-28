#!/usr/bin/env python
# src/main.py

import sys
import shutil
from config import INDEX_LIST
from downloader import download_and_transcode
from audio_extractor import extract_audio
from transcriber import transcribe_audio
from chunker import chunk_text
from embedder import embed_chunks
from pinecone_uploader import upsert_embeddings
from pathlib import Path
from logger import logger

def run_pipeline(video_url: str, pinecone_index: str):

    BASE = Path(__file__).parent.parent   # esto es tu carpeta raíz del proyecto

    for sub in ("videos", "audio"):
        d = BASE / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    print(f"Running pipeline for {video_url} into index '{pinecone_index}'\n")
    logger.info(f"Running pipeline for {video_url} into index '{pinecone_index}'\n")

    mp4 = download_and_transcode(video_url)
    print(f"Video saved to: {mp4}")
    logger.info(f"Video saved to: {mp4}")

    audio = extract_audio(mp4, output_format="mp3")
    print(f"Audio extracted to: {audio}")
    logger.info(f"Audio extracted to: {audio}")

    transcript = transcribe_audio(audio, language="en")
    print(f"Transcription complete (length: {len(transcript)} chars)")
    logger.info(f"Transcription complete (length: {len(transcript)} chars)")

    chunks = chunk_text(transcript, chunk_size=800, overlap=200)
    print(f"Created {len(chunks)} text chunks")
    logger.info(f"Created {len(chunks)} text chunks")

    embeddings = embed_chunks(chunks)
    print(f"Generated {len(embeddings)} embeddings")
    logger.info(f"Generated {len(embeddings)} embeddings")

    upsert_embeddings(embeddings, pinecone_index)
    print(f"Upserted all embeddings into Pinecone index '{pinecone_index}'\n")
    logger.info(f"Upserted all embeddings into Pinecone index '{pinecone_index}'\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/main.py <youtube_url> <pinecone_index>")
        print("Available indices:", INDEX_LIST)
        logger.info("Usage: python src/main.py <youtube_url> <pinecone_index>")
        logger.info("Available indices:", INDEX_LIST)
        sys.exit(1)

    video_url = sys.argv[1]
    pinecone_index = sys.argv[2]

    if pinecone_index not in INDEX_LIST:
        print(f"Error: '{pinecone_index}' is not in INDEX_LIST: {INDEX_LIST}")
        logger.info(f"Error: '{pinecone_index}' is not in INDEX_LIST: {INDEX_LIST}")
        sys.exit(1)

    run_pipeline(video_url, pinecone_index)
