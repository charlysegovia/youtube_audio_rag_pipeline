# src/server.py

from flask import Flask, request, render_template
from openai import OpenAI

from config import (
    INDEX_LIST,
    OPENAI_API_KEY,
    PINECONE_CLIENT,
    FLASK_HOST,
    FLASK_PORT,
    FLASK_DEBUG,
    EMBED_MODEL
)

from downloader import download_and_transcode
from audio_extractor import extract_audio
from transcriber import transcribe_audio
from chunker import chunk_text
from embedder import embed_chunks
from pinecone_uploader import upsert_embeddings

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)


@app.route("/process", methods=["GET", "POST"])
def process_video():
    status = []
    selected_index = INDEX_LIST[0]

    if request.method == "POST":
        url = request.form.get("video_url")
        selected_index = request.form.get("pinecone_index", selected_index)

        status.append("Downloading video...")
        video_path = download_and_transcode(url)
        status.append(f"Downloaded video: {video_path}")

        status.append("Extracting audio...")
        audio_path = extract_audio(video_path, output_format="mp3")
        status.append(f"Extracted audio: {audio_path}")

        status.append("Transcribing audio...")
        transcript = transcribe_audio(audio_path, language="en")
        status.append("Transcription complete")

        status.append("Chunking transcript...")
        chunks = chunk_text(transcript, chunk_size=800, overlap=200)
        status.append(f"{len(chunks)} chunks created")

        status.append("Generating embeddings...")
        embeddings = embed_chunks(chunks)
        status.append(f"{len(embeddings)} embeddings generated")

        status.append(f"Upserting to Pinecone index '{selected_index}'...")
        upsert_embeddings(embeddings, selected_index)
        status.append("Upsert complete")

    return render_template(
        "process.html",
        index_list=INDEX_LIST,
        selected_index=selected_index,
        status=status,
    )


@app.route("/ask", methods=["GET", "POST"])
def ask_question():
    answer = None
    selected_index = INDEX_LIST[0]

    if request.method == "POST":
        question = request.form.get("query")
        selected_index = request.form.get("pinecone_index", selected_index)

        # 1. Embed the question
        resp = openai_client.embeddings.create(
            input=question,
            model=EMBED_MODEL
        )
        question_embedding = resp.data[0].embedding

        # 2. Query Pinecone
        index = PINECONE_CLIENT.Index(selected_index)
        result = index.query(
            vector=question_embedding,
            top_k=5,
            include_metadata=True
        )
        contexts = [m.metadata.get("text", "") for m in result.matches]
        combined_context = "\n\n".join(contexts)

        # 3. Call ChatGPT with context
        prompt = (
            "You are a helpful assistant. Use the following context to answer the question.\n\n"
            f"Context:\n{combined_context}\n\n"
            f"Question:\n{question}"
        )
        chat_resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.1,
        )
        answer = chat_resp.choices[0].message.content

    return render_template(
        "ask.html",
        index_list=INDEX_LIST,
        selected_index=selected_index,
        answer=answer,
    )


if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
