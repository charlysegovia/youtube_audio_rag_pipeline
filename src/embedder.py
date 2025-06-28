import os
from openai import OpenAI
from config import EMBED_MODEL

# instantiate the OpenAI client using the key from the environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_chunks(chunks: list[str],
                 model: str = EMBED_MODEL) -> list[dict]:
    """
    Calls OpenAI’s embeddings API on each chunk of text.
    Returns a list of dicts with “id”, “embedding” and “text” keys.
    """
    results = []
    for i, chunk in enumerate(chunks):
        # send one chunk at a time (you can batch if you prefer)
        resp = client.embeddings.create(
            input=chunk,
            model=model
        )
        embedding = resp.data[0].embedding
        results.append({
            "id": str(i),
            "embedding": embedding,
            "text": chunk
        })
    return results
