import uuid
from config import PINECONE_CLIENT
from categories import get_categories_from_chunk

def upsert_embeddings(embeddings: list[dict], index_name: str, batch_size: int = 50):
    """
    Upserts embeddings into the specified Pinecone index, adding a 'category' metadata field.
    """
    index = PINECONE_CLIENT.Index(index_name)


    # generate a unique run identifier once per upsert call
    run_id = uuid.uuid4().hex
    
    vectors = []
    for i, item in enumerate(embeddings):
        # build a unique ID per chunk per run
        unique_id = f"{run_id}-{i}"

        cats = get_categories_from_chunk(item["text"])
        primary_category = cats[0] if cats else "other"

        vectors.append({
            "id": unique_id,                  # now unique every run
            "values": item["embedding"],
            "metadata": {
                "text": item["text"],
                "category": primary_category
            }
        })

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)
