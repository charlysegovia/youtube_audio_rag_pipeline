import tiktoken

def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 200,
    model_name: str = "cl100k_base"
) -> list[str]:
    """
    Splits `text` into chunks of up to `chunk_size` tokens,
    with `overlap` tokens shared between consecutive chunks.

    Returns a list of text chunks.
    """
    # initialize the tokenizer
    enc = tiktoken.get_encoding(model_name)
    token_ids = enc.encode(text)

    chunks: list[str] = []
    start = 0
    total_tokens = len(token_ids)

    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)
        chunk_tokens = token_ids[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        # advance by chunk_size - overlap
        start += chunk_size - overlap

    return chunks
