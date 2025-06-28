# src/categories.py

import json
from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

# predefined data engineering categories
CATEGORIES = [
    "data ingestion",
    "data transformation",
    "data modeling",
    "data warehousing",
    "etl",
    "elt",
    "streaming",
    "batch processing",
    "spark",
    "airflow",
    "kafka",
    "delta lake",
    "iceberg",
    "dbt",
    "data quality",
    "data governance",
    "data lakes",
    "data catalog",
    "metadata management",
    "monitoring",
]

def get_categories_from_chunk(text: str) -> list[str]:
    """
    Categorize `text` into our predefined CATEGORIES.
    Always returns a list; on any error, returns ["other"].
    """
    combined = ", ".join(CATEGORIES)
    system_msg = (
        f"You are a data engineering expert categorizing chunks of text. "
        f"The only valid categories are: {combined}. "
        "Return a JSON array of the relevant categories, "
        "with the rarest category first, and do not include any markdown."
    )
    user_msg = f"Categorize this data engineering text chunk: {text}"

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg}
            ],
            temperature=0.1,
        )
        content = resp.choices[0].message.content.strip()
        cats = json.loads(content)
        if isinstance(cats, list) and all(isinstance(c, str) for c in cats) and cats:
            return cats
    except Exception:
        pass

    return ["other"]
