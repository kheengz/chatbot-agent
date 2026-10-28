from typing import List
from openai import OpenAI
from config import config
from .system_prompt import get_system_prompt
from ddtrace.llmobs.decorators import embedding

EMBED_URL = f"{config.EMBED_BASE_URL}:{config.EMBED_PORT}/v1"
LLM_URL = f"{config.LLM_BASE_URL}:{config.LLM_PORT}/v1"

openai_client = OpenAI(
    base_url=LLM_URL,
    api_key=config.OPENAI_API_KEY,
)

openai_client_embedding = OpenAI(
    base_url=EMBED_URL,
    api_key=config.OPENAI_API_KEY,
)


# Not RAG-enabled, just a straightforward LLM call
def ask_the_llm(prompt: str) -> str:
    """
    Calls the LLM with the user and system prompts.
    """
    system_prompt = get_system_prompt()
    response = openai_client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content


@embedding(model_name=config.EMBED_MODEL)
def create_embeddings(text: str, prefix: str = "search_query") -> List[float]:
    """
    Creates embeddings for the given text using the configured embedding model.
    """
    response = openai_client_embedding.embeddings.create(
        model=config.EMBED_MODEL, input=[f"{prefix}: {text}"]
    )
    return response.data[0].embedding
