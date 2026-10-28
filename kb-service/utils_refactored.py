import os
import json
import uuid
import chromadb
from typing import List, Tuple
from nltk.tokenize import sent_tokenize
from openai import OpenAI

CHROMA_HOST = os.getenv("CHROMADB_HOST", "vector-store")
CHROMA_PORT = os.getenv("CHROMADB_PORT", "8000")
CHROMA_DATA_DIR: str = os.getenv("CHROMA_DATA_DIR", "/chroma/data")
KB_DATA_DIR = os.getenv("KB_DATA_DIR", "./kb_data")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "http://text-embedder")
EMBED_PORT = os.getenv("EMBED_PORT", "11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:v1.5")
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "fixed")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

chromadb_client = chromadb.PersistentClient(path=CHROMA_DATA_DIR)
openai_client_embedding = OpenAI(
    base_url=f"{EMBED_BASE_URL}:{EMBED_PORT}/v1",
    api_key=OPENAI_API_KEY,
)


# Document Processing Functions
def get_topic_from_filename(filename: str) -> str:
    """
    Extracts a topic name from a filename.
    For example: 'ancient-egypt.json' -> 'ancient-egypt'
    """
    base_name = os.path.splitext(os.path.basename(filename))[0]
    return base_name.lower().replace(" ", "_").replace("-", "_")


def fixed_size_chunk_data(text, chunk_size=500):
    """
    Fixed-size chunking: Splits text into chunks of specified size,
    ignoring document structure.
    """
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def sentence_chunk_text(text, max_chunk_size=500, overlap=100):
    """
    Structure-aware chunking that respects sentence boundaries
    while maintaining overlap.
    """
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], []

    for sentence in sentences:
        if sum(len(s) for s in current_chunk) + len(sentence) > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:]

        current_chunk.append(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def get_json_content(file) -> List[str]:
    """Extract content from JSON file as a list of strings."""
    json_data = json.load(file)
    if not isinstance(json_data, list):
        return []
    return [json.dumps(item) for item in json_data if isinstance(item, dict)]


def get_text_content(file) -> List[str]:
    """Extract content from text/markdown file as a single string in a list."""
    return [file.read()]


# Map file extensions to their content extractors
CONTENT_EXTRACTORS = {
    ".json": get_json_content,
    ".md": get_text_content,
    ".txt": get_text_content,
}


def process_file(file_path: str) -> List[Tuple[str, str]]:
    """Process a file and chunk its contents."""
    data = []
    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        with open(file_path, "r") as file:
            topic = get_topic_from_filename(file_path)
            content_list = CONTENT_EXTRACTORS[file_ext](file)

            for content in content_list:
                chunks = (
                    fixed_size_chunk_data(content)
                    if CHUNKING_STRATEGY == "fixed"
                    else sentence_chunk_text(content)
                )
                data.extend([(topic, chunk) for chunk in chunks])

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    return data


def load_data_for_kb() -> List[Tuple[str, str]]:
    """
    Loads and chunks data from all supported files in the KB_DATA_DIR.
    Returns list of (topic, text) tuples to be added to the knowledge base.
    """
    data = []
    supported_extensions = (".json", ".md", ".txt")

    for filename in os.listdir(KB_DATA_DIR):
        file_path = os.path.join(KB_DATA_DIR, filename)
        if os.path.isfile(file_path) and filename.endswith(supported_extensions):
            data.extend(process_file(file_path))

    return data


# Embedding Functions
def create_embeddings(text: str, prefix: str = "search_query") -> List[float]:
    """
    Creates embeddings for the given text using the configured embedding model.
    """
    response = openai_client_embedding.embeddings.create(
        model=EMBED_MODEL, input=[f"{prefix}: {text}"]
    )
    return response.data[0].embedding


# Vector Store Functions
def add_document(document: str, embedding: List[float], topic: str) -> None:
    """
    Adds a single document with its embedding to the vector store.
    Uses a collection that matches the provided topic.
    """
    collection = get_collection(topic)
    doc_id = str(uuid.uuid4())
    collection.add(ids=[doc_id], documents=[document], embeddings=[embedding])
    print(f"Added document {doc_id[:8]} to collection '{topic}'")


def get_collection(collection_name: str):
    """
    Returns the collection with the given name.
    If it doesn't exist, creates it.
    """
    collection = chromadb_client.get_or_create_collection(collection_name)
    doc_count = collection.count()
    print(f"Using collection '{collection_name}'")
    print(f"Current document count: {doc_count}")
    return collection


def clear_collections() -> None:
    """
    Deletes all existing collections from the ChromaDB instance.
    """
    for collection_name in chromadb_client.list_collections():
        chromadb_client.delete_collection(collection_name)
