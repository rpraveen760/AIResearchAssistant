import os
from pinecone import Pinecone, ServerlessSpec
import openai
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Load API keys from .env
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure the index exists (create if not)
if PINECONE_INDEX not in pc.list_indexes().names():
    print(f"🚀 Creating Pinecone index: {PINECONE_INDEX}")
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=1536,  # OpenAI embeddings dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to the Pinecone index
index = pc.Index(PINECONE_INDEX)

# Load text chunks
try:
    with open("text_chunks.txt", "r", encoding="utf-8") as f:
        chunks = f.read().split("--- Chunk")  # Ensure delimiter consistency
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]  # Remove empty chunks
    print(f"✅ Loaded {len(chunks)} text chunks.")
except FileNotFoundError:
    print("❌ Error: 'text_chunks.txt' not found. Make sure the file exists.")
    exit(1)

# Initialize OpenAI embeddings
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Batch upsert for efficiency
batch_size = 100
vectors_to_upsert = []

for i, chunk in enumerate(chunks):
    vector = embedding_model.embed_documents([chunk])[0]  # Generate embeddings
    vector_id = f"chunk-{i}"
    vectors_to_upsert.append((vector_id, vector, {"text": chunk}))

    if len(vectors_to_upsert) >= batch_size:
        index.upsert(vectors_to_upsert)
        print(f"🚀 Upserted {len(vectors_to_upsert)} vectors.")
        vectors_to_upsert = []  # Reset batch

# Final batch upsert
if vectors_to_upsert:
    index.upsert(vectors_to_upsert)
    print(f"🚀 Final upserted {len(vectors_to_upsert)} vectors.")

print(f"✅ Successfully stored {len(chunks)} chunks in Pinecone!")
