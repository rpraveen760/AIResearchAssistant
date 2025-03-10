import os
from pinecone import Pinecone
from dotenv import load_dotenv

# Load API keys
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to the index
index = pc.Index(PINECONE_INDEX)

# Fetch stored vector count
try:
    response = index.describe_index_stats()
    vector_count = response["total_vector_count"]

    if vector_count > 0:
        index.delete(delete_all=True)
        print(f"🗑️ Successfully deleted {vector_count} vectors from Pinecone!")
    else:
        print("✅ No vectors found. Pinecone is already empty!")

except Exception as e:
    print(f"❌ Error clearing Pinecone index: {e}")
