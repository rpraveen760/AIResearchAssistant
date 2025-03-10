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
    vector_count = response.get("total_vector_count", 0)
    print(f"📊 Pinecone Index: {PINECONE_INDEX}")
    print(f"✅ Total Vectors Stored: {vector_count}")
except Exception as e:
    print(f"❌ Error fetching index stats: {e}")
