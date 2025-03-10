import os
import pinecone
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore



# Load API keys
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Initialize Pinecone using the new API format
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Connect to the Pinecone index
index = pc.Index(PINECONE_INDEX)

# Initialize OpenAI embeddings
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Set up LangChain's Pinecone retriever
vector_store = PineconeVectorStore(index, embedding_model)


def retrieve(query, top_k=5):
    """Retrieve top-k most relevant chunks from Pinecone based on query."""
    query_embedding = embedding_model.embed_query(query)

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    retrieved_texts = [match["metadata"]["text"] for match in results["matches"]]

    print("\n🔹 Top Retrieved Chunks:")
    for i, text in enumerate(retrieved_texts, 1):
        print(f"\nChunk {i}:\n{text[:500]}...")  # Print first 500 characters

    return retrieved_texts


if __name__ == "__main__":
    user_query = input("\n🔍 Enter your query: ")
    retrieve(user_query)
