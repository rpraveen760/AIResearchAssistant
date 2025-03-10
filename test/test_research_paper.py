import os
from dotenv import load_dotenv
from langchain_community.utilities import SerpAPIWrapper

# ✅ Load environment variables
load_dotenv()


def test_serpapi():
    """Test fetching research papers using SerpAPI."""

    # ✅ Ensure the API key is loaded
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    if not serpapi_key:
        raise ValueError("❌ SerpAPI API key not found. Ensure it's in the .env file and loaded correctly.")

    # ✅ Initialize SerpAPI with the loaded key
    search = SerpAPIWrapper(serpapi_api_key=serpapi_key)

    # ✅ Define a research query
    query = "Self-Attention in Transformers site:arxiv.org OR site:researchgate.net OR site:aclweb.org"

    # ✅ Run the search
    results = search.run(query)

    # ✅ Print results
    print(f"🔍 Test API Response: {results}")


# ✅ Run the test
if __name__ == "__main__":
    test_serpapi()
