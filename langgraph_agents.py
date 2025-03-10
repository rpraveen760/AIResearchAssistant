import os
import pinecone
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
#from langchain_core.prompts import PromptTemplate
#from summarization_agent import summarization_agent
from langchain.tools import Tool
import wikipedia


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)

vector_store = PineconeVectorStore.from_existing_index(
    index_name=PINECONE_INDEX, embedding=embedding_model, text_key="text"
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})


def retrieval_agent(state):
    """Fetch relevant chunks from Pinecone."""
    query = state["query"]

    # ✅ Use the new retrieval method
    retrieved_docs = retriever.invoke(query)

    # ✅ Extract retrieved text
    retrieved_texts = [doc.page_content for doc in retrieved_docs]

    # ✅ Debugging: Print retrieved documents
    print("\n🔍 Retrieved Chunks (for Debugging):")
    for i, text in enumerate(retrieved_texts):
        print(f"Chunk {i + 1}:\n", text, "\n")

    # ✅ Ensure retrieved chunks are stored in the state correctly
    state["retrieved_chunks"] = retrieved_texts if retrieved_texts else ["No chunks retrieved."]
    state["docs"] = retrieved_docs  # Keep the original docs

    return state



def answer_generation_agent(state):
    """Generates an answer based on the summarized documents."""
    summary = state.get("summary")

    # ✅ Debugging: Ensure summary is passed correctly
    if not summary or summary == "No summary available.":
        print("\n🚨 No valid summary found! Using retrieved docs instead.")
        docs = state.get("docs", [])
        summary = "\n\n".join([doc.page_content for doc in docs]) if docs else "No data available."

    # ✅ Debugging: Print the summary before generating the answer
    print("\n📌 Answer Agent Received Summary:\n", summary)

    # Answer Generation Prompt
    answer_prompt = f"""
    Based on the following summarized information, answer the user's query:

    {summary}

    Provide a clear and concise response:
    """

    # Call LLM to generate answer
    response = llm.invoke(answer_prompt)

    # Store the generated answer in state
    state["answer"] = response.content if hasattr(response, "content") else str(response)

    return state





def search_wikipedia(query):
    """Search Wikipedia and return a summary of the top result."""
    try:
        # Get the most relevant Wikipedia page
        page = wikipedia.page(query, auto_suggest=True)
        summary = wikipedia.summary(query, sentences=3)
        return f"**{page.title}**\n\n{summary}\n\n🔗 [Read more]({page.url})"
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple results found: {', '.join(e.options[:5])}..."
    except wikipedia.exceptions.PageError:
        return "No Wikipedia page found for this query."
    except Exception as e:
        return f"Error: {str(e)}"
