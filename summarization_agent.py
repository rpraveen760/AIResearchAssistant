from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)


def summarization_agent(state):
    """Summarizes the retrieved documents before passing them to the answer generation agent."""
    docs = state.get("docs", [])

    if not docs:
        print("\n🚨 No documents retrieved! Skipping summarization.")
        state["summary"] = "No relevant documents found."
        return state

    # Combine retrieved documents
    context = "\n\n".join([doc.page_content for doc in docs])

    # Summarization prompt
    summary_prompt = f"""
    Summarize the following text while keeping the key information intact:

    {context}

    Provide a concise summary:
    """

    # Call OpenAI GPT-4 to generate the summary
    summary_response = llm.invoke(summary_prompt)

    # ✅ Debugging: Print the summary before storing it
    print("\n📝 Generated Summary:\n", summary_response.content)

    # ✅ Store summary properly in state
    state["summary"] = summary_response.content if hasattr(summary_response, "content") else str(summary_response)

    # ✅ Return updated state
    return state

