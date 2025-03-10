import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph_agents import retrieval_agent, answer_generation_agent
from verification_agent import verification_agent
from typing import TypedDict
from summarization_agent import summarization_agent

load_dotenv()

class AgentState(TypedDict):
    query: str
    docs: list
    answer: str

# Initialize workflow
workflow = StateGraph(AgentState)

# Add agents
workflow.add_node("retrieval_agent", retrieval_agent)
workflow.add_node("summarization_agent", summarization_agent)  # ✅ Added summarization
workflow.add_node("answer_generation_agent", answer_generation_agent)
workflow.add_node("verification_agent", verification_agent)

# Set entry point
workflow.set_entry_point("retrieval_agent")

# ✅ Corrected edges
workflow.add_edge("retrieval_agent", "summarization_agent")  # ✅ Pass retrieval → summarization
workflow.add_edge("summarization_agent", "answer_generation_agent")  # ✅ Pass summarized text → answer generation
workflow.add_edge("answer_generation_agent", "verification_agent")  # ✅ Verify final answer
workflow.add_edge("verification_agent", END)  # ✅ Workflow completes here

# Compile workflow
app = workflow.compile()

if __name__ == "__main__":
    user_query = input("\n🔍 Enter your query: ")
    inputs = {"query": user_query}
    result = app.invoke(inputs)  # ✅ clearly corrected to positional input here

    print("\n📌 Question:", user_query)
    print("\n💡 Verified Answer:\n", result["answer"])
