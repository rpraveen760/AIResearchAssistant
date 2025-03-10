# Quick test in the same directory (optional):
from langgraph_agents import retrieval_agent, answer_generation_agent

state = {"query": "What is self-attention?"}
state = retrieval_agent(state={"query": "What is self-attention?"})
state["query"] = "What is self-attention?"  # ensure query key is set clearly
state = answer_generation_agent(state)
print(state["answer"])
