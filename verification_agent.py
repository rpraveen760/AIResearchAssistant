import guardrails as gd
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langgraph_agents import search_wikipedia  # Make sure this is imported

# Load OpenAI API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ Correct Guardrails Schema
guardrails_schema = """
<rail version="0.1">
<output>
    <bool name="is_grounded"/>
    <string name="corrected_answer"/>
</output>

<prompt>
You are an AI verification assistant. Your job is to ensure that the provided answer is completely supported by the given context. 
If the answer includes any information not found in the context, rewrite it using only the provided context.

### **Context:**
{{context}}

### **Answer:**
{{answer}}

**Instructions:**
- Return `is_grounded = true` if the answer is fully supported by the context.
- If the answer contains extra/unverified information, return a corrected version that strictly follows the context.
</prompt>
</rail>
"""

# ✅ Corrected OpenAI Chat Model
llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)

# ✅ FIXED: `messages` is now explicitly a keyword-only argument
def llm_api_guardrails(*, messages: list, **kwargs):
    """Callable function for Guardrails that extracts user query and invokes OpenAI."""
    prompt_text = messages[-1]["content"]  # Extract last user message (proper formatting)
    response = llm.invoke(prompt_text)  # Call OpenAI API
    return response.content if hasattr(response, "content") else str(response)

# ✅ Correct Guardrails Initialization
guard = gd.Guard.from_rail_string(guardrails_schema)

def verification_agent(state):
    """Checks if the LLM's answer is factually grounded in the retrieved context."""
    docs = state["docs"]
    answer = state["answer"]
    context = "\n\n".join([doc.page_content for doc in docs])

    # ✅ Search Wikipedia for more context
    wiki_summary = search_wikipedia(answer)  # Retrieve Wikipedia summary for the answer
    wiki_summary = wiki_summary if wiki_summary else "No relevant Wikipedia information found."

    # ✅ Format messages properly for Guardrails
    messages = [
        {"role": "system", "content": "You are a strict verification AI."},
        {"role": "user", "content": f"Context:\n{context}\n\nWikipedia Summary:\n{wiki_summary}\n\nAnswer:\n{answer}"}
    ]

    # ✅ Call Guardrails with messages correctly
    validated_output = guard(  # ✅ FIXED: Now returning a single object
        llm_api=llm_api_guardrails,  # ✅ FIXED: Now using function that accepts messages as keyword-only
        messages=messages  # ✅ FIXED: Now passing structured messages
    )

    # ✅ If the answer isn't grounded, replace it with the corrected version
    if not validated_output["is_grounded"]:
        state["answer"] = validated_output["corrected_answer"]

    return state
