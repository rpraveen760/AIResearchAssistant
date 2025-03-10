import os
import pinecone
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Load API keys
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Initialize Pinecone (corrected)
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Initialize OpenAI embeddings and LLM
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)

# Connect to existing Pinecone index
vector_store = PineconeVectorStore.from_existing_index(
    index_name=PINECONE_INDEX,
    embedding=embedding_model,
    text_key="text"
)

# Retriever setup (top 5 relevant chunks)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Define prompt explicitly
prompt_template = """
You are an expert assistant specialized in explaining concepts from the paper "Attention Is All You Need".
Answer the user's query clearly and concisely using ONLY the provided context. Do not include any external information.

📌 **Question:** {question}

📖 **Context:** {context}

📝 **Answer:**
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

# Create RetrievalQA chain explicitly defining input keys
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

import time

import time
import gradio as gr


def get_answer(query):
    """Retrieve and stream answer chunks progressively."""
    result = qa_chain.invoke({"query": query})
    retrieved_chunks = [doc.page_content for doc in result.get("source_documents", [])]

    retrieved_chunks_str = ""

    # ✅ Stream retrieved chunks progressively
    if retrieved_chunks:
        for chunk in retrieved_chunks:
            retrieved_chunks_str += chunk + "\n\n---\n\n"
            yield gr.update(value="Processing..."), gr.update(value=retrieved_chunks_str)
            time.sleep(0.5)  # Slow down streaming slightly

    # ✅ If no chunks are found, yield a default message
    if not retrieved_chunks:
        yield gr.update(value="No chunks retrieved."), gr.update(value="")

    # ✅ Finally, stream the answer
    answer_text = result["result"]
    streamed_answer = ""
    for word in answer_text.split():
        streamed_answer += word + " "
        yield gr.update(value=streamed_answer), gr.update(value=retrieved_chunks_str)
        time.sleep(0.05)  # Adjust streaming speed


if __name__ == "__main__":
    user_query = input("\n🔍 Enter your query: ")
    answer = get_answer(user_query)
    print("\n💡 Generated Answer:\n", answer)
