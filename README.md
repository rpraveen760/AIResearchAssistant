📚 AI Research Explorer
🚀 An interactive LLM-powered Q&A system using Retrieval-Augmented Generation (RAG) to extract insights from research papers. This project leverages LangChain, LangGraph, Pinecone, Gradio, and Wikipedia search integration for an advanced research assistant.

🔍 Features
✅ Multi-Agent Architecture → Implements LangGraph to manage multiple agents for retrieval, verification, summarization, and response generation.
✅ Efficient Semantic Search → Uses Pinecone vector database with hybrid retrieval (dense + keyword search).
✅ Streaming Responses → Built with Gradio for real-time answer streaming, enhancing user experience.
✅ Wikipedia Search Integration → Supplement research-based answers with external knowledge retrieval via LangChain tools.
✅ Guardrails AI for Verification → Ensures factual correctness, preventing hallucinations.
✅ User Feedback & Improvements → Collects feedback to improve system accuracy over time.

🛠️ Tech Stack
💡 LLMs → OpenAI GPT-4 (can be extended to other models)
🔗 LangChain & LangGraph → Orchestrates multi-agent workflows
📖 Pinecone → Stores vector embeddings for fast retrieval
🌐 Wikipedia API → Fetches external knowledge for better context
🎨 Gradio → Interactive frontend for easy user interaction
🛡️ Guardrails AI → Verification to ensure reliable responses
⚙️ Installation
1️⃣ Clone the repository

bash
Copy
Edit
git clone https://github.com/rpraveen760/AIResearchAssistant.git
cd AIResearchAssistant
2️⃣ Create a virtual environment

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate      # Windows
3️⃣ Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
4️⃣ Set up environment variables
Create a .env file and add your API keys:

ini
Copy
Edit
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=your_pinecone_index
🚀 Running the Application
Run the Gradio interface:

bash
Copy
Edit
python gradio_interface.py
It will launch a local Gradio app where you can interact with the AI.

🖥️ Demo
🎥 Check out the full demonstration video: [Demo Link]
(Upload to YouTube or LinkedIn and update this link.)

📌 Example Queries
"Explain multi-head attention in Transformers."
"How does self-attention work?"
"Compare Transformers and RNNs."
