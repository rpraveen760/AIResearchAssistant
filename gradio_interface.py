import gradio as gr
from langgraph_orchestration import app
import csv
import os
from generate_answer import get_answer
from langgraph_agents import search_wikipedia


import time
import gradio as gr

def generate_answer(query):
    """Streams retrieved chunks and the final answer progressively."""
    for answer, retrieved_chunks in get_answer(query):  # ✅ Ensure correct unpacking
        yield answer["value"], retrieved_chunks["value"]  # ✅ Extract only the values, remove dict structure
        time.sleep(0.05)  # ✅ Adjust streaming speed




# Function to record user feedback
def record_feedback(query, answer, rating):
    feedback_file = "user_feedback.csv"
    exists = os.path.isfile(feedback_file)
    with open(feedback_file, mode="a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["Query", "Answer", "Rating"])
        writer.writerow([query, answer, rating])
    return "✅ Feedback recorded! Thank you."

with gr.Blocks(title="📚 AI Research Explorer") as demo:
    gr.Markdown("## 📚 AI Research Explorer\n### Interactive Q&A on 'Attention Is All You Need'")

    query = gr.Textbox(label="🔍 Enter your query:", lines=3)
    submit_btn = gr.Button("Get Answer")
    answer = gr.Textbox(label="✅ Verified Answer", lines=10)

    # ✅ Wikipedia Search Button
    wiki_search_btn = gr.Button("Search Wikipedia")
    wiki_output = gr.Markdown(label="🌍 Wikipedia Summary", visible=False)  # Fixed this to work correctly


    # ✅ Add Wikipedia Search Function
    def display_wikipedia(query):
        wiki_result = search_wikipedia(query)
        return gr.update(visible=True, value=wiki_result)  # Return updated value with markdown formatting

    wiki_search_btn.click(
        fn=display_wikipedia,
        inputs=query,
        outputs=wiki_output
    )

    # ✅ Show Retrieved Chunks Button
    show_chunks_btn = gr.Button("Show Retrieved Chunks")
    retrieved_chunks_output = gr.Textbox(label="📑 Retrieved Chunks", interactive=False, lines=10, visible=False)


    gr.HTML("<div style='text-align: left; font-size: 16px; font-weight: bold;'>⭐ Rate the answer</div>")
    rating = gr.Radio(choices=[1, 2, 3, 4, 5], label="", value=3, interactive=True, type="index")

    feedback_button = gr.Button("Submit Feedback")
    feedback_status = gr.Markdown("")

    # ✅ Button Click Actions
    submit_btn.click(
        fn=generate_answer,
        inputs=query,
        outputs=[answer, retrieved_chunks_output] # ✅ Allow progressive updates
    )


    def display_chunks(chunks):
        return gr.update(visible=True, value=chunks)

    show_chunks_btn.click(
        fn=display_chunks,
        inputs=retrieved_chunks_output,
        outputs=retrieved_chunks_output
    )

    feedback_button.click(
        fn=record_feedback,
        inputs=[query, answer, rating],
        outputs=feedback_status
    )

demo.launch(share=True)
