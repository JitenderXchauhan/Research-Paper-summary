import gradio as gr
from model import extract_text_from_pdf, chunk_text, summarize_chunks, save_summary_to_pdf
import os 

def process_pdf(pdf_file):
    try:
        text = extract_text_from_pdf(pdf_file)
        if not text.strip():
            return "No extractable text found in PDF.", None
        chunks = chunk_text(text)
        summary = summarize_chunks(chunks)
        pdf_output = save_summary_to_pdf(summary)
        return summary, pdf_output
    except Exception as e:
        return f"Error: {e}", None

demo = gr.Interface(
    fn=process_pdf,
    inputs=gr.File(label="Upload PDF", type="filepath"),
    outputs=[
        gr.Textbox(label="Generated Summary", lines=20),
        gr.File(label="Download PDF Summary")
    ],
    title="Research Paper Summarizer",
    description="Upload a research paper PDF. This app extracts, summarizes, and gives you a summarized PDF using Groq's LLaMA model."
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860)) 
    demo.launch(server_name="0.0.0.0", server_port=port)
