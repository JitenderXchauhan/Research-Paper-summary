from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from groq import Groq
from PyPDF2 import PdfReader
import os
import textwrap
from dotenv import load_dotenv
import tempfile

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is missing in .env")

groq_client = Groq(api_key=api_key)

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def chunk_text(text, max_words=900):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def summarize_chunks(chunks, model="llama3-70b-8192"):
    summaries = []
    for chunk in chunks:
        prompt = f"Summarize the following academic content clearly and concisely:\n\n{chunk}"
        response = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        summary = response.choices[0].message.content.strip()
        summaries.append(summary)
    return "\n\n".join(summaries)

def split_line(text, max_chars=100):
    return textwrap.wrap(text, width=max_chars)

def save_summary_to_pdf(summary_text):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    output_path = temp_file.name

    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    margin = 40
    max_lines = 60

    text_object = c.beginText(margin, height - margin)
    text_object.setFont("Helvetica", 11)
    lines_written = 0

    for line in summary_text.split("\n"):
        for wrapped in split_line(line):
            if lines_written >= max_lines:
                c.drawText(text_object)
                c.showPage()
                text_object = c.beginText(margin, height - margin)
                text_object.setFont("Helvetica", 11)
                lines_written = 0
            text_object.textLine(wrapped)
            lines_written += 1

    c.drawText(text_object)
    c.save()
    return output_path
