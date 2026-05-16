import streamlit as st
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader


# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="📄",
    layout="wide"
)


# =========================================
# LOAD MODELS
# =========================================

@st.cache_resource
def load_models():

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    model_name = "google/flan-t5-large"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return embedding_model, tokenizer, model


embedding_model, tokenizer, model = load_models()


# =========================================
# GLOBALS
# =========================================

index = None
chunks = []


# =========================================
# PDF LOADER
# =========================================

def load_pdf(file):

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:

        extracted = page.extract_text()

        if extracted:
            text += extracted + "\n"

    return text


# =========================================
# SMART CHUNKING
# =========================================

def create_chunks(text, size=120):

    output = []

    start = 0

    while start < len(text):

        end = start + size

        chunk = text[start:end]

        output.append(chunk)

        start += size

    return output


# =========================================
# BUILD VECTOR DATABASE
# =========================================

def build_index(text):

    global index, chunks

    chunks = create_chunks(text)

    embeddings = embedding_model.encode(chunks)

    embeddings = np.array(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)


# =========================================
# RETRIEVAL
# =========================================

def retrieve(query, k=4):

    query_embedding = embedding_model.encode([query])

    query_embedding = np.array(query_embedding)

    distances, indices = index.search(query_embedding, k)

    docs = []

    for idx in indices[0]:

        if idx < len(chunks):

            docs.append(chunks[idx])

    return docs


# =========================================
# GENERATE ANSWER
# =========================================

def generate_answer(question):

    docs = retrieve(question)

    context = "\n\n".join(docs)

    prompt = f"""
You are an intelligent AI assistant.

Answer the question using ONLY the provided context.

Rules:
- Give a short and professional answer.
- Summarize information clearly.
- Do not copy large blocks of text.
- If skills are asked, provide bullet points.
- If answer not found, say "Information not found in document."

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.3
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer, docs


# =========================================
# UI
# =========================================

st.title("📄 AI PDF Assistant")

st.write("Upload any PDF and ask questions using AI.")

uploaded_pdf = st.file_uploader(
    "Upload PDF",
    type="pdf"
)


if uploaded_pdf:

    with st.spinner("Reading PDF..."):

        text = load_pdf(uploaded_pdf)

    if text.strip() == "":

        st.error("No readable text found in PDF.")

    else:

        with st.spinner("Creating AI index..."):

            build_index(text)

        st.success("PDF indexed successfully!")

        question = st.text_input(
            "Ask a question about the PDF"
        )

        if question:

            with st.spinner("Generating answer..."):

                answer, docs = generate_answer(question)

            st.subheader("Answer")

            st.write(answer)

            st.subheader("Retrieved Sources")

            for i, doc in enumerate(docs):

                with st.expander(f"Source Chunk {i+1}"):

                    st.write(doc)