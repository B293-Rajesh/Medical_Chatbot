# rag_utils.py
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- Embedding Model ----------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ---------------- Build or Load FAISS ----------------
if not os.path.exists("faiss_index"):

    docs = []
    for file in os.listdir("medical_data"):
        with open(f"medical_data/{file}", "r", encoding="utf-8") as f:
            docs.append(f.read())

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.create_documents(docs)

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("faiss_index")

else:
    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

# ---------------- Simple Local Response Generator ----------------
def generate_local_response(context, query):
    return f"""
Based on your symptoms, here is preliminary guidance:

{context[:500]}...

This may relate to one of the above mentioned conditions.

Basic advice:
• Maintain hygiene
• Avoid known triggers
• Seek medical consultation if symptoms persist

This is preliminary guidance. Please consult a doctor.
"""

# ---------------- RAG Function ----------------
def get_answer(user_query):
    docs = db.similarity_search(user_query, k=3)
    context = "\n".join([d.page_content for d in docs])
    response = generate_local_response(context, user_query)
    return response