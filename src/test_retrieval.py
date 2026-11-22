import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv(override=True)
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "").strip()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db",
    collection_name="sarkari_docs",
)

query = "What authority or agency investigates cyber crimes and social media complaints?"

print("=== RECENT DOCS ONLY (2025) ===")
recent_docs = vectorstore.similarity_search(query, k=2, filter={"version": "recent"})
for d in recent_docs:
    print(f"[{d.metadata['title']} | Page {d.metadata.get('page', 0) + 1}]")
    print(d.page_content[:250].replace("\n", " "), "...\n")

print("=== PREVIOUS DOCS ONLY (2016) ===")
prev_docs = vectorstore.similarity_search(query, k=2, filter={"version": "previous"})
for d in prev_docs:
    print(f"[{d.metadata['title']} | Page {d.metadata.get('page', 0) + 1}]")
    print(d.page_content[:250].replace("\n", " "), "...\n")