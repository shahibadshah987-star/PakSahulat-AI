import os
import time
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv(override=True)
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "").strip()

# Clean reset so all 4 sources are indexed cleanly without duplicates
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

docs_to_index = [
    {
        "path": "data/raw/peca_2016.pdf",
        "title": "PECA 2016 (Original Act)",
        "category": "Cyber_Fraud",
        "version": "previous",
        "year": "2016",
    },
    {
        "path": "data/raw/peca_2025.pdf",
        "title": "PECA Amendment Act 2025",
        "category": "Cyber_Fraud",
        "version": "recent",
        "year": "2025",
    },
    {
        "path": "data/raw/hec_attestation_manual.pdf",
        "title": "HEC Degree Attestation Manual",
        "category": "Education",
        "version": "recent",
        "year": "2025",
    },
    {
        "path": "data/raw/nadra_frc_policy.txt",
        "title": "NADRA FRC & Identity Policy",
        "category": "Identity",
        "version": "recent",
        "year": "2025",
    },
    {
        "path": "data/raw/nadra_cnic_policy.txt",
        "title": "NADRA Citizen Identity & CNIC Reference Guide",
        "category": "Identity",
        "version": "recent",
        "year": "2025",
    },
]

splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
all_chunks = []

for item in docs_to_index:
    if item["path"].endswith(".pdf"):
        loader = PyPDFLoader(item["path"])
    else:
        loader = TextLoader(item["path"], encoding="utf-8")

    raw_docs = loader.load()
    chunks = splitter.split_documents(raw_docs)
    for chunk in chunks:
        chunk.metadata.update({
            "title": item["title"],
            "category": item["category"],
            "version": item["version"],
            "year": item["year"],
        })
    all_chunks.extend(chunks)
    print(f"Prepared {len(chunks)} chunks from {item['title']} ({item['category']})")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db",
    collection_name="sarkari_docs",
)

# Add in small batches of 25 to stay safely under free-tier limits
batch_size = 25
total = len(all_chunks)
for i in range(0, total, batch_size):
    batch = all_chunks[i : i + batch_size]
    vectorstore.add_documents(batch)
    print(f"Saved batch {i + 1} to {min(i + batch_size, total)} of {total} chunks...")
    if i + batch_size < total:
        time.sleep(20)

print(f"\nSuccess! All {total} multi-domain chunks saved permanently in ./chroma_db")