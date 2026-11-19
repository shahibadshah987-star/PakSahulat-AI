import os
from dotenv import load_dotenv
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY", "").strip()

# 1. Direct check with Google's official SDK
client = genai.Client(api_key=api_key)
res = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Say 'Direct SDK working!' in 3 words.",
)
print("SDK Test:", res.text.strip())

# 2. LangChain check
os.environ["GOOGLE_API_KEY"] = api_key
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")
lc_res = llm.invoke("Say 'Sarkari Kaam Asaan is ready!' in English and Urdu.")
print("LangChain Test:\n", lc_res.content)