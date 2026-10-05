import os
import numpy as np
import faiss
import pdfplumber

from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------------------------
# 1. Load Environment Variables
# --------------------------------------------------

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("URL")
)

# --------------------------------------------------
# 2. PDF Path
# --------------------------------------------------

pdf_path = r"C:\Users\divyarya\OneDrive - Capgemini\Desktop\rag_mini_project\Documents\Holiday_Calendar_2026.pdf"

# --------------------------------------------------
# 3. Extract Text + Tables
# --------------------------------------------------

documents = []

# ---------- TEXT ----------

reader = PdfReader(pdf_path)

for page_num, page in enumerate(reader.pages):

    text = page.extract_text()

    if text:

        documents.append({
            "text": text,
            "metadata": {
                "source": "Holiday_Calendar_2026.pdf",
                "page": page_num + 1,
                "type": "text"
            }
        })

# ---------- TABLES ----------

with pdfplumber.open(pdf_path) as pdf:

    for page_num, page in enumerate(pdf.pages):

        tables = page.extract_tables()

        for table_num, table in enumerate(tables):

            rows = []

            for row in table:

                rows.append(
                    " | ".join(
                        str(cell) if cell else ""
                        for cell in row
                    )
                )

            table_text = "\n".join(rows)

            documents.append({
                "text": table_text,
                "metadata": {
                    "source": "Holiday_Calendar_2026.pdf",
                    "page": page_num + 1,
                    "table": table_num + 1,
                    "type": "table"
                }
            })

print(f"Documents extracted: {len(documents)}")

# --------------------------------------------------
# 4. Chunking
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = []

for doc in documents:

    split_texts = splitter.split_text(
        doc["text"]
    )

    for split_text in split_texts:

        chunks.append({
            "text": split_text,
            "metadata": doc["metadata"]
        })
print(doc["metadata"])
print(f"Chunks created: {len(chunks)}")

# --------------------------------------------------
# 5. Get Text Only
# --------------------------------------------------

texts = [
    chunk["text"]
    for chunk in chunks
]

# --------------------------------------------------
# 6. Generate Embeddings
# --------------------------------------------------

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)

embeddings = np.array(
    [
        item.embedding
        for item in response.data
    ],
    dtype=np.float32
)

print("Embeddings created")

# --------------------------------------------------
# 7. Create FAISS Index
# --------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("Vectors stored:", index.ntotal)

# --------------------------------------------------
# 8. User Question
# --------------------------------------------------

user_prompt = input("\nAsk Question: ")

# --------------------------------------------------
# 9. Create Query Embedding
# --------------------------------------------------

query_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=[user_prompt]
)

query_vector = np.array(
    [query_response.data[0].embedding],
    dtype=np.float32
)

# --------------------------------------------------
# 10. Search FAISS
# --------------------------------------------------

k = 5

distances, indices = index.search(
    query_vector,
    k
)

# --------------------------------------------------
# 11. Retrieve Chunks
# --------------------------------------------------

retrieved_chunks = [
    chunks[i]
    for i in indices[0]
]

# --------------------------------------------------
# 12. Build Context
# --------------------------------------------------

context_parts = []

for chunk in retrieved_chunks:

    metadata = chunk["metadata"]

    context_parts.append(
        f"""
Source: {metadata.get('source')}
Page: {metadata.get('page')}
Type: {metadata.get('type')}

Content:
{chunk['text']}
"""
    )

context = "\n\n".join(context_parts)

# --------------------------------------------------
# 13. Create RAG Prompt
# --------------------------------------------------

prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the context provided.

CONTEXT:
{context}

QUESTION:
{user_prompt}

RULES:
1. Use only the provided context.
2. Mention page numbers when possible.
3. Mention source documents.
4. Do not hallucinate.
5. If answer is not present in the context, say:
   'I could not find this information in the document.'
"""

# --------------------------------------------------
# 14. Generate Final Answer
# --------------------------------------------------

response = client.chat.completions.create(
    model="openai.gpt-5",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

answer = response.choices[0].message.content

# --------------------------------------------------
# 15. Print Answer
# --------------------------------------------------

print("\n" + "=" * 60)
print("ANSWER")
print("=" * 60)

print(answer)

# --------------------------------------------------
# 16. Print Sources
# --------------------------------------------------

print("\nSources Used:\n")

for chunk in retrieved_chunks:

    metadata = chunk["metadata"]

    print(
        f"Page {metadata.get('page')} | "
        f"Type: {metadata.get('type')}"
    )
