import os
import requests
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")


url = "https://dummyjson.com/posts"

response = requests.get(url)

if response.status_code != 200:
    raise Exception("Failed to fetch data")

posts = response.json()["posts"]

print(f"\nPosts Loaded : {len(posts)}")


documents = []

for post in posts:

    text = f"""
Post ID : {post['id']}

Title :
{post['title']}

Content :
{post['body']}

Tags :
{', '.join(post['tags'])}

Likes :
{post['reactions']['likes']}

Dislikes :
{post['reactions']['dislikes']}

Views :
{post['views']}

User ID :
{post['userId']}
"""

    documents.append(Document(page_content=text))

print("Documents Created :", len(documents))


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

print("Chunks Created :", len(docs))


embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vector_db = FAISS.from_documents(docs, embedding)

print("FAISS Vector Database Created")


retriever = vector_db.as_retriever(
    search_kwargs={"k":3}
)


llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama-3.3-70b-versatile",
    temperature=0
)


qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

print("\nRAG Chatbot Ready")
print("Type 'exit' to quit.")


while True:

    question = input("\nAsk Question : ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    result = qa.invoke({"query": question})

    print("\nAnswer:")
    print(result["result"])

    print("\nRetrieved Documents:")

    for i, doc in enumerate(result["source_documents"], start=1):
        print(f"\n------------- Document {i} -------------")
        print(doc.page_content)