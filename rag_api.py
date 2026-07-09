import os
import requests
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Get Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is missing. Please check your .env file.")

# Fetch user data from the API
url = "https://jsonplaceholder.typicode.com/users"
response = requests.get(url)
if response.status_code != 200:
    raise ValueError(f"Failed to fetch data from {url}. Status code: {response.status_code}")

users = response.json()
print("Users Loaded :", len(users))

# Create documents from user data
documents = []
for user in users:
    text = f"""
    ID : {user['id']}
    Name : {user['name']}
    Username : {user['username']}
    Email : {user['email']}
    Phone : {user['phone']}
    Website : {user['website']}

    Company :
    {user['company']['name']}

    Catch Phrase :
    {user['company']['catchPhrase']}

    City :
    {user['address']['city']}

    Street :
    {user['address']['street']}
    """
    documents.append(Document(page_content=text))

print("Documents Created :", len(documents))

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = splitter.split_documents(documents)
print("Chunks Created :", len(docs))

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS vector database
vector_db = FAISS.from_documents(docs, embeddings)
print("FAISS Vector Database Created")

# Create retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama-3.3-70b-versatile",
    temperature=0
)

# Create RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

print("\nRAG Model Created")

# Interactive Question-Answering Loop
while True:
    question = input("\nAsk Question (type 'exit' to quit): ")

    if question.lower() == "exit":
        print("Exiting...")
        break

    try:
        result = qa.invoke({"query": question})

        print("\nAnswer: ", result['result'])

        print("\nRetrieved Documents:")
        for doc in result['source_documents']:
            print("----------------------------------------------")
            print(f" - {doc.page_content}")

    except Exception as e:
        print(f"Error: {e}")