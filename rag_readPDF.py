import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

loader = PyPDFLoader("/Users/pragativisghwakarma/Documents/PROJECTS/GenAI_Tranning/RAG_Example/Data/company_policy.pdf ")

documents = loader.load()

print(f"\nTotal Pages Loaded : {len(documents)}")

for i, doc in enumerate(documents):
    print(f"\nPage {i + 1} : {doc.page_content[:100]}...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs  = splitter.split_documents(documents) 

print(f"\nTotal Chunks Created : {len(docs)}")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_db = FAISS.from_documents(docs, embeddings)

print("\nVector Database Created Successfully!")

retriever = vector_db.as_retriever(search_kwargs={"k": 3})

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

print("\nRAG Model Created Successfully!")

while True:
    question = input("\nAsk a Question (or type 'exit' to quit): ")

    if question.lower() == "exit":
        break

    result = qa.invoke({"query": question})

    print("\nAnswer : ", result['result'])

    print("\nRetrieved Documents : ")

    for i, doc in enumerate(result['source_documents']):
        print(f"\nChunk {i + 1} :")
        print("----------------------------------------------")
        print(f" - {doc.page_content}")