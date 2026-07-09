import os
import json
from dotenv import load_dotenv

from langchain_core.documents import Document
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

# Read students.json
with open("students.json", "r") as file:
    students = json.load(file)

print("Students Loaded:", len(students))

# Create documents from student data
documents = []
for student in students:
    status = "Passed" if student["marks"] >= 50 else "Failed"
    text = f"""
Student ID: {student['id']}
Name: {student['name']}
Course: {student['course']}
Marks: {student['marks']}
Status: {status}
"""
    documents.append(Document(page_content=text))

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS vector database
vector_db = FAISS.from_documents(documents, embeddings)
print("FAISS Vector Database Created")

# Create retriever (k = number of students to ensure all records are retrieved for calculations like average/most popular course)
retriever = vector_db.as_retriever(search_kwargs={"k": len(students)})

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

print("\nStudent Database RAG Bot Ready")
print("Type 'exit' to quit.")

# Interactive Loop
while True:
    question = input("\nAsk Question: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    if not question.strip():
        continue

    try:
        result = qa.invoke({"query": question})
        print("\nAnswer:")
        print(result["result"])
    except Exception as e:
        print(f"Error: {e}")