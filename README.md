# 🚀 Enterprise RAG Chatbot

An Enterprise-grade **Retrieval-Augmented Generation (RAG)** chatbot built using **Python**, **LangChain**, and **Large Language Models (LLMs)**. The application enables users to ask natural language questions over custom documents by combining semantic search with AI-powered response generation.

---

## 📌 Features

- 📄 Multi-document ingestion
- 🔍 Semantic document retrieval
- 🤖 AI-powered question answering
- 🧠 Vector embeddings
- 📚 Context-aware responses
- ⚡ Fast retrieval pipeline
- 🔄 Modular architecture
- 🔐 Environment variable configuration
- 📂 Easy to extend with additional data sources

---

## 🛠️ Technologies Used

- Python
- LangChain
- FAISS / ChromaDB
- Vector Embeddings
- Retrieval-Augmented Generation (RAG)
- Large Language Models
- Environment Variables (.env)

---

## 📂 Project Structure

```
Enterprise-RAG-Chatbot/
│
├── Assignment1.py
├── Assignment2.py
├── Assignment3.py
├── Assignment4.py
├── data/
├── vectorstore/
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/Pragativishwakarma/Enterprise-RAG-Chatbot.git
```

### Navigate

```bash
cd Enterprise-RAG-Chatbot
```

### Create Virtual Environment

Mac/Linux

```bash
python3 -m venv venv
```

Windows

```bash
python -m venv venv
```

### Activate

Mac/Linux

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file.

```env
OPENAI_API_KEY=your_api_key
GROQ_API_KEY=your_api_key
```

---

## ▶️ Run

Example:

```bash
python Assignment1.py
```

or

```bash
python Assignment4.py
```

---

## 🏗️ RAG Workflow

```
Documents
     │
     ▼
Document Loader
     │
     ▼
Text Splitter
     │
     ▼
Embeddings
     │
     ▼
Vector Database
     │
     ▼
Retriever
     │
     ▼
Large Language Model
     │
     ▼
Answer Generation
```

---

## 📖 Learning Outcomes

This project demonstrates:

- Retrieval-Augmented Generation
- LangChain Pipelines
- Prompt Engineering
- Vector Search
- Embedding Models
- Semantic Search
- Document Chunking
- AI Chatbots
- LLM Integration
- Enterprise Knowledge Retrieval

---

## 🚀 Future Improvements

- Streamlit UI
- PDF Upload
- Conversation Memory
- Hybrid Search
- Multi-Agent Workflow
- Citation Support
- Reranking
- Web Search Integration
- Authentication
- Cloud Deployment

---

## 👩‍💻 Author

**Pragati Vishwakarma**

AI & Data Science Engineer

GitHub:
https://github.com/Pragativishwakarma

LinkedIn:
https://www.linkedin.com/in/pragati-vishwakarma-893ba8284

---

⭐ If you found this project useful, please consider giving it a Star.
