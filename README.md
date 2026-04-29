# 🤖 Multi-Utility AI Agent (LangGraph + RAG + HITL)

A **production-style AI agent** built using **LangGraph, LangChain, and Streamlit** that combines:

* 📄 **Document Intelligence (RAG over PDFs)**
* 📊 **Real-time Financial Data**
* 🧠 **Multi-step reasoning with tool usage**
* ⏸️ **Human-in-the-Loop approvals (HITL)**
* 💾 **Persistent memory across conversations**

---

## 🚀 🔴 Live Demo

👉 **Try the app here:**
🔗 https://agent-using-langgraph-k5zkvjcidrcdryg9q5fxbu.streamlit.app/

> ⚠️ Note: First load may take a few seconds due to cold start.

---

## ✨ Why This Project Stands Out

Most AI chatbot projects are stateless and prompt-based.
This project demonstrates a **real-world AI system design**:

✅ Stateful execution using LangGraph
✅ Intelligent tool orchestration
✅ Safe automation with human approval
✅ Persistent memory (SQLite checkpointing)
✅ Hybrid intelligence (RAG + APIs + reasoning)

---

## 🏗️ Architecture Overview

```id="arch-diagram"
                ┌───────────────────────────┐
                │       Streamlit UI        │
                │  (Chat + File Upload)     │
                └────────────┬──────────────┘
                             │
                             ▼
                ┌───────────────────────────┐
                │      LangGraph Agent      │
                │     (StateGraph Flow)     │
                └────────────┬──────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   RAG Tool   │    │  Stock API   │    │ Web Search   │
│ (FAISS + PDF)│    │ AlphaVantage │    │ DuckDuckGo   │
└──────────────┘    └──────────────┘    └──────────────┘
        │
        ▼
┌───────────────────────────┐
│ Human-in-the-Loop Control │
│ (Approve / Reject Actions)│
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│ Persistent Memory (SQLite)│
└───────────────────────────┘
```

---

## 🧠 Core Capabilities

### 📄 Document Question Answering (RAG)

* Upload PDFs per chat thread
* FAISS-based vector search
* Returns contextual answers with citations

---

### 📊 Financial Intelligence

* Real-time stock prices
* Buy/Sell simulation with approval system

---

### 🛠️ Tool-Oriented Reasoning

The agent dynamically selects tools:

* 🔎 Web Search
* 🧮 Calculator
* 📊 Stock API
* 📚 PDF RAG

---

### ⏸️ Human-in-the-Loop (HITL)

Critical actions require approval:

```text id="f2v8x1"
Approve buying 10 shares of AAPL?
```

---

### 💾 Persistent Conversations

* SQLite checkpointing
* Multi-thread chat support
* Survives app restarts

---

## 🔄 Example Interactions

### 📄 PDF Query

```text id="m7p8q2"
User: Summarize this document
→ Uses RAG → Returns contextual answer
```

### 📊 Stock Query

```text id="n1x7k3"
User: Price of TSLA
→ Calls stock API → Returns real-time data
```

### 💼 Safe Trading

```text id="z9c4v5"
User: Buy 5 shares of AAPL
→ Requires approval → Executes if approved
```

---

## 📁 Project Structure

```id="structure-block"
.
├── Agent_backend.py
├── Agent_frontend.py
├── requirements.txt
├── .gitignore
├── chatbot.db
└── faiss_indexes/
```

---

## ⚙️ Setup

```bash id="setup-block"
git clone https://github.com/Shubham-kumar1-hub/Agent-using-Langgraph.git
cd Agent-using-Langgraph
pip install -r requirements.txt
streamlit run Agent_frontend.py
```

---

## 🔑 Environment Variables

```env id="env-block"
GROQ_API_KEY=your_key
ALPHA_VANTAGE_API_KEY=your_key
```

---

## 🧩 Tech Stack

* LangGraph
* LangChain
* Streamlit
* FAISS
* HuggingFace Embeddings
* Groq (LLaMA 3.3)
* SQLite

---

## 🚀 What This Demonstrates

* Agent-based system design
* Tool orchestration
* RAG implementation
* Human-in-the-loop workflows
* Persistent AI systems

---

## 👨‍💻 Author

**Shubham Kumar**
GitHub: https://github.com/Shubham-kumar1-hub

---

## ⭐ Support

If you found this useful, consider giving it a ⭐
