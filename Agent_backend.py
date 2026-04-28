from __future__ import annotations

import os
import sqlite3
import tempfile
import requests

from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command

load_dotenv()

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------
# FAISS persistence
# -------------------
 
FAISS_DIR = "faiss_indexes"
os.makedirs(FAISS_DIR, exist_ok=True)

# -------------------
# PDF retriever store
# -------------------
 
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(therad_id: Optional[str]):
    """
    Returning the retriever for this thread.
    First checking in-memory cache; if missing, tries to load
    the persisted FAISS index from disk so the retriever
    survives a backend restart.
    """

    if not therad_id:
        return None
    
    # 1. In-memory chache hit
    if therad_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[therad_id]
    
    # 2. Disk fallback -> reloading persisted FAISS index
    index_path = os.path.join(FAISS_DIR, str(therad_id))

    if os.path.exists(index_path):
        try:
            vector_store = FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 4, "fetch_k": 20},
            )
            _THREAD_RETRIEVERS[therad_id] = retriever
            return retriever
        except Exception:
            pass

    return None

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None):

    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")
    
    # Saves the uploaded pdf bytes as a real .pdf file so it can be used by PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name
 
    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
 
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = splitter.split_documents(docs)
 
        # ---------- CLEAN TEXT ----------
        texts = []
        metadatas = []
 
        for doc in chunks:
            text = doc.page_content
 
            if text is None:
                continue
 
            # Removing spaces
            text = str(text).strip()

            # Removing null bytes and non-UTF-8 characters that break the tokenizer
            text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

            # Striping control characters (except newlines and tabs) that cause TextEncodeInput errors
            text = "".join(ch for ch in text if ch >= " " or ch in "\n\t")
            text = text.strip()

            if len(text) == 0:
                continue
 
            texts.append(text)
            metadatas.append(doc.metadata)
 
        if len(texts) == 0:
            raise ValueError("No valid text extracted from the PDF")
        
        # ---------- VECTOR STORE ----------
        vector_store = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
        )
 
        # Saving FAISS index to disk so it survives restarts
        index_path = os.path.join(FAISS_DIR, str(thread_id))
        vector_store.save_local(index_path)
 
        # Using MMR retrieval for diverse, non-redundant chunks
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 20},
        )
 
        _THREAD_RETRIEVERS[str(thread_id)] = retriever
 
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(texts),
        }
 
        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(texts),
        }
 
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass



# -------------------
# Tools
# -------------------
 
search_tool = DuckDuckGoSearchRun(region="us-en")
 
 
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Performing a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
 
        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}
 
 
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage. Returns price data or a descriptive error.
    """
    if not API_KEY:
        return {"error": "ALPHA_VANTAGE_API_KEY is not configured on the server."}
 
    try:
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={API_KEY}"
        )

        # Calling API
        r = requests.get(url, timeout=10)
        r.raise_for_status()

        data = r.json()

        # Extracting Stock data
        quote = data.get("Global Quote", {})
        if not quote:
            return {
                "error": (
                    f"No data found for symbol '{symbol}'. "
                    "Check the ticker or try again later."
                )
            }
 
        return {
            "symbol": quote.get("01. symbol", symbol),
            "price": quote.get("05. price"),
            "open": quote.get("02. open"),
            "high": quote.get("03. high"),
            "low": quote.get("04. low"),
            "volume": quote.get("06. volume"),
            "latest_trading_day": quote.get("07. latest trading day"),
            "previous_close": quote.get("08. previous close"),
            "change": quote.get("09. change"),
            "change_percent": quote.get("10. change percent"),
        }
 
    except requests.Timeout:
        return {"error": "Request timed out while fetching stock data. Try again."}
    except requests.RequestException as e:
        return {"error": f"Network error fetching stock data: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}
 
 
@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.
 
    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision ("yes" / anything else).
    """
 
    # This pauses the graph and returns control to the owner.
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")
 
    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined by the owner.",
            "symbol": symbol,
            "quantity": quantity,
        }
 
 
@tool
def sell_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate selling a given quantity of a stock symbol.
 
    HUMAN-IN-THE-LOOP:
    Before confirming the sale, this tool will interrupt
    and wait for a human decision ("yes" / anything else).
    """
 
    # This pauses the graph and returns control to the caller
    decision = interrupt(f"Approve selling {quantity} shares of {symbol}? (yes/no)")
 
    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Sell order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }
    else:
        return {
            "status": "cancelled",
            "message": f"Sale of {quantity} shares of {symbol} was declined by the owner.",
            "symbol": symbol,
            "quantity": quantity,
        }
 
 
@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    Returns source page numbers alongside the extracted context.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }
 
    result = retriever.invoke(query)
 
    # Including page citations so users can verify the source
    context = [
        f"[Page {doc.metadata.get('page', '?')}]: {doc.page_content}"
        for doc in result
    ]
    metadata = [doc.metadata for doc in result]
 
    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }
 
 
tools = [search_tool, get_stock_price, purchase_stock, sell_stock, calculator, rag_tool]
 
llm_with_tools = llm.bind_tools(tools)
 
# -------------------
# State
# -------------------
 
 
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_id: Optional[str]
    uploaded_filename: Optional[str]   # track which document is loaded
    tool_call_count: int               # guard against infinite tool loops
 
 
# -------------------
# Nodes
# -------------------
 
MAX_TOOL_CALLS = 10
 
 
def chat_node(state: ChatState, config=None):
 
    thread_id = state.get("thread_id")
 
    # Fallback: read from config if state doesn't have it yet
    if not thread_id and config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")
 
    # Guard against infinite tool-call loops
    tool_call_count = state.get("tool_call_count", 0)
    if tool_call_count > MAX_TOOL_CALLS:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I've made too many tool calls trying to answer your question. "
                        "Please rephrase or break it into smaller steps."
                    )
                )
            ],
            "tool_call_count": 0,
        }
 
    uploaded_filename = state.get("uploaded_filename", "")
    doc_hint = (
        f"A PDF named '{uploaded_filename}' is indexed for this thread. "
        "Use rag_tool to answer document questions."
        if uploaded_filename
        else "No PDF has been uploaded for this thread yet."
    )
 
    system_message = SystemMessage(
        content=(
            "You are a multi-tool financial and document assistant.\n\n"
            "TOOLS AVAILABLE:\n"
            "- rag_tool: Answer questions from the uploaded PDF. Always pass thread_id.\n"
            "- get_stock_price: Get real-time stock prices by ticker symbol. Never guess prices.\n"
            "- purchase_stock: Simulate buying stocks — requires human approval.\n"
            "- sell_stock: Simulate selling stocks — requires human approval.\n"
            "- calculator: Arithmetic operations (add / sub / mul / div).\n"
            "- duckduckgo_search: Search the web for current information.\n\n"
            "RULES:\n"
            "- Always use rag_tool for document questions, not your own knowledge.\n"
            "- Never guess stock prices — always call get_stock_price.\n"
            f"- Current thread_id is `{thread_id}` — include it in every rag_tool call.\n"
            f"- {doc_hint}\n"
        )
    )
 
    messages = [system_message, *state["messages"]]
 
    response = llm_with_tools.invoke(messages, config=config)
 
    return {
        "messages": [response],
        "thread_id": thread_id,
        "tool_call_count": tool_call_count + 1,
    }
 
 
tool_node = ToolNode(tools)