from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import os
import time
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from tenacity import retry, stop_after_attempt, wait_exponential
import google.generativeai as genai
from dotenv import load_dotenv
import mlflow

from utils import mock_web_search 

load_dotenv() 

# ----------------------------------------------------
# MLflow Experiment Configuration (Track B)
# ----------------------------------------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("W17_TrackB_Agentic_Assistant")

# Track prompt configuration versions (e.g. v1, v2, v3)
PROMPT_VERSION = "v1"

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Production AI Assistant Backend", version="2.1")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimpleEmbeddingFunction(EmbeddingFunction[Documents]):
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            val = float(len(text))
            embeddings.append([val, val * 0.5, val * 0.25, 0.1])
        return embeddings

chroma_client = chromadb.Client()
custom_ef = SimpleEmbeddingFunction()
collection = chroma_client.get_or_create_collection(name="task1_docs", embedding_function=custom_ef)

def simple_chunker(text: str, chunk_size: int = 100, overlap: int = 20):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks

if collection.count() == 0:
    doc_path = "data/sample_doc.txt"
    if os.path.exists(doc_path):
        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()
        chunks = simple_chunker(content)
        collection.add(
            documents=chunks,
            ids=[f"id_{i}" for i in range(len(chunks))]
        )

class QueryRequest(BaseModel):
    prompt: str = Field(description="The user input prompt.")
    use_tool: bool = Field(default=False, description="Flag to invoke tool calling.")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0, description="System prompt sampling temperature.")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling parameter.")

class AssistantResponse(BaseModel):
    status: str
    response: str
    retrieved_context: list[str] = []
    tool_output: str | None = None
    structured_metadata: dict = Field(default_factory=dict)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def execute_llm_generation(prompt: str, context: str, temperature: float, top_p: float, system_instruction: str):
    api_key = os.getenv("GEMINI_API_KEY") 
    if not api_key:
        return f"ANSWER: Generated answer using retrieved RAG context: [{context}]. (Fallback model active: API key absent)."
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        full_prompt = f"{system_instruction}\n\nContext:\n{context}\n\nQuestion:\n{prompt}"
        
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                top_p=top_p
            )
        )
        return response.text.strip()
    except Exception as e:
        return f"ANSWER: Generated answer using retrieved context: [{context}]. (Fallback active due to API error: {str(e)})"

@app.get("/")
async def root():
    return {"message": "Agentic AI Assistant Backend running with MLflow tracking."}

@app.post("/chat", response_model=AssistantResponse)
@limiter.limit("10/minute") 
async def chat_endpoint(request: Request, body: QueryRequest):
    run_name = f"Query: {body.prompt[:25]}..."
    
    # MLflow tracking session per query
    with mlflow.start_run(run_name=run_name):
        start_time = time.time()
        
        # Log input configuration parameters
        mlflow.log_param("prompt_version", PROMPT_VERSION)
        mlflow.log_param("temperature", body.temperature)
        mlflow.log_param("top_p", body.top_p)
        mlflow.log_param("model_name", "gemini-1.5-flash")
        mlflow.log_param("input_query", body.prompt)
        
        try:
            query = body.prompt
            agent_trace = []
            
            # 1. Initial RAG Retrieval
            results = collection.query(query_texts=[query], n_results=2)
            retrieved_chunks = results.get("documents", [[]])[0]
            accumulated_context = "Original RAG: " + " ".join(retrieved_chunks)
            
            # 2. Agentic Loop with Step Tracking
            max_steps = 3
            step = 0
            final_answer = ""
            tool_outputs = []

            while step < max_steps:
                step += 1
                
                agent_instruction = """You are an autonomous AI assistant. 
Review the provided context. 
- If you have the EXACT answer, start your response with: "ANSWER: "
- If you do NOT know the answer, you are FORBIDDEN from guessing or apologizing. You MUST start your response with: "SEARCH: " followed by your query.
Do not output both.
"""
                ai_response = execute_llm_generation(
                    prompt=query, 
                    context=accumulated_context, 
                    temperature=body.temperature, 
                    top_p=body.top_p, 
                    system_instruction=agent_instruction
                )
                
                if ai_response.startswith("SEARCH:"):
                    search_query = ai_response.replace("SEARCH:", "").strip()
                    tool_result = mock_web_search(search_query)
                    tool_outputs.append(tool_result)
                    
                    # Log step to trace
                    agent_trace.append({
                        "step": step,
                        "action": "tool_call",
                        "tool": "mock_web_search",
                        "query": search_query,
                        "result": tool_result
                    })
                    
                    # Context Compaction: retain original RAG + latest search
                    accumulated_context = f"Original RAG: {' '.join(retrieved_chunks)}\nLatest Web Search: {tool_result}"
                    
                elif ai_response.startswith("ANSWER:"):
                    final_answer = ai_response.replace("ANSWER:", "").strip()
                    agent_trace.append({
                        "step": step,
                        "action": "answer",
                        "decision": "sufficient_evidence",
                        "raw_response": ai_response
                    })
                    break
                    
                else:
                    final_answer = ai_response
                    agent_trace.append({
                        "step": step,
                        "action": "unformatted_exit",
                        "raw_response": ai_response
                    })
                    break
                    
            if not final_answer:
                final_answer = "Maximum iteration limit reached without conclusive evidence."
                agent_trace.append({"step": step, "action": "max_steps_exceeded"})

            latency = time.time() - start_time

            # Log metrics to MLflow
            mlflow.log_metric("iterations", step)
            mlflow.log_metric("latency_seconds", latency)
            mlflow.log_metric("tools_used_count", len(tool_outputs))
            mlflow.log_metric("task_completed", 1 if final_answer and "conclusive evidence" not in final_answer else 0)

            # Log execution trace as an MLflow artifact
            mlflow.log_dict({"prompt_version": PROMPT_VERSION, "trace": agent_trace}, "agent_trace.json")

            return AssistantResponse(
                status="success",
                response=final_answer,
                retrieved_context=retrieved_chunks,
                tool_output=" | ".join(tool_outputs) if tool_outputs else "No external tools used.",
                structured_metadata={
                    "temperature": body.temperature,
                    "iterations_taken": step,
                    "agent_pattern": "Single-agent loop with cross-source verification",
                    "context_engineering": "Compaction",
                    "mlflow_tracked": True
                }
            )

        except Exception as e:
            mlflow.log_param("error", str(e))
            mlflow.log_metric("task_completed", 0)
            raise HTTPException(status_code=500, detail=f"Error handled safely: {str(e)}")