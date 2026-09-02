from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

app = FastAPI(title="Task 1 AI Assistant Backend", version="1.0")

from fastapi.middleware.cors import CORSMiddleware

# Add CORS middleware to allow requests from the frontend container
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local testing/Docker
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods including OPTIONS, POST, etc.
    allow_headers=["*"],
)
# Custom offline embedding function that requires zero downloads
class SimpleEmbeddingFunction(EmbeddingFunction[Documents]):
    def __call__(self, input: Documents) -> Embeddings:
        # Generate simple pseudo-embeddings based on text length/characters for local testing
        embeddings = []
        for text in input:
            # Create a dummy 4-dimensional vector based on the text string
            val = float(len(text))
            embeddings.append([val, val * 0.5, val * 0.25, 0.1])
        return embeddings

# Initialize ChromaDB client with our custom offline embedding function
chroma_client = chromadb.Client()
custom_ef = SimpleEmbeddingFunction()
collection = chroma_client.get_or_create_collection(name="task1_docs", embedding_function=custom_ef)

# Simple native Python text chunker
def simple_chunker(text: str, chunk_size: int = 100, overlap: int = 20):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks

# Ingest sample document on startup if collection is empty
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
    prompt: str = Field(description="The user input prompt for the assistant.")
    use_tool: bool = Field(default=False, description="Flag to simulate tool calling.")

class AssistantResponse(BaseModel):
    status: str
    response: str
    retrieved_context: list[str] = []
    tool_output: str | None = None

@app.get("/")
async def root():
    return {"message": "Task 1 AI Assistant Backend with RAG is running."}

@app.post("/chat", response_model=AssistantResponse)
async def chat_endpoint(request: QueryRequest):
    try:
        query = request.prompt
        
        results = collection.query(query_texts=[query], n_results=2)
        retrieved_chunks = results.get("documents", [[]])[0]
        
        tool_result = None
        if request.use_tool:
            tool_result = f"Tool Execution Result: Processed query parameters for '{query}' successfully."

        context_str = " ".join(retrieved_chunks)
        reply = f"Generated answer based on local RAG context. Context found: [{context_str}]"
        
        return AssistantResponse(
            status="success",
            response=reply,
            retrieved_context=retrieved_chunks,
            tool_output=tool_result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))