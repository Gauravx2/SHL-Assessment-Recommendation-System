from fastapi import FastAPI
from pydantic import BaseModel
import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# allow Streamlit’s origin (adjust if you host elsewhere)
origins = [
    "http://localhost:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "API is running"}


# Path configuration
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR.parent / "processing" / "processed_dataset.json"
EMBEDDINGS_PATH = BASE_DIR / "data" / "embeddings.npy"

# Load data and embeddings
with open(DATA_PATH) as f:
    assessments = json.load(f)
    
embeddings = np.load(EMBEDDINGS_PATH)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

model = SentenceTransformer('all-MiniLM-L6-v2')

class Query(BaseModel):
    text: str
    max_results: int = 10

@app.post("/recommend")
async def recommend(query: Query):
    # Encode query
    query_embedding = model.encode([query.text])
    
    # Search FAISS index
    distances, indices = index.search(query_embedding, query.max_results)
    
    results = []
    for idx in indices[0]:
        assessment = assessments[idx]
        results.append({
            "Assessment Name": assessment["title"],
            "URL": assessment["url"],
            "Remote Testing": assessment["remote_testing"],
            "Adaptive Support": assessment.get("adaptive_supported", "No Information"),
            "Duration": f"{assessment['duration_minutes']} minutes",
            "Test Types": ", ".join(assessment["test_types_full"])
        })
    
    return {"query": query.text, "results": results}