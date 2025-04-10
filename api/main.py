from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
import time

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR.parent / "data_process" / "processed_dataset.json"
FAISS_INDEX_PATH = BASE_DIR / "data" / "faiss_index.bin"

# Load resources once at startup
@app.on_event("startup")
def load_assets():
    global assessments, index, model
    
    try:
        # Load assessment data
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            assessments = json.load(f)
        
        # Load FAISS index
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        
        # Load embedding model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Successfully loaded all resources")
    except Exception as e:
        print(f"Failed to initialize: {str(e)}")
        raise e

class Query(BaseModel):
    text: str
    max_results: int = 5

@app.post("/recommend")
async def recommend(query: Query):
    start_time = time.time()
    
    try:
        # Generate query embedding
        query_embedding = model.encode(
            [query.text],
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype(np.float32)

        # FAISS search
        scores, indices = index.search(query_embedding, query.max_results)
        
        # Process results for frontend compatibility
        results = []
        for idx in indices[0]:
            assessment = assessments[idx]
            results.append({
                "Assessment Name": assessment["Assessment_name"],
                "URL": assessment["URL"],
                "Remote Testing": "Yes" if assessment["Remote_testing_support"] else "No",
                "Adaptive Support": assessment.get("Adaptive/IRT Support", "No"),
                "Duration": f"{assessment['Duration_minutes']} minutes",
                "Test Types": ", ".join(assessment["Test_types"])
            })

        return {
            "query": query.text,
            "processing_time": f"{(time.time() - start_time)*1000:.2f}ms",
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {
        "status": "active",
        "assessments_loaded": len(assessments),
        "faiss_index_size": index.ntotal if index else 0
    }