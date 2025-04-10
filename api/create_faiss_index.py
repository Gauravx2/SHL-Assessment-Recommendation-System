import faiss
import numpy as np
from pathlib import Path

def create_faiss_index():
    base_dir = Path(__file__).parent
    embeddings_path = base_dir / "data" / "embeddings.npy"
    index_path = base_dir / "data" / "faiss_index.bin"

    # Load embeddings
    embeddings = np.load(embeddings_path)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    index.add(embeddings)
    
    # Save index
    faiss.write_index(index, str(index_path))
    print(f"Created FAISS index with {index.ntotal} vectors at {index_path}")

if __name__ == "__main__":
    create_faiss_index()