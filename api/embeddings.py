from sentence_transformers import SentenceTransformer
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path

def preprocess_data():
    base_dir = Path(__file__).parent.parent
    processed_data_path = base_dir / "data_process" / "processed_dataset.json"
    embeddings_path = base_dir / "api" / "data" / "embeddings.npy"
    
    # Create directory if needed
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {processed_data_path}")
    with open(processed_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Initialize model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Extract preprocessed embedding texts
    embedding_texts = [item["embedding_text"] for item in tqdm(data, desc="Loading assessments")]
    
    # Generate embeddings
    embeddings = model.encode(
        embedding_texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=32  # Optimized for typical GPU memory
    )
    
    # Save embeddings
    np.save(embeddings_path, embeddings.astype(np.float32))  # Reduced precision for efficiency
    print(f"Saved {len(embeddings)} embeddings to {embeddings_path}")

if __name__ == "__main__":
    preprocess_data()   