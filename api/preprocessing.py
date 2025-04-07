from sentence_transformers import SentenceTransformer
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path

def preprocess_data():
    # Define paths using relative paths
    base_dir = Path(__file__).parent.parent  # Goes up one level from /api
    processed_data_path = base_dir / "processing" / "processed_dataset.json"
    embeddings_path = base_dir / "api" / "data" / "embeddings.npy"
    
    # Create data directory if it doesn't exist
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)

    # Load your preprocessed data
    print(f"Loading data from: {processed_data_path}")
    try:
        with open(processed_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {processed_data_path}")
        print("Current working directory:", Path.cwd())
        return

    # Initialize model
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create combined text for embeddings
    print("Creating embedding texts...")
    embedding_texts = []
    for item in tqdm(data, desc="Processing assessments"):
        components = [
            item.get("summarization", ""),
            " ".join(item.get("keywords", [])),
            " ".join(item.get("test_types_full", []))
        ]
        combined_text = " ".join(components).strip()
        embedding_texts.append(combined_text)
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = model.encode(
        embedding_texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Save embeddings
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings to: {embeddings_path}")
    print(f"Successfully processed {len(data)} assessments")

if __name__ == "__main__":
    preprocess_data()