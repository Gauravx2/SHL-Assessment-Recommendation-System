import json
from pathlib import Path
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import _stop_words
import torch
import nltk
from nltk.corpus import stopwords
import logging
from tqdm import tqdm

# Configuration
DATA_DIR = Path(__file__).parent
RAW_FILE = DATA_DIR.parent / "crawler" / "data" / "shl_products.json"
PROCESSED_JSON = DATA_DIR / "processed_assessments.json"
PROCESSED_JSON.parent.mkdir(parents=True, exist_ok=True)

# Test type mapping
TEST_TYPE_LABELS = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations"
}

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(DATA_DIR / "processing.log"),
        logging.StreamHandler()
    ]
)

def initialize_models():
    """Initialize NLP models with error handling"""
    try:
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = set(_stop_words.ENGLISH_STOP_WORDS)

    device = 0 if torch.cuda.is_available() else -1
    
    try:
        summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            device=device,
            torch_dtype=torch.float16 if device == 0 else torch.float32
        )
        tfidf = TfidfVectorizer(
            stop_words=list(stop_words),
            ngram_range=(1, 2),
            max_features=100
        )
        return summarizer, tfidf
    except Exception as e:
        logging.error(f"Model initialization failed: {e}")
        return None, None

def generate_summary(text, summarizer):
    """Generate assessment summary with fallback"""
    try:
        result = summarizer(
            text,
            max_length=60,
            min_length=30,
            do_sample=False,
            truncation=True
        )
        return result[0]['summary_text']
    except Exception as e:
        logging.warning(f"Summarization failed: {e}")
        return text[:150] + "..." if len(text) > 150 else text

def extract_keywords(text, tfidf, entry):
    """Extract keywords with test type context"""
    keywords = []
    
    # Add test type labels first
    if 'test_types' in entry:
        keywords.extend(TEST_TYPE_LABELS.get(t, t) for t in entry['test_types'])
    
    # Add duration if available
    if 'duration_minutes' in entry:
        keywords.append(f"{entry['duration_minutes']} minutes")
    
    # TF-IDF keywords if text available
    if text.strip():
        try:
            tfidf_matrix = tfidf.fit_transform([text])
            features = tfidf.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            top_kws = [features[i] for i in scores.argsort()[::-1] if scores[i] > 0.1][:8]
            keywords.extend(kw for kw in top_kws if kw not in keywords)
        except Exception as e:
            logging.warning(f"TF-IDF failed: {e}")
    
    return [kw for kw in keywords if kw and len(kw) > 2]

def process_entry(entry, summarizer, tfidf):
    """Process a single assessment entry"""
    context = " | ".join(str(v) for k,v in entry.items() if k not in ['summarization', 'keywords'])
    
    return {
        **entry,
        'summarization': generate_summary(context, summarizer),
        'keywords': extract_keywords(context, tfidf, entry),
        'test_types_full': [TEST_TYPE_LABELS.get(t, t) for t in entry.get('test_types', [])]
    }

def main():
    logging.info("Starting assessment processing")
    
    # Load models
    summarizer, tfidf = initialize_models()
    if not summarizer:
        logging.error("Critical model initialization failure")
        return

    # Load data
    try:
        with open(RAW_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Loaded {len(data)} assessments")
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return

    # Process data
    processed = []
    for entry in tqdm(data, desc="Processing"):
        try:
            processed.append(process_entry(entry, summarizer, tfidf))
        except Exception as e:
            logging.error(f"Failed to process entry: {e}")
            processed.append({**entry, 'error': str(e)})

        # Periodic save
        if len(processed) % 10 == 0:
            with open(PROCESSED_JSON, 'w', encoding='utf-8') as f:
                json.dump(processed, f, indent=2)

    # Final save
    with open(PROCESSED_JSON, 'w', encoding='utf-8') as f:
        json.dump(processed, f, indent=2)
    logging.info(f"Saved processed data to {PROCESSED_JSON}")

if __name__ == "__main__":
    
    main()