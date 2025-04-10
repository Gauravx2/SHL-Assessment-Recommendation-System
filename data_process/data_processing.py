import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent
RAW_FILE = DATA_DIR.parent / "crawler" / "data" / "shl_products.json"
PROCESSED_JSON = DATA_DIR / "processed_dataset.json"

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

def parse_duration(text):
    """Extract numerical duration from text"""
    # Try to extract number following an "=" sign
    match = re.search(r'=\s*(\d+)', text)
    if match:
        return int(match.group(1))
    # Fallback: try to match a digit sequence preceding "min"
    match = re.search(r'(\d+)\s*min', text, re.IGNORECASE)
    return int(match.group(1)) if match else None

def process_entry(entry):
    """Create optimized entry structure with embedding-ready text"""
    processed = {
        # Core metadata
        "Assessment_name": entry["title"].strip(),
        "URL": entry["url"].strip(),
        
        # Text fields for embedding
        "Description": entry["description"].strip(),
        "Job_levels": entry["job_levels"].strip().rstrip(','),
        "Languages": entry["languages"].strip().rstrip(','),
        "Test_types": [TEST_TYPE_LABELS[t] for t in entry["test_types"]],
        
        # Numerical fields
        "Duration_minutes": parse_duration(entry["duration_minutes"]),
        
        # Boolean flags
        "Remote_testing_support": entry["remote_testing"].lower(),
        "Adaptive/IRT Support": entry["adaptive_supported"].lower(),
        
        # Combined text for embeddings
        "embedding_text": ""
    }

    # Build optimized text for embeddings
    text_parts = []
    
    # Core description
    text_parts.append(processed["Description"])
    
    # Job levels
    if processed["Job_levels"]:
        text_parts.append(f"Applicable job levels: {processed['Job_levels']}")
    
    # Languages
    if processed["Languages"]:
        text_parts.append(f"Available languages: {processed['Languages']}")
    
    # Duration
    if processed["Duration_minutes"]:
        text_parts.append(f"Test duration: {processed['Duration_minutes']} minutes")
    
    # Test types
    if processed["Test_types"]:
        text_parts.append("Measures: " + ", ".join(processed["Test_types"]))
    
    # Remote testing
    text_parts.append(f"Remote testing available: {'Yes' if processed['Remote_testing_support'] else 'No'}")
    
    # Adaptive support
    text_parts.append(f"Adaptive testing: {'Supported' if processed['Adaptive/IRT Support'] else 'Not supported'}")

    # Combine all parts
    processed["embedding_text"] = ". ".join(text_parts)
    
    return processed

def main():
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    processed = [process_entry(e) for e in data]
    
    with open(PROCESSED_JSON, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()
