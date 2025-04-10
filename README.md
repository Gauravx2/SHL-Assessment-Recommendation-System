## 🚀 SHL Assessment Recommendation System

This project recommends SHL assessments based on natural language queries or job descriptions. It uses a combination of traditional NLP and modern ML techniques to understand user intent and return the most suitable SHL assessments.

---

### 📌 Features

- 🔍 Accepts natural queries (e.g., “I want to hire a frontend developer with limited time”)
- ⚙️ Returns recommended SHL assessments with test type, time, and relevant skills
- ⚡ Fast and accurate similarity search using vector embeddings
- 🌐 Accessible via web app and API

---

### 🔧 Tech Stack & Tools

| Layer            | Tools / Libraries                                                                 |
|------------------|-----------------------------------------------------------------------------------|
| 🗨️ Crawling       | `Scrapy`                                                                          |
| 🧠 NLP Pipeline   | `NLTK`, `transformers` (pipeline for summarization), `TfidfVectorizer`            |
| 🔍 Embeddings     | `sentence-transformers` (`all-MiniLM-L6-v2`)                                      |
| 📚 Vector Search  | `FAISS`                                                                                                                                                |
| 💥 Web App        | `Streamlit`                                                                        |
| 🧖‍♂️ API Endpoint   | `FastAPI`                                                                          |
| ☁️ Hosting        | `ngrok` for demo link,                          |

---

### 🔗 Important URLs

- **Web Demo**: [🔗 Open Web App](https://d1f5-2405-201-a40b-855-3c46-e9a1-ebbe-2bbe.ngrok-free.app)  
- **API Endpoint**: `https://d1f5-2405-201-a40b-855-3c46-e9a1-ebbe-2bbe.ngrok-free.app/recommend?query=YOUR_QUERY`  
  Example:  
  ```
  GET https://d1f5-...ngrok-free.app/recommend?query=frontend developer with limited time
  ```
- **GitHub Repo**: [https://github.com/Gauravx2/SHL-Assessment-Recommendation-System](https://github.com/Gauravx2/SHL-Assessment-Recommendation-System)

---

### 📄 Project Structure

```
shl_assessment_recommender/
│___crawler/data/shl_product.json scraped data
├── api/                    # FastAPI backend
├── frontend/app.py        # Streamlit web frontend
├── embeddings/             # Precomputed FAISS index and embedding data
├── processing/processed_dataset.json                # Keyword extraction, summarization, processing
├── evaluation/metrics.py                # evaluation functions
├── requirements.txt
├── README.md
└── run_api.sh / run_streamlit.sh
```

---

### 📌 How It Works

 **Data Crawling**: Collected structured and unstructured SHL assessment data using `Scrapy`.
Solution Steps
**Data Preprocessing**

Combined metadata (description, job levels, languages, etc.) into a single embedding-friendly text field.

Standardized formats (e.g., "Yes"/"No" for booleans, comma-separated lists).

Extracted numerical durations using regex.

Embedding Generation

Used sentence-transformers to convert text into 384-dimensional vectors.

Normalized embeddings for cosine similarity.

Stored embeddings as float32 for efficiency.

**FAISS Indexing**

Created a IndexFlatIP (inner product) index for fast similarity search.

Saved index to disk for quick reloading.

**API & Frontend**

FastAPI: Exposed /recommend endpoint to query the FAISS index.

Streamlit: Simple UI to input queries and display results in a table.

CORS configured for local development.

**Key Optimizations**
Efficient Search: FAISS enables millisecond-level query responses.

Scalable: Index supports thousands of assessments with low latency.

User-Friendly: Frontend displays critical fields (name, URL, duration, etc.).
 **Demo Hosting**:
   - Hosted on `localhost`, shared externally via `ngrok`.

