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

1. **Data Crawling**: Collected structured and unstructured SHL assessment data using `Scrapy`.
2. **NLP Preprocessing**:
   - Extracted keywords using `NLTK` and summarization using `transformers` pipeline.
   - Created TF-IDF features for additional filtering.
3. **Embedding Generation**:
   - Used `sentence-transformers` (`all-MiniLM-L6-v2`) to generate dense vector embeddings for:
     - Keywords
     - Summaries
     - Test types
4. **Similarity Search**:
   - Used `FAISS` for fast nearest-neighbor search on embedding space.
5. **Frontend & API**:
   - `Streamlit` for interactive frontend.
   - `FastAPI` to serve predictions as JSON.
6. **Demo Hosting**:
   - Hosted on `localhost`, shared externally via `ngrok`.

