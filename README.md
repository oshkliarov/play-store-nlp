# App Intelligence Dashboard

A modular, production-ready NLP dashboard designed to extract, sanitize, and analyze mobile application reviews from Google Play and the Apple App Store. The project features a dual-engine architecture allowing users to choose between a local multilingual BERT model (offline inference) and the cloud-based Gemini API (high-accuracy LLM analysis).

## Features

* **Multi-Platform Scraping:** Extracts user reviews seamlessly from Google Play and the official Apple App Store RSS feed.
* **Dual-Engine NLP Analysis:** 
  * *Local Model:* Uses a fine-tuned multilingual BERT model and Zero-Shot classification for offline inference.
  * *Cloud Engine:* Integrates Google Gemini API for context-aware, highly accurate sentiment analysis and structured JSON categorization.
* **Smart Data Sanitization:** Advanced RegEx filtering and heuristic rules to eliminate spam, gibberish, user tags, and non-contextual single-word reviews.
* **Modular Architecture:** Clean separation of concerns (`app.py` for UI, `scraper.py` for data ingestion, and `nlp_engine.py` for ML processing).
* **Interactive Visualization:** Built with Streamlit and Plotly for KPI metrics, sentiment distribution charts, and time-series trends.

## Tech Stack

* **Language:** Python 3.10+
* **Framework:** Streamlit, Plotly
* **Data Processing:** Pandas, Regex
* **Machine Learning & AI:** Hugging Face Transformers, PyTorch, Google GenAI SDK (Gemini)
* **Scraping:** `google-play-scraper`, `requests`

## Project Structure

```text
app-intelligence-nlp/
│
├── app.py             # Streamlit user interface & main orchestration
├── scraper.py         # Data ingestion and cleaning module
├── nlp_engine.py      # Dual-engine NLP processing (Local BERT & Gemini API)
├── requirements.txt   # Project dependencies
└── README.md          # Project documentation