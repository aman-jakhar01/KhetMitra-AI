# KhetMitra AI 🌾

KhetMitra AI is an AI-powered wheat farming assistant designed to help farmers make better farming decisions using Machine Learning, Deep Learning, RAG, and AI Agents.

## Features

- **Crop Doctor:** Detects wheat leaf conditions, including healthy leaves, nitrogen deficiency, and leaf rust.
- **Yield Prediction:** Predicts wheat yield using historical and weather-related data.
- **Weather & Irrigation:** Provides weather information and irrigation insights.
- **AI Farming Assistant:** Answers farming questions using agricultural documents through Retrieval-Augmented Generation (RAG).
- **AI Agents:** Uses LangGraph to route farming queries to the appropriate AI capabilities.

## Tech Stack

- Python
- Streamlit
- PyTorch & ResNet18
- XGBoost
- LangChain & LangGraph
- FAISS
- Hugging Face Embeddings
- Groq LLM API
- Open-Meteo API

## Run Locally

```bash
git clone https://github.com/aman-jakhar01/KhetMitra-AI
cd KhetMitra-AI

pip install -r requirements.txt
streamlit run app.py
```

Configure your Groq API key in the `.env` file before running the application.

## Project Goal

To make AI-powered agricultural insights more accessible and help wheat farmers with crop health monitoring, yield estimation, weather awareness, and farming guidance.