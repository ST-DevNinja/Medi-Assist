# MediAssist

**MediAssist** is a modular AI-powered medical assistant platform that combines Information Retrieval, Machine Learning, and AI logic to assist in medical diagnosis, disease prediction, and health information retrieval.

---

## Features

- **Ask Medi-Assist:**  
  A Retrieval-Augmented Generation (RAG) module using SentenceTransformers and FAISS for medical document retrieval and LLM (Groq LLaMA 3-70B) for context-aware answer generation.

- **Disease Predictor:**  
  Machine Learning models to predict diseases such as diabetes, lung cancer, prostate cancer, and skin cancer based on patient data.

- **Find Nearby Doctor:**  
  AI logic module planned for integrating doctor recommendation based on symptoms and location.

---

## Project Structure

MediAssist/
├── ai_module/ # AI algorithms (e.g. A* search, triage logic)
├── data/ # Medical knowledge base & datasets
├── ir_module/ # Information Retrieval and RAG system
├── ml_module/ # Machine Learning models for disease prediction
├── text_files/ # Auxiliary files and scripts
├── ui/ # Streamlit UI application
└── README.md # Project overview and instructions



---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ST-DevNinja/Medi-Assist.git
   cd Medi-Assist

2. Usage
To start the Streamlit UI app:

streamlit run ui/app.py
