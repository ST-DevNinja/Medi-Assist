import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Reduce TensorFlow noise
import time
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import torch
import json

# ----------------------------
# Environment & Device Setup
# ----------------------------

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"GPU available: {use_cuda}")

# ----------------------------
# Load Embedding Model (SentenceTransformer)
# ----------------------------
start = time.time()
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
if use_cuda:
    embedding_model = embedding_model.to(device)
print(f"Embedding model loaded in {time.time() - start:.2f} seconds.")

# ----------------------------
# Groq API Config
# ----------------------------

# Store your Groq API key securely, e.g. in environment variable
GROQ_API_KEY = "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"
if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY environment variable not set.")
    exit(1)

# *** Correct API endpoint and domain ***
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"  # use the model from your example curl

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

# ----------------------------
# Medical Synonyms for Query Expansion
# ----------------------------
MEDICAL_SYNONYMS = {
    "stomach pain": ["abdominal pain", "belly pain", "gastric pain", "tummy ache"],
    "headache": ["head pain", "cephalalgia"],
    "fever": ["high temperature", "pyrexia"],
    "cough": ["dry cough", "wet cough", "persistent cough"],
    "cold": ["runny nose", "nasal congestion", "sneezing"],
    "nausea": ["feeling sick", "urge to vomit"],
    "vomiting": ["throwing up", "emesis"],
    "diarrhea": ["loose motion", "frequent bowel movement"],
    "pain": ["discomfort", "soreness", "ache", "burning sensation", "sharp pain", "tenderness"]
}

# ----------------------------
# IR + RAG Class
# ----------------------------
class IRRetriever:
    def __init__(self, data_path=os.path.join(os.path.dirname(__file__), '..', 'data')):
        self.data_path = data_path
        self.model = embedding_model
        self.documents = []
        self.embeddings = []
        self.index = None
        self.doc_map = []

    def load_documents(self):
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.txt'):
                    full_path = os.path.join(root, file)
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        self.documents.append(content)
                        rel_path = os.path.relpath(full_path, self.data_path)
                        self.doc_map.append(rel_path)

    def embed_and_index(self):
        self.embeddings = self.model.encode(self.documents, convert_to_numpy=True)
        self.embeddings = normalize(self.embeddings)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def expand_query(self, query):
        expanded = [query]
        query_lower = query.lower()
        for key, synonyms in MEDICAL_SYNONYMS.items():
            if key in query_lower:
                expanded.extend(synonyms)
        return expanded

    def search(self, query, top_k=3, threshold=0.2):
        expanded_queries = self.expand_query(query)
        query_embeddings = self.model.encode(expanded_queries, convert_to_numpy=True)
        query_embeddings = normalize(query_embeddings)
        query_vec = np.mean(query_embeddings, axis=0, keepdims=True)
        D, I = self.index.search(query_vec, top_k)

        results = []
        for idx, score in zip(I[0], D[0]):
            if score >= threshold:
                results.append((self.doc_map[idx], self.documents[idx], score))
        return results

    def generate_answer(self, query, context):
        if not context or context.strip() == "":
            context = "No relevant medical context found."

        final_prompt = (
            "You are a knowledgeable and empathetic medical assistant AI. Based on the following medical context, "
            "answer the user's question with detailed and helpful advice. Include possible causes, symptoms, treatment options, "
            "and advice on when to seek medical attention. Do not repeat the question."
        )

        # Groq expects a chat format with 'messages' array (like OpenAI chat completions)
        messages = [
            {"role": "system", "content": final_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
        ]

        payload = {
            "model": GROQ_MODEL_NAME,
            "messages": messages,
            "max_tokens": 600,
            "temperature": 0.7,
            "top_p": 0.9,
            "n": 1,
            "stop": ["<END>"]
        }

        #print("\n🧠 Payload sent to Groq API:")
        #print(json.dumps(payload, indent=2))

        try:
            response = requests.post(GROQ_API_URL, headers=HEADERS, json=payload)
            response.raise_for_status()
            data = response.json()
            # Groq API follows OpenAI chat format
            generated_text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            return generated_text
        except Exception as e:
            print("❌ Error calling Groq API:", e)
            return "Sorry, I couldn't generate an answer at this time."

# ----------------------------
# Main Script
# ----------------------------
if __name__ == "__main__":
    ir = IRRetriever()
    ir.load_documents()
    ir.embed_and_index()

    while True:
        query = input("\nEnter your medical query (or type 'exit' to quit): ").strip()
        if query.lower() in {'exit', 'quit'}:
            print("Exiting search.")
            break

        results = ir.search(query, top_k=3, threshold=0.2)

        if not results:
            print("\nNo relevant documents found above the threshold.")
            combined_context = ""
        else:
            print("\n---------Top retrieved documents and their similarity scores:--------\n")
            for name, _, score in results:
                print(f"{name} (Score: {score:.3f})")

            combined_context = "\n".join([text for _, text, _ in results])

        answer = ir.generate_answer(query, combined_context)
        print("\n\n\n🤖 LLM (RAG) Generated Answer:\n")
        print(answer)


