import os

from dotenv import load_dotenv
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Reduce TensorFlow noise
import time
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import torch


# ----------------------------
# Load environment variables
# ----------------------------
#env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ ERROR: GROQ_API_KEY not set in environment.")
    exit(1)





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

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

# ----------------------------
# Medical Synonyms for Query Expansion
# ----------------------------
MEDICAL_SYNONYMS = {
    "stomach pain": [
        "abdominal pain", "belly pain", "gastric pain", "tummy ache", 
        "lower abdominal pain", "upper abdominal pain", "cramping", "digestive discomfort"
    ],
    "headache": [
        "head pain", "cephalalgia", "migraine", "tension headache", 
        "pounding head", "splitting headache", "head throbbing"
    ],
    "fever": [
        "high temperature", "pyrexia", "elevated body temperature", 
        "hot flashes", "chills", "febrile"
    ],
    "cough": [
        "dry cough", "wet cough", "persistent cough", "chesty cough", 
        "whooping cough", "irritating cough", "throat clearing"
    ],
    "cold": [
        "runny nose", "nasal congestion", "sneezing", "blocked nose", 
        "sniffles", "postnasal drip", "sore throat from cold"
    ],
    "nausea": [
        "feeling sick", "urge to vomit", "queasiness", "upset stomach", 
        "stomach unease", "sick to the stomach"
    ],
    "vomiting": [
        "throwing up", "emesis", "retching", "projectile vomiting", 
        "expelling stomach contents", "barfing", "puking"
    ],
    "diarrhea": [
        "loose motion", "frequent bowel movement", "watery stool", 
        "intestinal upset", "bowel urgency", "digestive disturbance"
    ],
    "pain": [
        "discomfort", "soreness", "ache", "burning sensation", 
        "sharp pain", "tenderness", "stabbing pain", "throbbing", 
        "nagging pain", "localized pain", "chronic pain"
    ],
    "fatigue": [
        "tiredness", "exhaustion", "lethargy", "lack of energy", 
        "weariness", "feeling drained", "low stamina"
    ],
    "shortness of breath": [
        "breathlessness", "difficulty breathing", "dyspnea", 
        "labored breathing", "tight chest", "trouble catching breath"
    ],
    "rash": [
        "skin irritation", "redness", "hives", "dermatitis", 
        "itchy skin", "skin inflammation", "bumps on skin"
    ]
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

        if context.strip() == "" or context.strip().lower().startswith("no relevant"):
            return "No relevant information was found to answer your question."

        
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
            print("Retrying with lower threshold ...")
            results = ir.search(query, top_k=3, threshold=0.1)

        if not results:
            print("Still no relevant documents found.")
            combined_context = ""
            #continue
        else:
            print("\n---------Top retrieved documents and their similarity scores:--------\n")
            scores = []
            for name, _, score in results:
                scores.append(score)
                print(f"{name} (Score: {score:.3f})")

            # Evaluation Metric: Average Similarity Score
            avg_score = sum(scores) / len(scores) if scores else 0.0
            print(f"\n📊 Evaluation Metric: Average Similarity Score of Top-{len(scores)} = {avg_score:.3f}")

            combined_context = "\n".join([text for _, text, _ in results])


        answer = ir.generate_answer(query, combined_context)
        print("\n\n\n🤖 LLM (RAG) Generated Answer:\n")
        print(answer)


