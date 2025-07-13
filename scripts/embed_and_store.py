from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def load_chunks(path="data/chunks.txt"):
    with open(path, "r", encoding="utf-8") as f:
        chunks = [chunk.strip() for chunk in f.read().split("\n\n") if chunk.strip()]
    return chunks

def embed_and_store():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunks = load_chunks()

    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    faiss.write_index(index, "data/faiss_index.index")
    with open("data/chunks.npy", "wb") as f:
        np.save(f, chunks)
    print("✅ Embeddings stored in FAISS and chunks.npy")

if __name__ == "__main__":
    embed_and_store()
