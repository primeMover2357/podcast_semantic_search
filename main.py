import os
import glob
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Load the pre-trained embedding model for semantic understanding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to load and chunk transcripts
def load_transcripts(directory):
    data = []
    for filepath in glob.glob(os.path.join(directory, '*.txt')):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            # Split into chunks (by double newlines for paragraphs; adjust splitter if your transcripts use different separators)
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                data.append({
                    'source': os.path.basename(filepath),
                    'chunk_id': i,
                    'text': chunk
                })
    return data

# Generate vector embeddings
def generate_embeddings(data):
    texts = [item['text'] for item in data]
    embeddings = model.encode(texts)
    return embeddings

# Build FAISS index for similarity search
def build_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
    index.add(np.array(embeddings).astype('float32'))
    return index

# Save index and metadata
def save_index_and_metadata(index, data, index_path='index.faiss', metadata_path='metadata.pkl'):
    faiss.write_index(index, index_path)
    with open(metadata_path, 'wb') as f:
        pickle.dump(data, f)

# Load index and metadata
def load_index_and_metadata(index_path='index.faiss', metadata_path='metadata.pkl'):
    index = faiss.read_index(index_path)
    with open(metadata_path, 'rb') as f:
        data = pickle.load(f)
    return index, data

# Perform semantic search
def semantic_search(query, index, data, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype('float32'), top_k)
    results = []
    for i in range(top_k):
        idx = indices[0][i]
        if idx < len(data):
            results.append({
                'source': data[idx]['source'],
                'chunk_id': data[idx]['chunk_id'],
                'text': data[idx]['text'],
                'distance': distances[0][i]
            })
    return results

def chunk_text(text, chunk_size=200):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


if __name__ == '__main__':
    transcripts_dir = 'transcripts'
    index_path = 'index.faiss'
    metadata_path = 'metadata.pkl'

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        print("Building FAISS index from transcripts...")
        data = load_transcripts(transcripts_dir)
        if not data:
            print("No transcripts found. Add .txt files to 'transcripts/' and rerun.")
            exit(1)
        embeddings = generate_embeddings(data)
        index = build_index(embeddings)
        save_index_and_metadata(index, data, index_path, metadata_path)
        print("Index built.")
    else:
        print("Loading existing FAISS index...")
        index, data = load_index_and_metadata(index_path, metadata_path)

    print("\nReady for queries. Enter text to search semantically across transcripts.")
    while True:
        query = input("\nQuery (or 'quit'): ").strip()
        if query.lower() == 'quit':
            break
        if not query:
            continue
        results = semantic_search(query, index, data)
        if not results:
            print("No results found.")
            continue
        print("\nTop results:")
        for res in results:
            print(f"\nSource: {res['source']} (Chunk {res['chunk_id']}) | Distance: {res['distance']:.4f}")
            print(f"Text: {res['text']}")