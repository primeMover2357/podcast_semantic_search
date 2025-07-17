import os
import glob
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import re

# Load the pre-trained embedding model for semantic understanding
model = SentenceTransformer('all-MiniLM-L6-v2')

def seconds_to_hms(seconds):
    """Convert seconds to HH:MM:SS format."""
    try:
        seconds = float(seconds)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    except (ValueError, TypeError):
        return "N/A"

def load_transcripts(directory):
    data = []
    for filepath in glob.glob(os.path.join(directory, '*.txt')):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            # Split into chunks (by double newlines)
            chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
            for i, chunk in enumerate(chunks):
                # Extract timestamp in [HH:MM:SS] or [seconds] format
                match = re.match(r'\[(\d{2}:\d{2}:\d{2}|\d+\.\d+)\]\s*(.*)', chunk)
                timestamp = match.group(1) if match else 'N/A'
                chunk_text = match.group(2) if match else chunk
                # Convert HH:MM:SS to seconds for internal storage, keep as string for display
                if ':' in timestamp:
                    h, m, s = map(int, timestamp.split(':'))
                    timestamp_seconds = h * 3600 + m * 60 + s
                else:
                    timestamp_seconds = timestamp
                data.append({
                    'source': os.path.basename(filepath),
                    'chunk_id': i,
                    'text': chunk_text,
                    'timestamp': timestamp_seconds  # Store as seconds or 'N/A'
                })
    print(f"Loaded {len(data)} chunks from {len(glob.glob(os.path.join(directory, '*.txt')))} files:")
    for item in data[:5]:  # Log first few for verification
        print(f"Source: {item['source']}, Chunk {item['chunk_id']}, Timestamp: {seconds_to_hms(item['timestamp'])}, Text: {item['text'][:50]}...")
    return data

def generate_embeddings(data):
    texts = [item['text'] for item in data]
    embeddings = model.encode(texts)
    return embeddings

def build_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
    index.add(np.array(embeddings).astype('float32'))
    return index

def save_index_and_metadata(index, data, index_path='index.faiss', metadata_path='metadata.pkl'):
    faiss.write_index(index, index_path)
    with open(metadata_path, 'wb') as f:
        pickle.dump(data, f)

def load_index_and_metadata(index_path='index.faiss', metadata_path='metadata.pkl'):
    index = faiss.read_index(index_path)
    with open(metadata_path, 'rb') as f:
        data = pickle.load(f)
    return index, data

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
                'timestamp': seconds_to_hms(data[idx]['timestamp']),
                'distance': distances[0][i]
            })
    return results

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
            print(f"\nSource: {res['source']} (Chunk {res['chunk_id']}) | Timestamp: {res['timestamp']} | Distance: {res['distance']:.4f}")
            print(f"Text: {res['text']}")