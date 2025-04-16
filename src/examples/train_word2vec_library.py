from datasets import load_dataset
from gensim.models import Word2Vec
import multiprocessing
import time

# 1. Load the dataset
print("Loading dataset...")
dataset = load_dataset("afmck/text8")

# 2. Preprocess: split into words, chunk into pseudo-sentences
print("Preprocessing dataset...")
text = dataset['train'][0]['text']
tokens = text.split()
chunk_size = 1000
sentences = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
print(f"Created {len(sentences)} pseudo-sentences.")

# 3. Train the Word2Vec models

# Train CBOW model
print("Training CBOW model...")
start_time = time.time()
cbow_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=5,
    workers=multiprocessing.cpu_count(),
    sg=0  # CBOW
)
cbow_elapsed = time.time() - start_time
print(f"CBOW training completed in {cbow_elapsed:.2f} seconds.")

# Save CBOW model
cbow_model.save("models/word2vec_text8_cbow.model")
print("CBOW model saved to 'word2vec_text8_cbow.model'.")

# Train Skip-gram model
print("Training Skip-gram model...")
start_time = time.time()
skipgram_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=5,
    workers=multiprocessing.cpu_count(),
    sg=1  # Skip-gram
)
skipgram_elapsed = time.time() - start_time
print(f"Skip-gram training completed in {skipgram_elapsed:.2f} seconds.")

# Save Skip-gram model
skipgram_model.save("models/word2vec_text8_skipgram.model")
print("Skip-gram model saved to 'word2vec_text8_skipgram.model'.")
