import os
import numpy as np
from gensim.models import Word2Vec

# Load and use the model
model = Word2Vec.load("models/word2vec_text8_cbow.model")

# Get vector for a word
print(model.wv['king'])

# Find similar words
print(model.wv.most_similar('king'))
print(model.wv.most_similar('queen'))

def sentence_to_vec(sentence, model):
    words = sentence.lower().split()
    valid_words = [word for word in words if word in model.wv]

    if not valid_words:
        return np.zeros(model.vector_size)  # fallback if no known words

    vectors = [model.wv[word] for word in valid_words]
    return np.mean(vectors, axis=0)  # shape: (vector_size,)

vector = sentence_to_vec("Hello my name is Paul Graham", model)

print(vector)

print(f"vector size {vector.shape}")