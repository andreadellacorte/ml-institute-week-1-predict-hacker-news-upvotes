import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from src.training.train_word2vec_cbow import Word2VecCBOW
from src.training.create_tokeniser import load_tokeniser

def test_word2vec_model(model_path, token_to_index_path, index_to_token_path):
    # Load the tokeniser
    token_to_index, index_to_token = load_tokeniser(index_to_token_path, token_to_index_path)

    # Load the model
    vocab_size = len(token_to_index)
    embedding_dim = 100  # Ensure this matches the training configuration
    model = Word2VecCBOW(vocab_size, embedding_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    context_words = 'i want a glass of'.split()

    # Test the model with a sample context
    sample_context = [token_to_index[token] for token in context_words if token in token_to_index]
    if len(sample_context) < len(context_words):
        print("Sample context tokens are not in the vocabulary.")
        return

    context_tensor = torch.tensor([sample_context], dtype=torch.long)
    with torch.no_grad():
        output = model(context_tensor)
        predicted_index = torch.argmax(output, dim=1).item()
        # Handle unknown indices by mapping to <UNK>
        predicted_token = index_to_token.get(predicted_index)

    print(f"Sample context: {sample_context}")
    print(f"Predicted token: {predicted_token}")

if __name__ == "__main__":
    model_path = "data/models/word2vec_cbow.pth"
    token_to_index_path = "data/processed/token_to_index.csv"
    index_to_token_path = "data/processed/index_to_token.csv"

    test_word2vec_model(model_path, token_to_index_path, index_to_token_path)