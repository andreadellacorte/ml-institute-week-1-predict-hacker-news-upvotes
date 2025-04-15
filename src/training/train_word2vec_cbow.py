import csv
from datasets import load_dataset
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm

# Load the token to index mapping from a CSV file
def load_tokeniser(index_to_token_filepath, token_to_index_filepath):
    print("Loading tokeniser...")
    token_to_index = {}
    with open(index_to_token_filepath, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            token, index = row
            token_to_index[token] = int(index)

    index_to_token = {}
    with open(token_to_index_filepath, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            index, token = row
            index_to_token[index] = token

    print(f"Tokeniser loaded with {len(token_to_index)} tokens.")
    return token_to_index, index_to_token

# Define the Word2Vec CBOW model
class Word2VecCBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2VecCBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context):
        embedded = self.embeddings(context).mean(dim=1)
        out = self.linear(embedded)
        return out

# Define a custom dataset for CBOW
class CBOWDataset(Dataset):
    def __init__(self, text, token_to_index, context_size):
        self.text = text
        self.token_to_index = token_to_index
        self.context_size = context_size
        self.data = self.create_cbow_data()

    def create_cbow_data(self):
        data = []
        for i in range(self.context_size, len(self.text) - self.context_size):
            context = self.text[i - self.context_size:i] + self.text[i + 1:i + self.context_size + 1]
            target = self.text[i]
            context_indices = [self.token_to_index[token] for token in context if token in self.token_to_index]
            target_index = self.token_to_index.get(target, None)
            if len(context_indices) == 2 * self.context_size and target_index is not None:
                data.append((context_indices, target_index))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0]), torch.tensor(self.data[idx][1])

if __name__ == '__main__':
    # 1. Load tokens
    print("Loading tokens...")
    token_to_index_filepath = 'data/processed/token_to_index.csv'
    index_to_token_filepath = 'data/processed/index_to_token.csv'
    token_to_index, index_to_token = load_tokeniser(token_to_index_filepath, index_to_token_filepath)

    # print some information on the above variables
    print(f"Token to index mapping: {list(token_to_index.items())[:10]}")
    print(f"Index to token mapping: {list(index_to_token.items())[:10]}")

    # 2. Load the dataset
    print("Loading dataset...")
    dataset = load_dataset("afmck/text8")
    text = dataset['train'][0]['text']

    print(len(text.split()))

    # 3. No need for pre-processing as the dataset is already preprocessed

    # 4. Prepare the dataset for CBOW
    print("Preparing CBOW dataset...")
    context_size = 2
    tokenized_text = [token for token in text.split() if token in token_to_index]
    cbow_dataset = CBOWDataset(tokenized_text, token_to_index, context_size)
    dataloader = DataLoader(cbow_dataset, batch_size=64, shuffle=True)

    # 5. Initialize the model, loss function, and optimizer
    print("Initializing model...")
    vocab_size = len(token_to_index)
    embedding_dim = 100
    model = Word2VecCBOW(vocab_size, embedding_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 6. Train the model
    print("Training model...")
    num_epochs = 5
    for epoch in range(num_epochs):
        epoch_loss = 0
        for context, target in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

    # 7. Save the trained model
    print("Saving model...")
    torch.save(model.state_dict(), 'data/models/word2vec_cbow.pth')
    print("Model saved.")
