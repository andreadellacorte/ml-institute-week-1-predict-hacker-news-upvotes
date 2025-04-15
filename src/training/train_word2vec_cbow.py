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

# tokenise the text; if the token is not in token_to_index, replace it with the <UNK> token
def tokenise_text(text, token_to_index):
    print("Tokenising text...")
    tokens = text.split()
    tokenised_text = []
    for token in tokens:
        if token in token_to_index:
            tokenised_text.append(token_to_index[token])
        else:
            tokenised_text.append(token_to_index['<UNK>'])
    return tokenised_text

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
    def __init__(self, tokenised_text, context_size):
        self.tokenised_text = tokenised_text
        self.context_size = context_size
        self.data = self.create_cbow_data()

    def create_cbow_data(self):
        data = []
        for i in range(self.context_size, len(self.tokenised_text) - self.context_size):
            context = self.tokenised_text[i - self.context_size:i] + self.tokenised_text[i + 1:i + self.context_size + 1]
            target = self.tokenised_text[i]
            if len(context) == 2 * self.context_size and target is not None:
                data.append((context, target))
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

    # 3. No need for pre-processing as the dataset is already preprocessed

    # 4. Prepare the dataset for CBOW
    
    tokenised_text = tokenise_text(text, token_to_index)
    print(f"Text tokenised with {len(tokenised_text)} tokens.")
    unique_tokens = Counter(tokenised_text)
    print(f"Number of unique tokens in tokenised text: {len(unique_tokens)}")

    # save tokenised text to file
    tokenised_text_filepath = 'data/processed/tokenised_text.txt'
    with open(tokenised_text_filepath, 'w') as file:
        for token in tokenised_text:
            file.write(f"{token} ")
    print(f"Tokenised text saved to {tokenised_text_filepath}.")

    print("Preparing CBOW dataset...")
    context_size = 2
    cbow_dataset = CBOWDataset(tokenised_text, context_size)
    dataloader = DataLoader(cbow_dataset, batch_size=256, shuffle=True, num_workers=4)

    # 5. Initialize the model, loss function, and optimizer
    print("Initializing model...")
    vocab_size = len(token_to_index)
    embedding_dim = 100

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Word2VecCBOW(vocab_size, embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 6. Train the model
    print("Training model...")
    num_epochs = 5
    for epoch in range(num_epochs):
        epoch_loss = 0
        start_epoch_time = time.time()
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for context, target in dataloader:
                start_sample_time = time.time()
                context, target = context.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(context)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix(loss=loss.item(), sample_time=f"{time.time() - start_sample_time:.4f}s")
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Time: {time.time() - start_epoch_time:.2f} seconds")

    # 7. Save the trained model
    print("Saving model...")
    torch.save(model.state_dict(), 'data/models/word2vec_cbow.pth')
    print("Model saved.")
