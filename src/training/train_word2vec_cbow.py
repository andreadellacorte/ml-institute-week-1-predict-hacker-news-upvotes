import csv
import multiprocessing
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm

# Load the token to index mapping from a CSV file
def load_tokeniser(token_to_index_filepath, index_to_token_filepath):
    print("Loading tokeniser...")
    token_to_index = {}
    with open(token_to_index_filepath, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            token, index = row
            token_to_index[token] = int(index)

    index_to_token = {}
    with open(index_to_token_filepath, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            index, token = row
            index_to_token[int(index)] = token

    print(f"Tokeniser loaded with {len(token_to_index)} tokens.")
    return token_to_index, index_to_token

def load_tokenised_text(filepath):
    print("Loading tokenised text...")
    with open(filepath, 'r') as file:
        tokenised_text = [int(token) for token in file.read().split()]
    print(f"Tokenised text loaded with {len(tokenised_text)} tokens.")
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
    def __init__(self, text, context_size, token_to_index):
        self.text = text
        self.context_size = context_size
        self.token_to_index = token_to_index
        self.unk = token_to_index.get("<UNK>")
        self.data = self.create_cbow_data()

    def create_cbow_data(self):
        data = []
        print("Creating CBOW data...")
        i = 0
        with tqdm(total=len(self.text), desc="Processing CBOW data") as pbar:
            for i in range(self.context_size, len(self.text) - self.context_size):
                context = [self.token_to_index.get(token, self.unk) for token in self.text[i - self.context_size:i] + self.text[i + 1:i + self.context_size + 1]]
                target = self.token_to_index.get(self.text[i], self.unk)
                if len(context) == 2 * self.context_size and target is not None:
                    data.append((context, target))
                if i % 1000 == 0:
                    pbar.update(1000)
                    pbar.update(len(self.text) % 100)  # Update remaining iterations
        print("CBOW data creation complete.")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)

if __name__ == '__main__':
    # 1. Load tokens
    print("Loading tokens...")
    token_to_index_filepath = 'data/processed/token_to_index.csv'
    index_to_token_filepath = 'data/processed/index_to_token.csv'
    token_to_index, index_to_token = load_tokeniser(token_to_index_filepath, index_to_token_filepath)
    
    # 1. Load the dataset
    print("Loading dataset...")
    dataset = load_dataset("afmck/text8")
    text = dataset['train'][0]['text']

    print("Preparing CBOW dataset...")
    context_size = 2
    cbow_dataset = CBOWDataset(text.split(), context_size, token_to_index)
    
    num_workers = min(4, multiprocessing.cpu_count())  # Use up to 4 workers or the number of CPU cores available
    dataloader = DataLoader(cbow_dataset, batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=True)

    # 3. Initialize the model, loss function, and optimizer
    print("Initializing model...")
    vocab_size = len(token_to_index)
    embedding_dim = 200

    # 4. Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = Word2VecCBOW(vocab_size, embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Train the model
    print("Training model...")
    num_epochs = 5
    scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None  # Use mixed precision only if GPU is available
    for epoch in range(num_epochs):
        epoch_loss = 0
        i = 0
        start_epoch_time = time.time()
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for context, target in dataloader:
                start_sample_time = time.time()
                context, target = context.to(device, non_blocking=True), target.to(device, non_blocking=True)
                optimizer.zero_grad()
                if scaler:  # Mixed precision training
                    with torch.amp.autocast():
                        output = model(context)
                        loss = criterion(output, target)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:  # Standard training
                    output = model(context)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                epoch_loss += loss.item()
                if i % 1000 == 0:
                    pbar.update(1000)
                    pbar.set_postfix(loss=loss.item(), sample_time=f"{(time.time() - start_sample_time) * 1000:.2f}ms")
                i += 1
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Avg Loss: {avg_loss:.4f}, Time: {time.time() - start_epoch_time:.2f} seconds")

    # 6. Save the trained model and optimizer state
    print("Saving model and optimizer state...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, 'data/models/word2vec_cbow.pth')
    print("Model and optimizer state saved.")