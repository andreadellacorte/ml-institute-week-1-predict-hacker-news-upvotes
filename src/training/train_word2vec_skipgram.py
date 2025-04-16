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

# Define the Word2Vec Skip-gram model
class Word2VecSkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2VecSkipGram, self).__init__()
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, target, context):
        target_embeds = self.input_embeddings(target)
        context_embeds = self.output_embeddings(context)
        scores = torch.sum(target_embeds * context_embeds, dim=1)
        return scores

# Define a custom dataset for Skip-gram
class SkipGramDataset(Dataset):
    def __init__(self, text, context_size, token_to_index):
        self.text = text
        self.context_size = context_size
        self.token_to_index = token_to_index
        self.unk = token_to_index.get("<UNK>", -1)  # Default to -1 if "<UNK>" is not found
        if self.unk == -1:
            raise ValueError("The '<UNK>' token is missing from the token_to_index mapping.")
        if len(self.text) < 2 * self.context_size + 1:
            raise ValueError(f"Dataset is too short for the given context size ({self.context_size}). "
                             f"Minimum required length is {2 * self.context_size + 1}, but got {len(self.text)}.")
        self.data = self.create_skipgram_data()

    def create_skipgram_data(self):
        with tqdm(total=len(self.text), desc="Processing Skip-gram data") as pbar:
            for i in range(self.context_size, len(self.text) - self.context_size):
                target = self.token_to_index.get(self.text[i], self.unk)
                context = [
                    self.token_to_index.get(self.text[j], self.unk)
                    for j in range(i - self.context_size, i + self.context_size + 1)
                    if j != i
                ]
                for ctx in context:
                    if target is not None and ctx is not None:
                        data.append((target, ctx))
                if i % 100 == 0:  # Update progress bar every 100 iterations
                    pbar.update(100)
                    pbar.update(len(self.text) % 100)  # Update remaining iterations
        print("Skip-gram data creation complete.")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target, context = self.data[idx]
        return torch.tensor(target, dtype=torch.long), torch.tensor(context, dtype=torch.long)

if __name__ == '__main__':
    # 1. Load tokens
    print("Loading tokens...")
    token_to_index_filepath = 'data/processed/token_to_index.csv'
    index_to_token_filepath = 'data/processed/index_to_token.csv'
    token_to_index, index_to_token = load_tokeniser(token_to_index_filepath, index_to_token_filepath)

    # 2. Load the dataset
    print("Loading dataset...")
    dataset = load_dataset("afmck/text8")
    text = dataset['train'][0]['text']

    print("Preparing Skip-gram dataset...")
    context_size = 2
    skipgram_dataset = SkipGramDataset(text.split(), context_size, token_to_index)

    num_workers = min(4, multiprocessing.cpu_count())  # Use up to 4 workers or the number of CPU cores available
    dataloader = DataLoader(skipgram_dataset, batch_size=64, shuffle=True, num_workers=num_workers, pin_memory=True)

    # 3. Initialize the model, loss function, and optimizer
    print("Initializing model...")
    vocab_size = len(token_to_index)
    embedding_dim = 200

    # 4. Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = Word2VecSkipGram(vocab_size, embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Train the model
    print("Training model...")
    num_epochs = 5
    scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None  # Use mixed precision only if GPU is available
    for epoch in range(num_epochs):
        epoch_loss = 0
        start_epoch_time = time.time()
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for target, context in dataloader:
                start_sample_time = time.time()
                target, context = target.to(device, non_blocking=True), context.to(device, non_blocking=True)
                context = context.view(-1)  # Ensure context is a 1D tensor of target indices
                optimizer.zero_grad()
                if scaler:  # Mixed precision training
                    with torch.amp.autocast():
                        scores = model(target, context)
                        loss = -torch.mean(torch.log(torch.sigmoid(scores)))
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:  # Standard training
                    output = model(target)
                    loss = criterion(output, context)
                    loss.backward()
                    optimizer.step()
                epoch_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix(loss=loss.item(), sample_time=f"{(time.time() - start_sample_time) * 1000:.2f}ms")
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Avg Loss: {avg_loss:.4f}, Time: {time.time() - start_epoch_time:.2f} seconds")

    # 6. Save the trained model and optimizer state
    print("Saving model and optimizer state...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, 'data/models/word2vec_skipgram.pth')
    print("Model and optimizer state saved.")