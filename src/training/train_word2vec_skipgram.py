import os
import csv
import requests
import multiprocessing
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm
import wandb

embedding_dim = 200
batch_size = 128  # Adjusted batch size to balance speed and memory usage
num_epochs = 1
context_size = 2
pin_memory = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 8
dataset = "afmck/text8"
model = "Word2Vec Skip-gram"
learning_rate = 0.01  # Slightly increased learning rate to match larger batch size

ntfy_topic = "mlx7-institute-dellacorte"

# Enable CUDA_LAUNCH_BLOCKING for debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Start a new wandb run to track this script.
run = wandb.init(
    project="word2vec_skipgram",
    entity="dellacorte-me",  # Replace with your wandb entity
    config={
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "device": device,
        "embedding_dim": embedding_dim,
        "context_size": context_size,
        "dataset": dataset,
        "model": model
    },
    name=f"{model}_{dataset}_bs{batch_size}_lr{learning_rate}_epochs{num_epochs}",
)

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
        self.data = self.create_skipgram_data()

    def create_skipgram_data(self):
        data = []
        print("Creating Skip-gram data...")
        with tqdm(total=len(self.text), desc="Processing Skip-gram data") as pbar:
            for i in range(self.context_size, len(self.text) - self.context_size):
                target = self.token_to_index.get(self.text[i], self.unk)
                context = [
                    self.token_to_index.get(self.text[j], self.unk)
                    for j in range(i - self.context_size, i + self.context_size + 1)
                    if j != i
                ]
                # Validate indices
                if target < 0 or target >= len(self.token_to_index):
                    continue
                context = [ctx for ctx in context if 0 <= ctx < len(self.token_to_index)]
                for ctx in context:
                    data.append((target, ctx))
                if i % 1000 == 0:  # Update progress bar every 100 iterations
                    pbar.update(1000)
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
    dataset = load_dataset(dataset)
    text = dataset['train'][0]['text']

    print("Preparing Skip-gram dataset...")
    skipgram_dataset = SkipGramDataset(text.split(), context_size, token_to_index)

    num_workers = min(num_workers, multiprocessing.cpu_count())  # Use multiple workers for data loading
      # Enable pinned memory for faster data transfer to GPU
    dataloader = DataLoader(skipgram_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    # 3. Initialize the model, loss function, and optimizer
    print("Initializing model...")
    vocab_size = len(token_to_index)
    
    # 4. Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = Word2VecSkipGram(vocab_size, embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    wandb.watch(model, log="all")

    # 5. Train the model
    print("Training model...")
    scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None  # Ensure mixed precision training is enabled
    for epoch in range(num_epochs):
        epoch_loss = 0
        start_epoch_time = time.time()
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for batch_idx, (target, context) in enumerate(dataloader):
                batch_start_time = time.time()
                target, context = target.to(device, non_blocking=True), context.to(device, non_blocking=True)
                context = context.view(-1)  # Ensure context is a 1D tensor of target indices

                # Debugging: Check for out-of-bounds indices
                if torch.any(target < 0) or torch.any(target >= vocab_size):
                    print("Error: Out-of-bounds target indices detected.")
                    continue
                if torch.any(context < 0) or torch.any(context >= vocab_size):
                    print("Error: Out-of-bounds context indices detected.")
                    continue

                optimizer.zero_grad()
                if scaler:  # Mixed precision training
                    with torch.amp.autocast(device_type=device.type):
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
                pbar.set_postfix(
                    loss=loss.item(),
                    batch_time=f"{(time.time() - batch_start_time) * 1000:.2f}ms"
                )

                run.log({"loss": loss.item()})
        avg_loss = epoch_loss / len(dataloader)

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Avg Loss: {avg_loss:.4f}, Time: {time.time() - start_epoch_time:.2f} seconds")
        run.log({"epoch": epoch + 1, "loss": avg_loss})

        epoch_model_filepath = f'data/models/word2vec_skipgram_epoch_{epoch + 1}.pth'

        # Save the model and optimizer state after every epoch
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, epoch_model_filepath)

        wandb.save(os.path.join(wandb.run.dir, epoch_model_filepath))

        print(f"Model and optimizer state saved for epoch {epoch + 1}.")

        requests.post("https://ntfy.sh/" + ntfy_topic,
                      data="Completed {epoch} / {num_epochs}".encode(encoding='utf-8'))

        # First the 10 most similar words for an example dictionary

        print("Calculating cosine similarity for example words...")

        example_dict = {
            "word1": "hello",
            "word2": "banana",
            "word3": "apple",
            "word4": "car",
            "word5": "dog",
            "word6": "cat",
            "word7": "hacker",
            "word8": "computer",
            "word9": "science",
            "word10": "technology"
        }

        for word, example_word in example_dict.items():
            if word in token_to_index:
                word_index = token_to_index[word]
                example_word_index = token_to_index[example_word]

                # Get the embedding for the word
                word_embedding = model.input_embeddings(torch.tensor(word_index).to(device)).detach().cpu().numpy()
                example_word_embedding = model.input_embeddings(torch.tensor(example_word_index).to(device)).detach().cpu().numpy()

                # Validate tensor shapes
                if word_embedding.shape != example_word_embedding.shape:
                    print(f"Error: Shape mismatch for {word} and {example_word} embeddings.")
                    continue

                # Calculate cosine similarity
                similarity = torch.nn.functional.cosine_similarity(
                    torch.tensor(word_embedding),
                    torch.tensor(example_word_embedding),
                    dim=0
                )

                print(f"Cosine similarity between {word} and {example_word}: {similarity.item()}")

        # Guess the next word for some phrases

        phrases = [
            "I love to",
            "The sky is",
            "I enjoy",
            "The sun is",
            "I like to read"
        ]

        for phrase in phrases:
            phrase_tokens = phrase.split()
            phrase_indices = [token_to_index.get(token, -1) for token in phrase_tokens]
            phrase_tensor = torch.tensor(phrase_indices).to(device)

            # Get the embedding for the phrase
            phrase_embedding = model.input_embeddings(phrase_tensor).mean(dim=0, keepdim=True)

            if phrase_embedding.shape[1] != model.input_embeddings.weight.shape[1]:
                print("Error: Shape mismatch between phrase embedding and model embeddings.")
                continue

            # Get the top 5 most similar words
            similarities = torch.nn.functional.cosine_similarity(
                model.input_embeddings.weight,
                phrase_embedding,
                dim=1
            )
            top_5_indices = similarities.topk(5).indices

            print(f"Top 5 words for '{phrase}': {[index_to_token[i.item()] for i in top_5_indices]}")

    final_model_filepath = 'data/models/word2vec_skipgram_final.pth'

    requests.post("https://ntfy.sh/" + ntfy_topic,
                  data="Completed training".encode(encoding='utf-8'))

    # 6. Save the final trained model and optimizer state
    print("Saving final model and optimizer state...")
    torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
    }, final_model_filepath)
    wandb.save(os.path.join(wandb.run.dir, final_model_filepath))
    print("Final model and optimizer state saved.")

    run.finish()