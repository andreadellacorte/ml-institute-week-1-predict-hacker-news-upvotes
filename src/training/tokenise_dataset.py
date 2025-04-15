import csv
from collections import Counter
from datasets import load_dataset

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

if __name__ == '__main__':
    # 1. Load the dataset
    print("Loading dataset...")
    dataset = load_dataset("afmck/text8")
    text = dataset['train'][0]['text']

    # 2. Load tokeniser
    print("Loading tokens...")
    token_to_index_filepath = 'data/processed/token_to_index.csv'
    index_to_token_filepath = 'data/processed/index_to_token.csv'
    token_to_index, index_to_token = load_tokeniser(token_to_index_filepath, index_to_token_filepath)

    # 3. Tokenise the dataset
    tokenised_text = tokenise_text(text, token_to_index)
    print(f"Text tokenised with {len(tokenised_text)} tokens.")
    unique_tokens = Counter(tokenised_text)
    print(f"Number of unique tokens in tokenised text: {len(unique_tokens)}")

    # 4. Save tokenised text to file
    tokenised_text_filepath = 'data/processed/tokenised_text.txt'
    with open(tokenised_text_filepath, 'w') as file:
        for token in tokenised_text:
            file.write(f"{token} ")
    print(f"Tokenised text saved to {tokenised_text_filepath}")