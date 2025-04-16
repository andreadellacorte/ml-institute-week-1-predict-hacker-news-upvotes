from datasets import load_dataset

def tokenize_text(text):
    """
    Tokenizes the input text into a list of words and their frequency.

    Args:
        text (str): The input text to tokenize.

    Returns:
        list: A dictionary of lowercase words with punctuation removed, mapped
        to their frequency
    """

    # Remove punctuation, split into words, and convert to lowercase
    tokens = text.split()
    
    # Create a dictionary with word frequencies
    word_freq = {}
    for token in tokens:
        if token in word_freq:
            word_freq[token] += 1
        else:
            word_freq[token] = 1
    
    return word_freq

if __name__ == "__main__":
    # 1. Load the dataset
    print("Loading dataset...")
    dataset = load_dataset("afmck/text8")

    # 2. Preprocess the dataset
    print("Preprocessing dataset...")
    text = dataset['train'][0]['text']

    # 3. Tokenize the dataset
    print("Tokenising dataset...")
    tokens = tokenize_text(text)

    # 4. Print some data on the tokens
    print(f"Number of tokens: {len(tokens)}")
    print(f"First 10 tokens: {list(tokens.items())[:10]}")
    print(f"Number of unique tokens: {len(set(tokens))}")
    print(f"Most common token: {max(tokens, key=tokens.get)}")
    print(f"Frequency of most common token: {tokens[max(tokens, key=tokens.get)]}")

    # 5. Set a treshold below which tokens are removed from the tokens
    threshold = 5

    filtered_tokens = {word: freq for word, freq in tokens.items() if freq >= threshold}
    print(f"Number of tokens after filtering: {len(filtered_tokens)}")
    print(f"First 10 filtered tokens: {list(filtered_tokens.items())[:10]}")
    print(f"Number of unique filtered tokens: {len(set(filtered_tokens))}")

    # 6. Assign an index to each token
    token_to_index = {token: index for index, token in enumerate(filtered_tokens.keys())}
    print(f"Token to index mapping: {list(token_to_index.items())[:10]}")
    index_to_token = {index: token for token, index in token_to_index.items()}
    print(f"Index to token mapping: {list(index_to_token.items())[:10]}")

    # 7. Add to the tokeniser <UNK> for unknown tokens
    token_to_index['<UNK>'] = -1
    index_to_token[-1] = '<UNK>'

    # 8. Save filtered tokens to a data file
    output_file_path = "data/processed/token_to_index.csv"
    with open(output_file_path, 'w') as file:
        for token, index in token_to_index.items():
            file.write(f"{token},{index}\n")
    print(f"Filtered tokens saved to {output_file_path}.")

    # 8. Save filtered indexes to a data file
    output_file_path = "data/processed/index_to_token.csv"
    with open(output_file_path, 'w') as file:
        for index, token in index_to_token.items():
            file.write(f"{index},{token}\n")
    print(f"Index to token mapping saved to {output_file_path}.")