import json
from transformers import AutoTokenizer

# Step 1: Load LLaMA Tokenizer
model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained("facebook/llama-tokenizer")
# Step 2: Function to load and process JSON data
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Step 3: Extract and clean the text (assuming the JSON has a 'text' field)
def extract_text(data):
    if isinstance(data, list):
        return [item.get('text', '') for item in data]  # Extract 'text' field from each item
    elif isinstance(data, dict):
        return [data.get('text', '')]
    else:
        return []

# Step 4: Tokenize the text using LLaMA tokenizer
def tokenize_texts(texts):
    tokenized_data = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    return tokenized_data['input_ids']  # Returns tokenized ids

# Step 5: Save tokenized data
def save_tokenized_data(tokenized_data, output_file):
    with open(output_file, 'w') as f:
        json.dump(tokenized_data.tolist(), f)  # Convert tensor to list before saving

# Main function to process JSON and tokenize
def process_json_to_tokens(json_file, output_file):
    # Load JSON
    data = load_json(json_file)
    
    # Extract text from JSON
    texts = extract_text(data)
    
    # Tokenize the text
    tokenized_data = tokenize_texts(texts)
    
    # Save the tokenized output
    save_tokenized_data(tokenized_data, output_file)
    print(f"Tokenized data saved to {output_file}")

# Example Usage
json_file_path = 'converted_data_with_tags.json'  # Path to your input JSON file
output_token_file = 'tokenized_output.json'  # Path to save the tokenized data
process_json_to_tokens(json_file_path, output_token_file)
