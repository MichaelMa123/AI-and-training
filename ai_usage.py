import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Path to the fine-tuned model directory
fine_tuned_model_dir = "./fine-tuned-llama"  # Update this with your actual path

# Load the tokenizer and model from the local fine-tuned directory
tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_dir)
model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_dir, torch_dtype=torch.bfloat16, device_map="auto")

# Create a text-generation pipeline using the fine-tuned model
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Manually format the chat messages as a string
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "explain to me how to solve x^2=4"},
]

# Combine messages into a single prompt string
formatted_input = f"System: {messages[0]['content']}\nUser: {messages[1]['content']}\nPirateBot:"

# Generate a response using the fine-tuned model
outputs = pipe(
    formatted_input,
    max_new_tokens=256,
)

# Print the generated response
print(outputs[0]["generated_text"])
