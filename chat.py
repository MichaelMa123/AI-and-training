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

# Loop to continuously get user input and generate responses
print("Welcome to PirateBot! Type 'exit' to terminate the program.")
while True:
    # Get user input from the terminal
    user_input = input("User: ")

    # Check if the user wants to exit
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # Manually format the chat messages as a string
    formatted_input = f"System: You are a pirate chatbot who always responds in pirate speak!\nUser: {user_input}\nPirateBot:"

    # Generate a response using the fine-tuned model
    outputs = pipe(
        formatted_input,
        max_new_tokens=256,
    )

    # Print the generated response
    print(outputs[0]["generated_text"])
