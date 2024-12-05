import json
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

# Load tokenizer and model in 4-bit precision with BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Load model in 4-bit precision
    bnb_4bit_use_double_quant=True,  # Optional: double quantization for extra compression
    bnb_4bit_quant_type='nf4'  # Quantization type for better performance
)

model_name = "meta-llama/Llama-3.2-3B"

# Load tokenizer and model with quantization settings
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"  # Automatically maps model across available devices (GPU/CPU)
)

# Set pad_token if not defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Apply PEFT (LoRA) for parameter-efficient fine-tuning
peft_config = LoraConfig(
    r=8,  # Rank of LoRA decomposition
    lora_alpha=16,  # Scaling factor for LoRA
    target_modules=["q_proj", "v_proj"],  # LoRA for attention layers
    lora_dropout=0.1,  # Regularization for LoRA
    bias="none"  # No bias in LoRA layers
)
model = get_peft_model(model, peft_config)

# Load dataset from a JSON file
with open('converted_data_with_tags.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

# Tokenization function (including labels for training)
def tokenize_function(examples):
    # Tokenize the 'text' field and set the labels as the input_ids (shifted for causal language modeling)
    tokenized_inputs = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)
    
    # The labels are the input_ids copied for causal language modeling
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    
    return tokenized_inputs

# Convert JSON data to tokenized format
tokenized_data = [tokenize_function(example) for example in json_data]

# Split data into train and test sets (assuming a 80-20 split)
train_size = int(0.8 * len(tokenized_data))
train_dataset = tokenized_data[:train_size]
eval_dataset = tokenized_data[train_size:]

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",  # Directory for saving model and logs
    per_device_train_batch_size=1,  # Small batch size to fit in 24GB GPU memory
    gradient_accumulation_steps=16,  # Accumulate gradients to simulate larger effective batch size
    num_train_epochs=3,  # Number of training epochs
    logging_dir="./logs",  # Directory for logging
    logging_steps=10,  # Log every 10 steps
    fp16=True,  # Enable mixed precision to reduce memory
    optim="adamw_torch_fused",  # Use a more memory-efficient optimizer
    report_to="none",  # Disable reporting to W&B or similar services
    save_strategy="steps",  # Save model at intervals
    save_steps=500,  # Save every 500 steps
    evaluation_strategy="steps",  # Evaluate during training at intervals
    eval_steps=500,  # Evaluate every 500 steps
    save_total_limit=3,  # Only keep the last 3 checkpoints to save disk space
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Start training
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./fine-tuned-llama2")
tokenizer.save_pretrained("./fine-tuned-llama2")
