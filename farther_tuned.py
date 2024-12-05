import json
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

# Load tokenizer and existing fine-tuned model (or a pre-trained checkpoint)
fine_tuned_model_dir = "./fine-tuned-llama2"  # Directory where the fine-tuned model is saved

tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    fine_tuned_model_dir,  # Load the fine-tuned model or checkpoint
    device_map="auto"
)

# Set pad_token if not defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Apply LoRA (if further LoRA fine-tuning is needed)
peft_config = LoraConfig(
    r=8,  # Rank of LoRA decomposition
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, peft_config)

# Load dataset from a JSON file
with open('converted_data_with_tags.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

# Tokenization function
def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

# Convert JSON data to tokenized format
tokenized_data = [tokenize_function(example) for example in json_data]

# Split data into train and test sets (assuming a 80-20 split)
train_size = int(0.8 * len(tokenized_data))
train_dataset = tokenized_data[:train_size]
eval_dataset = tokenized_data[train_size:]

# Define training arguments for resuming from a checkpoint
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,
    optim="adamw_torch_fused",
    report_to="none",
    save_strategy="steps",
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    save_total_limit=3,
    # Resume training from the last checkpoint
    resume_from_checkpoint=fine_tuned_model_dir  # Path to your checkpoint
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Start/resume training
trainer.train()

# Save the further fine-tuned model and tokenizer
model.save_pretrained("./further-fine-tuned-llama2")
tokenizer.save_pretrained("./further-fine-tuned-llama2")
