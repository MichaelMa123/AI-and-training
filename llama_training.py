from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import LoraConfig, PeftModel
import torch
import torch
from transformers import Trainer, TrainingArguments
model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,low_cpu_mem_usage=True,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map='auto')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Safe multiprocessing guard
if __name__ == "__main__":
    # Load the dataset
    dataset = load_dataset("lighteval/MATH", "all")

    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(examples['problem'], padding="max_length", truncation=True)

    # Tokenize dataset with multiprocessing
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        learning_rate=5e-5,
        logging_dir="./logs",
        logging_steps=10,
        fp16=True  # Enable mixed precision
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test']
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model and tokenizer
    model.save_pretrained("./fine-tuned-llama")
    tokenizer.save_pretrained("./fine-tuned-llama")
