from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import numpy as np
import evaluate
import torch
from torch import nn

# Load Yelp dataset
dataset = load_dataset("yelp_review_full")
print(dataset["train"][100])  # Optional: Preview one sample from the dataset

# Load pre-trained LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained("decapoda-research/llama-7b-hf")  # Example LLaMA model, adjust accordingly

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Create smaller training and evaluation datasets for faster fine-tuning
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# Load pre-trained LLaMA model for causal language modeling
model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")

# Modify the model to adapt for sequence classification
# Add a classification head by freezing the base model and adding an output layer for classification
class LLaMAForSequenceClassification(nn.Module):
    def __init__(self, base_model, num_labels):
        super(LLaMAForSequenceClassification, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(self.base_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Forward pass through the base LLaMA model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        # Take the hidden state of the last token in each sequence (useful for classification tasks)
        last_hidden_state = outputs[0][:, -1, :]
        # Pass through the classification head
        logits = self.classifier(last_hidden_state)

        # If labels are provided, calculate loss
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.classifier.out_features), labels.view(-1))

        return {"loss": loss, "logits": logits}

# Number of labels for classification task (5 labels for Yelp reviews)
num_labels = 5
model = LLaMAForSequenceClassification(model, num_labels)

# Define training arguments for GPU-based fine-tuning
training_args = TrainingArguments(
    output_dir="test_trainer_llama",       # Output directory
    evaluation_strategy="epoch",           # Evaluate after each epoch
    per_device_train_batch_size=16,        # Batch size per device during training
    per_device_eval_batch_size=16,         # Batch size per device during evaluation
    num_train_epochs=3,                    # Total number of training epochs
    weight_decay=0.01,                     # Strength of weight decay
    logging_dir="./logs_llama",            # Directory for logging
    logging_steps=10,                      # Log every 10 steps
    save_steps=500,                        # Save checkpoint every 500 steps
    fp16=True,                             # Enable mixed precision (faster training)
)

# Load accuracy metric for evaluation
metric = evaluate.load("accuracy")

# Compute metrics function for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Define the Trainer for fine-tuning
trainer = Trainer(
    model=model,                          # The modified LLaMA model with classification head
    args=training_args,                   # Training arguments
    train_dataset=small_train_dataset,    # Training dataset
    eval_dataset=small_eval_dataset,      # Evaluation dataset
    compute_metrics=compute_metrics,      # Function to compute evaluation metrics
)

# Fine-tune the LLaMA model
trainer.train()
