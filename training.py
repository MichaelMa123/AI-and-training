from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate

# Load Yelp dataset
from datasets import load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.ab")
dataset = load_dataset("yelp_review_full")
print(dataset["train"][100])  # Optional: Preview one sample from the dataset

# Load pre-trained BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Create smaller training and evaluation datasets for faster fine-tuning
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# Load pre-trained BERT model for sequence classification with 5 output labels
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./test_trainer",           # Output directory
    evaluation_strategy="epoch",           # Evaluate after each epoch
    per_device_train_batch_size=16,        # Batch size per device during training
    per_device_eval_batch_size=16,         # Batch size per device during evaluation
    num_train_epochs=3,                    # Total number of training epochs
    weight_decay=0.01,                     # Strength of weight decay
    logging_dir="./logs",                  # Directory for logging
    logging_steps=10,                      # Log every 10 steps
    save_steps=500,                        # Save checkpoint every 500 steps
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
    model=model,                          # The pre-trained model
    args=training_args,                   # Training arguments
    train_dataset=small_train_dataset,    # Training dataset
    eval_dataset=small_eval_dataset,      # Evaluation dataset
    compute_metrics=compute_metrics,      # Function to compute evaluation metrics
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")

print("Model and tokenizer saved to ./fine-tuned-model")
