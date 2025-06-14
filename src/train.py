from transformers import AutoModelForSequenceClassification, AutoTokenizer 
from transformers import Trainer, TrainingArguments
import torch
from datasets import load_dataset
import argparse
import re

# Parse arguments
parser = argparse.ArgumentParser(description="Train a BERT model for sequence classification.")
parser.add_argument('--dataset_name', type=str, default='stanfordnlp/imdb', help='Dataset name or path')
parser.add_argument('--model_name', type=str, default='google-bert/bert-base-uncased', help='Base model name or path')
parser.add_argument('--train_file', type=str, default='./train.txt', help='Path to the training file')
parser.add_argument('--output_dir', type=str, default='../output', help='Directory to save the model and tokenizer')
parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for training')
args = parser.parse_args()

# Configurations
dataset_name = args.dataset_name
model_name = args.model_name
train_file = args.train_file
output_dir = args.output_dir
epochs = args.epochs
batch_size = args.batch_size
#learning_rate = args.learning_rate

def train_model():
    # Prepare dataset
    dataset = load_dataset(dataset_name)
    print(f"Dataset '{dataset_name}' loaded successfully.")

    # Tokenize the dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True
    )
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    tokenized_dataset_train = tokenized_dataset['train']  # Use the training split
    tokenized_dataset_test = tokenized_dataset['test']  # Use the test split

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_steps=500,
        logging_dir='../logs',
        logging_steps=100,
        #learning_rate=learning_rate,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_test,
    )

    # Train
    print("Training:")
    trainer.train()

    # Save model
    match = re.search(r'[^/]+$', model_name)
    name = match.group(0)
    print(name)
    
    trainer.save_model(f'{output_dir}/{name}')
    tokenizer.save_pretrained(f'{output_dir}/{name}')

if __name__ == "__main__":
    train_model()
    print("Training complete. Model and tokenizer saved to:", output_dir)
    print("You can now use the trained model for inference or further fine-tuning.")
