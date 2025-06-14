from transformers import AutoTokenizer, AutoModelForSequenceClassification 
from transformers import Trainer, TrainingArguments 
from datasets import load_dataset
import evaluate
import argparse
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser(description="Inference for fine tuned model")
parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='Base model name or path')
parser.add_argument('--output_dir', type=str, default='../output', help='Directory to save the model and tokenizer')
args = parser.parse_args()

# Parameters
model_name = args.model_name
output_dir = args.output_dir
model_path = f'{output_dir}/{model_name}'

# Evaluate the model
def evaluate_model():
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_path) 

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

    # Load dataset
    dataset = load_dataset('stanfordnlp/imdb')
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True
    )
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    tokenized_dataset_test = tokenized_dataset['test']  # Use the test split

    # Load metric
    metric_accuracy = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = metric_accuracy.compute(predictions=predictions, references=labels)
        f1_score = metric_f1.compute(predictions=predictions, references=labels, average='macro')
        return {"accuracy": accuracy, "f1": f1_score} 
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=model_path), 
        eval_dataset=tokenized_dataset_test,
        compute_metrics=compute_metrics
    )

    # Evaluate
    print("Evaluating the model:")
    result = trainer.evaluate()
    print(f"Results\n--------\nModel: {model_name}\nDataset: 'stanfordnlp/imdb'")
    print(f"Accuracy: {result["eval_accuracy"]["accuracy"]}\nF1 Score: {result["eval_f1"]["f1"]}\n")
    
    return

if __name__ == '__main__':
    evaluate_model()
    print("Evaluation completed successfully!")