from transformers import AutoTokenizer, AutoModelForSequenceClassification  
from transformers import pipeline
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="Inference for fine tuned model")
parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='Base model name or path')
parser.add_argument('--input', type=str, default=None, help='Input string for classification')
args = parser.parse_args()

# Parameters
model_name = args.model_name
input_string = args.input

# Inference
def inference():
    # Load model
    model_path = f'../output/{model_name}'
    model = AutoModelForSequenceClassification.from_pretrained(model_path) 

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # pipeline
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    result = classifier(input_string)

    return result

if __name__ == '__main__':
    result = inference()
    print(result)