## A step by step guide to Fine Tuning a Language Model

This repository contains scripts to **fine-tuning a pre-trained language model** (like BERT, GPT-2) using Hugging Face libraries.

## What is Fine-Tuning?

Fine-tuning is the process of continuing the training of a pre-trained model on a **task-specific dataset**. 
This allows the model to adapt to a specific domain or task, such as sentiment analysis, question answering, or text generation.

---

## 🛠️ Tools We Use

| Tool                | Purpose                                |
|---------------------|----------------------------------------|
| `transformers`      | Model loading and training             |
| `datasets`          | Loading and preprocessing data         |
| `accelerate`        | Efficient multi-device training        |
| `evaluate`          | Metrics like accuracy, F1              |
| `peft`              | Parameter-efficient tuning (e.g. LoRA) |

---