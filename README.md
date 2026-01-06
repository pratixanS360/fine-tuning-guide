![Python](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)
![Hugging Face](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=black)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-green.svg)
![Status](https://img.shields.io/badge/status-experimental-orange.svg)

## A step by step guide to Fine Tuning a Language Model

This repository contains scripts to **fine-tune a pre-trained language model** (like BERT, GPT-2) using Hugging Face libraries.

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
