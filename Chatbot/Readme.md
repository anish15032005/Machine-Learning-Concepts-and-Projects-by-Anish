# 🧠 Text Simplification & Q&A Chatbots using Transformers + LangChain

This repository contains two Python-based chatbot models that utilize Hugging Face Transformers and LangChain to perform **text summarization**, **simplification**, and **question answering**. These models are optimized to run on GPU (CUDA-enabled) environments.

---

## 📂 Files Overview

### `example1.py` – Basic Age-Specific Summarizer 🧒👵

A simple chatbot that:
- Uses the `facebook/bart-large-cnn` model to summarize text.
- Simplifies the text based on the user's specified **age**.
- Wraps the Hugging Face pipeline using LangChain's `PromptTemplate` and `HuggingFacePipeline`.

#### 🔧 How It Works:
1. Asks the user to input:
   - The text they want to summarize.
   - The target **age** for simplification (e.g., "10", "18", "60").
2. Uses the provided age to modify the prompt dynamically.
3. Generates a simplified summary using the summarization model.

#### ✅ Example Usage:
```bash
python example1.py

####Install the required libraries
pip install transformers langchain langchain-huggingface torch
