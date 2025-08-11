---

# Sales Merchandise Chatbot (Flipkart 20K Dataset)

An AI-powered chatbot for fashion merchandise that delivers accurate, context-aware responses from a **20,000-product Flipkart dataset**.
Built using **FAISS**, **SentenceTransformers**, **LangGraph/LangChain**, and **Gemini LLM**, with a **Gradio** UI.

---

## Features

* **Semantic Search + FAISS** — Fast vector-based product retrieval over 20K items.
* **LangGraph Controller** — Decides when and how to retrieve data or use the LLM.
* **Gemini LLM Integration** — Generates natural, context-rich answers.
* **Gradio Interface** — Simple, interactive UI for users to chat with the bot.
* **100% Retrieval Accuracy** (in benchmark tests) with \~3s average response time.

---

## Project Structure

```
sales-bot/
├── app/
│   ├── main.py                  # Entry point for running chatbot UI
│   ├── evaluate.py              # Benchmark evaluation script
│   ├── vectorstore/
│   │   └── build_index.py       # Builds FAISS index from dataset
│   ├── chatbot/
│   │   └── bot.py               # LangGraph workflow & bot logic
├── data/                        # Place cleaned dataset here (not included in repo)
│   └── your_cleaned_dataset.csv
├── models/                      # Place downloaded SentenceTransformer model here (not included in repo)
│   └── models--sentence-transformers--all-MiniLM-L6-v2
├── .env                         # API keys (not committed to Git)
├── requirements.txt             # Python dependencies
```

---

## Dataset

This project uses the [Flipkart Products Dataset (20K items)](https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products), cleaned and preprocessed for semantic search.
It includes categories such as:

* Shoes
* T-shirts
* Jeans
* Flipkart Advantage products
* And more…

⚠️ **Dataset is not included in this repository** due to size limits. You must download it from Kaggle and place it in the `data/` folder.

---

## Models

The project uses the **SentenceTransformer model**:
[`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

⚠️ **Model files are not included** due to size limits.
Download the model and place it in:

```
models/models--sentence-transformers--all-MiniLM-L6-v2/
```

Ensure your `build_index.py` and `bot.py` scripts point to the correct local model path.

---

## Environment Variables

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_gemini_api_key
MODEL_PATH=your models folder path
```

Make sure `.env` is in `.gitignore`.

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/fatimafarhan2/sales-merch-chatbot.git
cd sales-bot
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download dataset & model**

* **Dataset:** Download from [Kaggle](https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products) → place in `data/`
* **Model:** Download from [Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) → place in `models/`

5. **Build FAISS index**

```bash
python -m app.vectorstore.build_index
```

---

## Running the Chatbot

```bash
python -m app.main
```

Or with `uvicorn`:

```bash
uvicorn app.main:app --reload
```

---

## Running Evaluation

```bash
python -m app.evaluate
```

---

## Evaluation Results

* **Retrieval Accuracy:** 100%
* **Average Response Time:** \~3 seconds
* **Coverage:** 100%

---

## Tech Stack

* **SentenceTransformer + FAISS** — Search index / memory
* **LangGraph / LangChain** — Workflow orchestration
* **Gradio** — Chat UI
* **Flipkart 20K Dataset** — Knowledge base

---
