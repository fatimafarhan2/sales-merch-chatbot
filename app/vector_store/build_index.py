from langchain.schema import Document
import pandas as pd
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Set local cache dirs for HF models if not already set
os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), "models"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.path.join(os.getcwd(), "models"))

# Create vectorstore directory if it doesn't exist
os.makedirs("app/vectorstore", exist_ok=True)

# Default paths
DATASET_PATH = os.getenv("DATASET_PATH", "data/your_cleaned_dataset.csv")
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(
        "models",
        "models--sentence-transformers--all-MiniLM-L6-v2",
        "snapshots",
        "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    )
)

def load_csv_file(file_path: str):
    """Load dataset and convert each row to a LangChain Document."""
    df = pd.read_csv(file_path)
    docs = []

    for _, row in df.iterrows():
        text = f"""Product Name: {row['product_name']}
Category Tree: {row['product_category_tree']}
Retail Price: ₹{row['retail_price']}
Discounted Price: ₹{row['discounted_price']}
Brand: {row['brand']}
Product Rating: {row['product_rating']}
Overall Rating: {row['overall_rating']}
Description: {row['description']}
Specifications: {row['product_specifications']}"""

        docs.append(Document(
            page_content=text,
            metadata={"source": row['product_url']}
        ))

    return docs

def vector_embeddings(docs):
    """Create and save FAISS vector index from documents."""
    print("Creating FAISS index...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_PATH,
        encode_kwargs={"batch_size": 64}
    )
    db = FAISS.from_documents(docs, embeddings)
    vectorstore_path = "app/vectorstore"

    print("Saving FAISS index...")
    db.save_local(vectorstore_path)
    print(f"FAISS index saved to {vectorstore_path}")

def load_vector_store():
    """Load FAISS index from disk."""
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_PATH)
    db = FAISS.load_local(
        folder_path="app/vectorstore",
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    return db

def load_dataframe(file_path=DATASET_PATH):
    """Load dataset into a pandas DataFrame."""
    return pd.read_csv(file_path)

if __name__ == "__main__":
    docs = load_csv_file(DATASET_PATH)
    vector_embeddings(docs)
