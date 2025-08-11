import time
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple
from app.chatbot.bot import graph, initial_message
from langchain_core.messages import HumanMessage


app = FastAPI(title="Chatbot Evaluation API")


session_graph = graph()
chat_history = [initial_message]

def chat_fn(user_input: str) -> str:
    chat_history.append(HumanMessage(content=user_input))
    result = session_graph.invoke(
        {"messages": chat_history},
        config={"configurable": {"thread_id": "test-session"}}
    )
    last_msg = result["messages"][-1]
    chat_history.append(last_msg)
    return last_msg.content

default_test_cases: List[Tuple[str, str]] = [
    ("Do you have any Puma t-shirts?", "Puma"),
    ("Which jeans are available for women?", "jeans"),
    ("Top-rated shoes above ₹2000", "shoes"),
    ("What is the most common brand?", "brand"),
    ("Show me Flipkart Advantage products", "Flipkart Advantage")
]

def evaluate(test_cases: List[Tuple[str, str]]):
    correct = 0
    times = []
    covered_keywords = set()

    for q, expected in test_cases:
        start = time.time()
        answer = chat_fn(q)
        end = time.time()
        times.append(end - start)

        if expected.lower() in answer.lower():
            correct += 1
            covered_keywords.add(expected.lower())

    accuracy = (correct / len(test_cases)) * 100
    avg_time = sum(times) / len(times)
    coverage = (len(covered_keywords) / len(test_cases)) * 100

    return {
        "Retrieval Accuracy (%)": round(accuracy, 2),
        "Average Response Time (s)": round(avg_time, 2),
        "Coverage/Scope (%)": round(coverage, 2),
        "Total Queries Tested": len(test_cases)
    }

class EvalRequest(BaseModel):
    test_cases: List[Tuple[str, str]] = default_test_cases

@app.get("/evaluate")
def evaluate_default():
    """Run evaluation with default test cases."""
    return evaluate(default_test_cases)

@app.post("/evaluate")
def evaluate_custom(data: EvalRequest):
    """Run evaluation with custom test cases."""
    return evaluate(data.test_cases)

if __name__ == "__main__":

    results = evaluate(default_test_cases)
    for k, v in results.items():
        print(f"{k}: {v}")
