from dotenv import load_dotenv
from app.vector_store.build_index import load_vector_store, load_dataframe
from langchain.tools.retriever import create_retriever_tool
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import Tool
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from typing_extensions import TypedDict
import re

# Load environment and data
load_dotenv()
df = load_dataframe()
memory = MemorySaver()

# Expanded assistant prompt
initial_message = SystemMessage(
    content=(
        "You are a helpful virtual shopping assistant for Flipkart, a leading Indian eCommerce store. "
        "You have access to a dataset with the following fields: "
        "product_url, product_name, product_category_tree, pid, retail_price, discounted_price, image, "
        "is_FK_Advantage_product, description, product_rating, overall_rating, brand, product_specifications. "
        "Use this structured information to answer user questions.\n\n"
        "⚠️ Always rely only on the data provided. Never make assumptions or hallucinate values. "
        "If a question requires data that is missing or unavailable, respond with:\n"
        "'Sorry, I don't have that information.'\n\n"
        "💡 You can answer questions about:\n"
        "- Prices (discounted and retail)\n"
        "- Product categories\n"
        "- Brands and their ratings\n"
        "- Product descriptions and specifications\n"
        "- Product names and availability\n"
        "- FK Advantage eligibility\n"
        "- Any patterns or filters the user may request (e.g. under ₹500, top-rated Samsung phones, etc.)\n\n"
        "If the user requests a product list or suggestions, retrieve items from the dataset accordingly. "
        "Do not generate or suggest items not present in the dataset."
        "ANY QUESTION OR STATEMENT THAT DOESNT RELATE TO YOUR JOB AS A CHATBOT FOR THE FLIPKART SHOPPING ASSISTANT JUST SAY ,'Sorry, I don't have that information on such topic or issue. thats it nothing more or less'\n\n"
        "You can tell about yourself as if someone ask and what you do"
    )
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def extract_price(text):
    match = re.search(r"(\d{2,7})", text)
    return int(match.group(1)) if match else None

def query_product_data(question: str) -> str:
    q = question.lower()

    if "price range" in q:
        min_price = df["discounted_price"].min()
        max_price = df["discounted_price"].max()
        return f"The price range of all products is ₹{min_price} to ₹{max_price}."

    elif "under" in q or "below" in q:
        price = extract_price(q)
        if price:
            count = (df["discounted_price"] < price).sum()
            return f"There are {count} products under ₹{price}."
        return "Please specify a price to filter under."

    elif "above" in q or "over" in q:
        price = extract_price(q)
        if price:
            filtered = df[df["discounted_price"] > price].sort_values(by="overall_rating", ascending=False).head(5)
            if not filtered.empty:
                products = "\n".join(
                    f"{row['product_name']} (₹{row['discounted_price']}, ⭐ {row['overall_rating']})"
                    for _, row in filtered.iterrows()
                )
                return f"Top products above ₹{price}:\n{products}"
            return f"No products found above ₹{price}."
        return "Please specify a price to filter above."

    elif "highest-rated brand" in q:
        grouped = df.groupby("brand")["overall_rating"].mean().dropna()
        best_brand = grouped.idxmax()
        rating = grouped.max()
        return f"The highest-rated brand is {best_brand} with an average rating of {rating:.2f}."

    elif "most common brand" in q:
        brand = df["brand"].mode()[0]
        return f"The most common brand is {brand}."

    elif "flipkart advantage" in q:
        adv_count = df["is_FK_Advantage_product"].astype(str).str.lower().eq("true").sum()
        return f"There are {adv_count} Flipkart Advantage products."

    return "This question cannot be answered with structured data. Try rephrasing or ask about prices, brands, or filters."

# Build LangGraph with tools
def graph():
    # Load retriever (FAISS index)
    retriever = load_vector_store().as_retriever(search_kwargs={"k": 8})

    # Tools
    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="ProductSearch",
        description="Use this to look up products from Flipkart by category, name, specs, etc."
    )

    structured_tool = Tool(
        name="query_product_data",
        description="Use this to answer product-wide questions like price range, top brands, and filtering.",
        func=query_product_data,
    )

    llm = init_chat_model("google_genai:gemini-2.0-flash")
    llm_with_tools = llm.bind_tools([retriever_tool, structured_tool])

    def chatbot(state: State):
        result = llm_with_tools.invoke(state["messages"])
        return {"messages": state["messages"] + [result]}

    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot)
    builder.add_node("tools", ToolNode([retriever_tool, structured_tool]))

    builder.set_entry_point("chatbot")
    builder.add_conditional_edges("chatbot", tools_condition)
    builder.add_edge("tools", "chatbot")
    builder.set_finish_point("chatbot")

    return builder.compile(checkpointer=memory)