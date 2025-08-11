import gradio as gr
from uuid import uuid4
from langchain_core.messages import HumanMessage

from app.chatbot.bot import graph, initial_message

session_id = str(uuid4())
chat_graph = graph()
chat_history = [initial_message]

def chat_fn(user_input,history):
 
    chat_history.append(HumanMessage(content=user_input))
    result = chat_graph.invoke(
        {"messages": chat_history},
        config={"configurable": {"thread_id": session_id}}
    )
    last_msg = result["messages"][-1]
    chat_history.append(last_msg)
    return last_msg.content

# Gradio UI
chat_ui = gr.ChatInterface(
    fn=chat_fn,
    title="Sales Assitant",
    description="Chatbot trained on fashion products from Flipkart. Ask for suggestions, deals, or explore categories!",
    theme=gr.themes.Soft(),
   examples=[
    "What are the best-rated shoes?",
    "Do you have any Puma t-shirts?",
    "Which jeans are available for women?",
    "Suggest some Flipkart Advantage products"],
    submit_btn="Send",
    # # retry_btn="Try Again",
    # undo_btn="Undo",
)


if __name__ == "__main__":
    chat_ui.launch()
