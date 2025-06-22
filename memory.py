from langchain.memory import ConversationBufferMemory

# Default memory (k = 8 message window)
memory = ConversationBufferMemory(
    k=8,
    return_messages=False
)


def remember(role: str, text: str):
    if role == "user":
        memory.chat_memory.add_user_message(text)
    else:
        memory.chat_memory.add_ai_message(text)
