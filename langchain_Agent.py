from dotenv import load_dotenv
load_dotenv(override=True)

import datetime
import pandas as pd

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# ------------------ TOOLS ------------------

@tool
def get_current_time(dummy: str = "") -> str:
    """Returns the current date and time."""
    return str(datetime.datetime.now())


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"Result = {result}"
    except Exception as e:
        return f"Error: {e}"


FILE_PATH = "/Users/pragativisghwakarma/Documents/ PROJECTS/GenAI_Tranning/RAG_Example/Data/sample.csv"


@tool
def analyze_csv(dummy: str = "") -> str:
    """Reads a CSV file and returns summary statistics."""
    try:
        df = pd.read_csv(FILE_PATH)

        output = []

        output.append("Columns:")
        output.append(", ".join(df.columns))

        output.append("\nShape:")
        output.append(str(df.shape))

        output.append("\nColumn Averages:")
        output.append(df.mean(numeric_only=True).to_string())

        return "\n".join(output)

    except Exception as e:
        return f"Error reading CSV: {e}"


# ------------------ MEMORY ------------------

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)


# ------------------ LLM ------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)


# ------------------ TOOLS ------------------

tools = [
    get_current_time,
    calculator,
    analyze_csv,
]


# ------------------ PROMPT ------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant with tool-calling abilities. "
            "Use the provided tools whenever necessary."
        ),

        MessagesPlaceholder(variable_name="chat_history"),

        ("human", "{input}"),

        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


# ------------------ AGENT ------------------

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)


agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
)


# ------------------ CHAT LOOP ------------------

print("\n====================================")
print(" Advanced LangChain + Groq Agent")
print("====================================")
print("Type 'exit' to quit.\n")

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("\nBot: Goodbye!")
        break

    response = agent_executor.invoke(
        {
            "input": user_input
        }
    )

    print("\nBot:")
    print("-----------------------------------")
    print(response["output"])
    print("-----------------------------------")