from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)

import os
import pandas as pd
import matplotlib.pyplot as plt
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import (
    AgentExecutor,
    create_tool_calling_agent
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain.memory import ConversationBufferMemory

SCRIPT_DIR = Path(__file__).resolve().parent
BANK_DATA_PATH = SCRIPT_DIR / "data" / "bank_data.csv"


# -------------------------------
# Tool 1 - Load Dataset
# -------------------------------

@tool
def load_bank_data(dummy: str = "") -> str:
    """Load the bank dataset."""

    if not BANK_DATA_PATH.exists():
        return "bank_data.csv not found. Please place it in the project folder."

    return "Bank dataset loaded successfully."


# -------------------------------
# Tool 2 - Expense Analysis
# -------------------------------

@tool
def analyze_expenses(dummy: str = "") -> str:
    """Analyze total income, expenses and savings."""

    df = pd.read_csv(BANK_DATA_PATH)

    income = df[df["Type"] == "Credit"]["Amount"].sum()
    expenses = df[df["Type"] == "Debit"]["Amount"].sum()
    savings = income - expenses

    report = f"""
Financial Summary

Total Income : ₹{income}

Total Expenses : ₹{expenses}

Monthly Savings : ₹{savings}
"""

    return report


# -------------------------------
# Tool 3 - Spending Categories
# -------------------------------

@tool
def spending_categories(dummy: str = "") -> str:
    """Show spending category wise."""

    df = pd.read_csv(BANK_DATA_PATH)

    debit_df = df[df["Type"] == "Debit"]

    category = debit_df.groupby("Category")["Amount"].sum().sort_values(ascending=False)

    return category.to_string()


# -------------------------------
# Tool 4 - Highest Spending
# -------------------------------

@tool
def highest_spending(dummy: str = "") -> str:
    """Find where user spends the most."""

    df = pd.read_csv(BANK_DATA_PATH)

    debit_df = df[df["Type"] == "Debit"]

    category = debit_df.groupby("Category")["Amount"].sum()

    highest = category.idxmax()
    amount = category.max()

    return f"You spend the most on {highest} (₹{amount})."


# -------------------------------
# Tool 5 - Budget Suggestions
# -------------------------------

@tool
def budget_suggestions(dummy: str = "") -> str:
    """Provide budget suggestions based on expenses."""

    df = pd.read_csv(BANK_DATA_PATH)

    debit_df = df[df["Type"] == "Debit"]

    category = debit_df.groupby("Category")["Amount"].sum()

    highest = category.idxmax()

    suggestions = f"""
Budget Suggestions

• Reduce spending in {highest}.

• Avoid frequent online shopping.

• Limit food delivery orders.

• Continue investing through SIP.

• Keep emergency savings equal to 6 months of expenses.

• Track monthly expenses regularly.
"""

    return suggestions


# -------------------------------
# Tool 6 - Expense Chart
# -------------------------------

@tool
def create_expense_chart(dummy: str = "") -> str:
    """Create a pie chart of spending categories."""

    df = pd.read_csv(BANK_DATA_PATH)

    debit_df = df[df["Type"] == "Debit"]

    category = debit_df.groupby("Category")["Amount"].sum()

    plt.figure(figsize=(8,8))

    plt.pie(
        category,
        labels=category.index,
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title("Expense Distribution")

    plt.savefig("expense_chart.png")

    plt.close()

    return "Expense chart saved as expense_chart.png."


# -------------------------------
# Memory
# -------------------------------

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# -------------------------------
# LLM
# -------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# -------------------------------
# Tools
# -------------------------------

tools = [
    load_bank_data,
    analyze_expenses,
    spending_categories,
    highest_spending,
    budget_suggestions,
    create_expense_chart
]

# -------------------------------
# Prompt
# -------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an AI Financial Advisor.

You can:

1. Analyze expenses.
2. Calculate monthly savings.
3. Show spending categories.
4. Tell where the user spends the most.
5. Suggest budgeting tips.
6. Generate an expense chart.

Always use tools whenever required.
"""
        ),

        MessagesPlaceholder(variable_name="chat_history"),

        ("human", "{input}"),

        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)

# -------------------------------
# Agent
# -------------------------------

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# -------------------------------
# Chat Loop
# -------------------------------

print("=====================================")
print(" AI Financial Advisor")
print("=====================================")
print("Type 'exit' to quit.\n")

while True:

    user_input = input("You : ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    response = agent_executor.invoke(
        {
            "input": user_input
        }
    )

    print("\nAdvisor:")
    print("----------------------------")
    print(response["output"])
    print("----------------------------")