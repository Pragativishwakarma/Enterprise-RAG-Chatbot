from dotenv import load_dotenv
load_dotenv(override=True)

import os
import smtplib
import requests
import pandas as pd
import matplotlib.pyplot as plt

from email.message import EmailMessage

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_tool_calling_agent


# ======================================================
# TOOLS
# ======================================================

@tool
def download_sales_data(dummy: str = "") -> str:
    """
    Download sales CSV file.
    """

    url = "https://drive.google.com/uc?export=download&id=1x4UDtoONLJnA_MT093FaeBaLpntYaXPl"

    response = requests.get(url)

    if response.status_code != 200:
        return "Failed to download CSV."

    with open("sales_data.csv", "wb") as f:
        f.write(response.content)

    return "sales_data.csv downloaded successfully."


@tool
def analyze_sales_data(file_path: str) -> str:
    """
    Analyze CSV file.
    """

    try:
        df = pd.read_csv(file_path)

        summary = []

        summary.append("Shape:")
        summary.append(str(df.shape))

        summary.append("\nColumns:")
        summary.append(", ".join(df.columns))

        summary.append("\nStatistics:")
        summary.append(df.describe(include="all").to_string())

        return "\n".join(summary)

    except Exception as e:
        return str(e)


@tool
def create_sales_chart(file_path: str) -> str:
    """
    Create chart from CSV.
    """

    try:
        df = pd.read_csv(file_path)

        numeric_cols = df.select_dtypes(include="number").columns

        if len(numeric_cols) == 0:
            return "No numeric columns found."

        column = numeric_cols[0]

        plt.figure(figsize=(10, 5))
        plt.plot(df[column])
        plt.title(f"{column} Trend")
        plt.xlabel("Index")
        plt.ylabel(column)
        plt.grid(True)

        plt.savefig("sales_chart.png")
        plt.close()

        return "sales_chart.png created successfully."

    except Exception as e:
        return str(e)


@tool
def email_report(dummy: str = "") -> str:
    """
    Email generated report.
    """

    try:

        sender_email = os.getenv("EMAIL_USER")
        sender_password = os.getenv("EMAIL_PASSWORD")

        receiver_email = "pragativish5@gmail.com"

        if sender_email is None or sender_password is None:
            return "EMAIL_USER or EMAIL_PASSWORD missing from .env"

        msg = EmailMessage()

        msg["Subject"] = "AI Generated Sales Report"

        msg["From"] = sender_email

        msg["To"] = receiver_email

        msg.set_content(
            "Please find the attached AI-generated sales report."
        )

        with open("sales_chart.png", "rb") as f:
            msg.add_attachment(
                f.read(),
                maintype="image",
                subtype="png",
                filename="sales_chart.png"
            )

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)

        return "Email sent successfully."

    except Exception as e:
        return str(e)


# ======================================================
# MEMORY
# ======================================================

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

# ======================================================
# LLM
# ======================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# ======================================================
# TOOLS LIST
# ======================================================

tools = [
    download_sales_data,
    analyze_sales_data,
    create_sales_chart,
    email_report,
]

# ======================================================
# PROMPT
# ======================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an Autonomous Workflow AI Agent.

You can:
- Download CSV files
- Analyze CSV data
- Create charts
- Send email reports

Always use the available tools whenever required.
Complete the user's workflow automatically.
"""
        ),

        MessagesPlaceholder(variable_name="chat_history"),

        ("human", "{input}"),

        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# ======================================================
# AGENT
# ======================================================

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
)

# ======================================================
# MAIN
# ======================================================

print("=" * 50)
print(" Autonomous Workflow Agent Started")
print("=" * 50)
print("Type 'exit' to quit.\n")

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    try:

        response = agent_executor.invoke(
            {
                "input": user_input
            }
        )

        print("\nBot:")
        print(response["output"])
        print()

    except Exception as e:
        print(e)