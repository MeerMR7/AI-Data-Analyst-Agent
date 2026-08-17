import os
import json
import sqlite3
import asyncio
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="SQL Agent API")

# Initialize Async Client for Groq API
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

DB_PATH = "company.db"


def execute_sql_query(sql_query: str):
    """Executes a SQL query against the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(sql_query)
        if sql_query.strip().upper().startswith("SELECT"):
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            result = [dict(zip(columns, row)) for row in rows]
        else:
            conn.commit()
            result = {"status": "success", "rows_affected": cursor.rowcount}
        conn.close()
        return result
    except Exception as e:
        conn.close()
        return {"error": str(e)}


def generate_chart(categories: list, values: list, chart_title: str, x_label: str, y_label: str):
    """Generates and saves a bar chart image to sales_chart.png."""
    try:
        plt.figure(figsize=(10, 6))
        plt.bar(categories, values, color="skyblue")
        plt.title(chart_title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xticks(rotation=45)
        plt.tight_layout()
        chart_path = "sales_chart.png"
        plt.savefig(chart_path)
        plt.close()
        return {"status": "chart generated", "file_path": chart_path}
    except Exception as e:
        return {"error": str(e)}


AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_sql_query",
            "description": "Execute a SQL query against the SQLite database. Available table: sales (id, region, category, amount, date).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql_query": {
                        "type": "string",
                        "description": "Valid SQLite query statement."
                    }
                },
                "required": ["sql_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_chart",
            "description": "Generate a bar chart visualization and save it as sales_chart.png.",
            "parameters": {
                "type": "object",
                "properties": {
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Labels for the X axis"
                    },
                    "values": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Numeric values for the Y axis"
                    },
                    "chart_title": {"type": "string"},
                    "x_label": {"type": "string"},
                    "y_label": {"type": "string"}
                },
                "required": ["categories", "values", "chart_title", "x_label", "y_label"]
            }
        }
    }
]


async def run_sql_agent(user_query: str, max_steps: int = 5):
    """Main agent loop that executes tool calls iteratively."""
    system_prompt = (
        "You are an expert AI Data Analyst. Your task is to answer questions about the company database.\n"
        "The database contains a 'sales' table with columns: id, region, category, amount, date.\n"
        "1. Always query the database using 'execute_sql_query' to retrieve data.\n"
        "2. If the user asks for a chart or visualization, call 'generate_chart'.\n"
        "3. Present a clear analytical summary as your final answer."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    execution_trace = []

    for step in range(max_steps):
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=AGENT_TOOLS,
            temperature=0.0
        )

        response_message = response.choices[0].message
        messages.append(response_message)

        if not response_message.tool_calls:
            return {
                "final_answer": response_message.content,
                "execution_trace": execution_trace
            }

        for tool_call in response_message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            execution_trace.append({
                "tool": func_name,
                "args": func_args
            })

            if func_name == "execute_sql_query":
                tool_result = execute_sql_query(func_args.get("sql_query"))
            elif func_name == "generate_chart":
                tool_result = generate_chart(
                    func_args.get("categories"),
                    func_args.get("values"),
                    func_args.get("chart_title"),
                    func_args.get("x_label"),
                    func_args.get("y_label")
                )
            else:
                tool_result = {"error": f"Unknown tool {func_name}"}

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_result)
            })

    return {
        "final_answer": "Reached maximum reasoning steps.",
        "execution_trace": execution_trace
    }
