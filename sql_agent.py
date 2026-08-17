import os
import json
import sqlite3
import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from openai import AsyncOpenAI

# Force load environment variables from local .env file
load_dotenv(override=True)

app = FastAPI(title="AI Data Analyst Agent (Powered by Groq)")

# =====================================================================
# 1. GROQ CLIENT CONFIGURATION
# =====================================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set in environment or .env file!")

client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

# =====================================================================
# 2. DATABASE SETUP
# =====================================================================
DB_PATH = "company_data.db"

def init_mock_database():
    """Populates local SQLite database with sample company sales data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY,
            region TEXT,
            category TEXT,
            amount REAL,
            date TEXT
        )
    """)
    cursor.execute("SELECT COUNT(*) FROM sales")
    if cursor.fetchone()[0] == 0:
        sample_data = [
            ("North", "Electronics", 1200.50, "2026-01-15"),
            ("North", "Furniture", 450.00, "2026-01-18"),
            ("South", "Electronics", 2300.00, "2026-02-01"),
            ("South", "Furniture", 890.20, "2026-02-10"),
            ("East", "Electronics", 3100.00, "2026-02-14"),
            ("East", "Furniture", 150.00, "2026-02-20"),
            ("West", "Electronics", 950.00, "2026-03-01"),
            ("West", "Furniture", 1100.00, "2026-03-05"),
        ]
        cursor.executemany("INSERT INTO sales (region, category, amount, date) VALUES (?, ?, ?, ?)", sample_data)
        conn.commit()
    conn.close()

init_mock_database()

# =====================================================================
# 3. AGENT TOOLS
# =====================================================================

async def get_db_schema() -> str:
    """Returns the SQL schema definition for all tables in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
    schemas = [row[0] for row in cursor.fetchall() if row[0]]
    conn.close()
    return "\n".join(schemas)

async def execute_sql_query(sql_query: str) -> str:
    """Executes SQLite query and returns results as JSON string or error text."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        if df.empty:
            return "Query executed successfully, but returned 0 rows."
        return df.to_json(orient="records")
    except Exception as e:
        return f"SQL Execution Error: {str(e)}"

async def generate_chart(data_json: Any, x_col: str, y_col: str, chart_title: str) -> str:
    """Generates a bar chart and saves it locally."""
    try:
        if isinstance(data_json, str):
            df = pd.read_json(data_json)
        else:
            df = pd.DataFrame(data_json)
            
        plt.figure(figsize=(8, 4))
        plt.bar(df[x_col].astype(str), df[y_col], color="#2563eb")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(chart_title)
        plt.tight_layout()
        
        file_path = "sales_chart.png"
        plt.savefig(file_path)
        plt.close()
        return f"Chart successfully saved to {file_path}"
    except Exception as e:
        return f"Chart Generation Error: {str(e)}"

TOOL_MAPPING = {
    "get_db_schema": get_db_schema,
    "execute_sql_query": execute_sql_query,
    "generate_chart": generate_chart,
}

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_db_schema",
            "description": "Fetches database schemas.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_sql_query",
            "description": "Executes a SQL query against SQLite database.",
            "parameters": {
                "type": "object",
                "properties": {"sql_query": {"type": "string"}},
                "required": ["sql_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_chart",
            "description": "Generates a bar chart from a list of data record objects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_json": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "List of record dictionaries returned by SQL query."
                    },
                    "x_col": {"type": "string", "description": "Column name for X-axis."},
                    "y_col": {"type": "string", "description": "Column name for Y-axis."},
                    "chart_title": {"type": "string", "description": "Title of the chart."}
                },
                "required": ["data_json", "x_col", "y_col", "chart_title"]
            }
        }
    }
]

# =====================================================================
# 4. REACT AGENT EXECUTION LOOP
# =====================================================================
async def run_sql_agent(user_prompt: str) -> Dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert Data Analyst AI Agent.\n"
                "1. Always inspect DB schema first using `get_db_schema`.\n"
                "2. Write and execute SQLite queries using `execute_sql_query`.\n"
                "3. When requested to generate a chart, pass the records returned by `execute_sql_query` directly into `generate_chart`.\n"
                "4. Return a clean analytical final answer with key business insights."
            )
        },
        {"role": "user", "content": user_prompt}
    ]
    execution_trace = []
    
    for turn in range(6):
response = await client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=messages,
    tools=AGENT_TOOLS,
    temperature=0.0
)
        
        message = response.choices[0].message
        
        if not message.tool_calls:
            return {"final_answer": message.content, "execution_trace": execution_trace}
        
        messages.append(message)
        
        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            raw_args = tool_call.function.arguments
            
            # SAFE ARGUMENT PARSING
            try:
                if isinstance(raw_args, str) and raw_args.strip():
                    fn_args = json.loads(raw_args)
                elif isinstance(raw_args, dict):
                    fn_args = raw_args
                else:
                    fn_args = {}
            except Exception:
                cleaned_str = str(raw_args).replace('\\"', '"').replace('\\\\', '\\')
                fn_args = json.loads(cleaned_str)

            if not isinstance(fn_args, dict):
                fn_args = {}

            execution_trace.append({"step": turn + 1, "tool": fn_name, "args": fn_args})
            
            if fn_name in TOOL_MAPPING:
                tool_output = await TOOL_MAPPING[fn_name](**fn_args)
            else:
                tool_output = f"Error: Tool {fn_name} not found."
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": fn_name,
                "content": str(tool_output)
            })

    return {"final_answer": "Reached maximum reasoning steps.", "execution_trace": execution_trace}

# =====================================================================
# 5. FASTAPI ENDPOINT
# =====================================================================

class QueryRequest(BaseModel):
    prompt: str = Field(
        ..., 
        json_schema_extra={"example": "What are the total sales per region? Plot a chart for it."}
    )

@app.post("/api/analyze")
async def analyze_data(request: QueryRequest):
    try:
        result = await run_sql_agent(request.prompt)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("sql_agent:app", host="0.0.0.0", port=8000, reload=True)
