import os
import asyncio
import streamlit as st
from sql_agent import run_sql_agent

# 1. Page Configuration
st.set_page_config(
    page_title="AI Data Analyst Dashboard",
    page_icon="📊",
    layout="wide"
)

# 2. Sidebar Configuration
st.sidebar.title("⚙️ Agent Status")
st.sidebar.success("Cloud Engine Ready")

st.sidebar.markdown("---")
st.sidebar.subheader("💡 Sample Prompts")
st.sidebar.markdown(
    """
    * *'What are the total sales per region? Plot a chart for it.'*
    * *'Which category generated the highest revenue?'*
    * *'Show total sales count for Furniture.'*
    """
)

# 3. Main Dashboard Header
st.title("📊 AI Data Analyst Dashboard")
st.write("Ask natural language questions about your company database in real-time.")

# 4. Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # Display stored SQL code if present
        if "sql_queries" in message:
            for sql in message["sql_queries"]:
                st.code(sql, language="sql")
                
        if "image" in message and os.path.exists(message["image"]):
            st.image(message["image"], use_container_width=True)

# 5. User Query Input
user_input = st.chat_input("Type your data query here...")

if user_input:
    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Process Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing data and generating SQL queries..."):
            # Clean up old chart before executing
            chart_path = "sales_chart.png"
            if os.path.exists(chart_path):
                try:
                    os.remove(chart_path)
                except Exception:
                    pass

            try:
                # Execute agent logic
                result = asyncio.run(run_sql_agent(user_input))
                final_answer = result.get("final_answer", "No analysis returned.")
                trace = result.get("execution_trace", [])
                
                # Extract executed SQL queries from trace
                extracted_sql = []
                for step in trace:
                    if step.get("tool") == "execute_sql_query":
                        query = step.get("args", {}).get("sql_query")
                        if query:
                            extracted_sql.append(query)

                # 1. Show Generated SQL Queries
                if extracted_sql:
                    st.subheader("🔍 Generated SQL Query")
                    for sql in extracted_sql:
                        st.code(sql, language="sql")

                # 2. Show Final Analytical Answer
                st.subheader("💡 Business Insight")
                st.write(final_answer)

                # 3. Render Chart if Generated
                has_chart = os.path.exists(chart_path)
                if has_chart:
                    st.subheader("📈 Visualization")
                    st.image(chart_path, use_container_width=True)

                # Save assistant response details to chat history
                msg_data = {"role": "assistant", "content": final_answer}
                if extracted_sql:
                    msg_data["sql_queries"] = extracted_sql
                if has_chart:
                    msg_data["image"] = chart_path

                st.session_state.messages.append(msg_data)

            except Exception as e:
                st.error(f"Error executing agent query: {e}")
