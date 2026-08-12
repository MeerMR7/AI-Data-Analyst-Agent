import os
import requests
import streamlit as st

st.set_page_config(page_title="AI Data Analyst Agent", page_icon="📊", layout="wide")

st.title("📊 AI Data Analyst Dashboard")
st.markdown("Ask natural language questions about your company database in real-time.")

with st.sidebar:
    st.header("⚙️ Agent Status")
    st.success("Connected to FastAPI Backend")
    st.markdown("---")
    st.markdown("### 💡 Sample Prompts")
    st.markdown("* *'What are the total sales per region? Plot a chart for it.'*")
    st.markdown("* *'Which category generated the highest revenue?'*")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "image" in msg and msg["image"] and os.path.exists(msg["image"]):
            st.image(msg["image"])

if user_prompt := st.chat_input("Type your data query here..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Agent analyzing schema, executing SQL, and generating insights..."):
            try:
                chart_path = "sales_chart.png"
                if os.path.exists(chart_path):
                    try:
                        os.remove(chart_path)
                    except Exception:
                        pass

                response = requests.post(
                    "http://localhost:8000/api/analyze",
                    json={"prompt": user_prompt},
                    timeout=120
                )

                if response.status_code == 200:
                    data = response.json()
                    final_answer = data.get("final_answer", "No answer returned.")
                    execution_trace = data.get("execution_trace", [])

                    st.markdown(final_answer)

                    saved_img = None
                    if os.path.exists(chart_path):
                        st.image(chart_path, caption="Agent Generated Visualization")
                        saved_img = chart_path

                    with st.expander("🔍 View Agent Thought Process & SQL Trace"):
                        st.json(execution_trace)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_answer,
                        "image": saved_img
                    })
                else:
                    st.error(f"Error {response.status_code}: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("⚠️ Could not connect to FastAPI server. Make sure `python sql_agent.py` is running on port 8000!")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")