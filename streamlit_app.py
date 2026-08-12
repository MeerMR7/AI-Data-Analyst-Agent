import os
import asyncio
import streamlit as st
from sql_agent import run_sql_agent

st.set_page_config(page_title="AI Data Analyst Agent", page_icon="📊", layout="wide")
st.title("📊 AI Data Analyst Agent")

prompt = st.text_input("Ask a question about your data:", placeholder="e.g., What are the total sales per region?")

if st.button("Submit", type="primary"):
    if prompt.strip():
        chart_path = "sales_chart.png"
        if os.path.exists(chart_path):
            try:
                os.remove(chart_path)
            except Exception:
                pass

        with st.spinner("Analyzing data..."):
            try:
                # Runs the agent directly inside Python — NO HTTP/FastAPI server required!
                result = asyncio.run(run_sql_agent(prompt))
                
                st.subheader("💡 Analysis & Findings")
                st.write(result.get("final_answer", "No analysis returned."))
                
                if os.path.exists(chart_path):
                    st.subheader("📈 Visualization")
                    st.image(chart_path, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error executing agent: {e}")
