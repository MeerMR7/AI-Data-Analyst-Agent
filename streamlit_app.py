import os
import asyncio
import streamlit as st
from sql_agent import run_sql_agent

# Page configuration
st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Data Analyst Agent")
st.write("Ask questions about your SQL database, and get automated insights and visualizations.")

# Input text box
prompt = st.text_input("Ask a question about your data:", placeholder="e.g., What are the total sales per region? Plot a chart for it.")

if st.button("Submit", type="primary"):
    if prompt.strip():
        # Clean up any previously generated chart to prevent showing outdated graphs
        chart_path = "sales_chart.png"
        if os.path.exists(chart_path):
            try:
                os.remove(chart_path)
            except Exception:
                pass

        with st.spinner("Analyzing data and generating query..."):
            try:
                # Run the async agent directly in Python
                result = asyncio.run(run_sql_agent(prompt))
                
                st.subheader("💡 Analysis & Findings")
                st.write(result.get("final_answer", "No analysis returned."))
                
                # Render chart if the agent created one
                if os.path.exists(chart_path):
                    st.subheader("📈 Visualization")
                    st.image(chart_path, caption="Generated Data Chart", use_container_width=True)
                    
            except Exception as e:
                st.error(f"An error occurred while processing your request: {e}")
    else:
        st.warning("Please enter a question before clicking Submit.")
