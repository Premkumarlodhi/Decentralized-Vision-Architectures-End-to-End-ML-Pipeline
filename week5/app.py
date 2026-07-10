import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📉 Federated Training Loss Monitor")

metrics = pd.read_csv("metrics/training_metrics.csv")

st.subheader("Global Loss vs Federated Rounds")
st.line_chart(metrics.set_index("round")["global_loss"])

st.success(
    f"Final Global Loss: {metrics['global_loss'].iloc[-1]:.4f}"
)