import time
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Cyber Project Stream Monitor", layout="wide")
st.title("Streaming Pipeline Monitor (Static + Dynamic)")

REFRESH_SEC = st.sidebar.slider("Refresh (sec)", 1, 10, 2)
CSV_PATH = st.sidebar.text_input("Results CSV path", "outputs/stream_results.csv")

placeholder = st.empty()

def load_df(path: str):
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return pd.DataFrame(columns=["event_id", "source", "p_malware", "decision", "latency_ms", "label", "user_action"])

while True:
    df = load_df(CSV_PATH)

    with placeholder.container():
        c1, c2, c3, c4 = st.columns(4)

        total = len(df)
        alerts = int((df["decision"] == "ALERT").sum()) if "decision" in df.columns else 0
        reviews = int((df["decision"] == "REVIEW").sum()) if "decision" in df.columns else 0
        passes = int((df["decision"] == "PASS").sum()) if "decision" in df.columns else 0

        c1.metric("Total events", total)
        c2.metric("ALERT", alerts)
        c3.metric("REVIEW", reviews)
        c4.metric("PASS", passes)

        st.subheader("Latest events")
        if total > 0:
            st.dataframe(df.sort_values("event_id", ascending=False).head(50), use_container_width=True)
        else:
            st.info("No events yet. Start the pipeline container or run the demo locally.")

        st.subheader("Decision distribution by source")
        if total > 0:
            pivot = (
                df.pivot_table(index="source", columns="decision", values="event_id", aggfunc="count", fill_value=0)
                .reset_index()
            )
            st.dataframe(pivot, use_container_width=True)

        st.subheader("Latency (ms) - latest 200")
        if "latency_ms" in df.columns and total > 0:
            # Ensure numeric
            tmp = df.copy()
            tmp["latency_ms"] = pd.to_numeric(tmp["latency_ms"], errors="coerce")
            st.line_chart(tmp.sort_values("event_id").tail(200).set_index("event_id")["latency_ms"])

    time.sleep(REFRESH_SEC)
