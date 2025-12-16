import os
import uuid
import streamlit as st
import matplotlib.pyplot as plt

from VideoProcessing import process_multimodal

st.set_page_config(layout="wide")
st.title("🧠 ASD Multimodal Overload Detection")

# ================= SESSION STATE =================
if "df" not in st.session_state:
    st.session_state.df = None

# ================= FILE UPLOAD =================
uploaded_videos = st.file_uploader(
    "Upload video(s)",
    type=["mp4", "mov", "avi", "mkv"],
    accept_multiple_files=True
)

if st.button("▶ Run Analysis"):

    if not uploaded_videos:
        st.error("Please upload at least one video.")
        st.stop()

    run_dir = f"video_samples_{uuid.uuid4().hex[:6]}"
    os.makedirs(run_dir, exist_ok=True)

    for v in uploaded_videos:
        with open(os.path.join(run_dir, v.name), "wb") as f:
            f.write(v.read())

    with st.spinner("Processing video(s)..."):
        df = process_multimodal(run_dir)

    if df is None or df.empty:
        st.error("No valid analysis windows produced.")
        st.stop()

    st.session_state.df = df
    st.success("Analysis completed!")

# ================= DISPLAY =================
if st.session_state.df is not None:
    df = st.session_state.df

    st.subheader("Preview")
    st.dataframe(df.head(20))

    video_id = st.selectbox("Select video", df["video_id"].unique())

    vdf = df[df["video_id"] == video_id].copy()

    def ts_to_sec(t):
        h, m, s = map(int, t.split(":"))
        return h * 3600 + m * 60 + s

    vdf["time"] = vdf["timestamp"].apply(ts_to_sec)
    state_map = {"Calm": 0, "Moderate Overload": 1, "Severe Overload": 2}
    vdf["state_level"] = vdf["overall_state"].map(state_map)

    fig, ax = plt.subplots()
    ax.plot(vdf["time"], vdf["state_level"], marker="o")

    ax.scatter(
        vdf[vdf["overload_flag"] == 1]["time"],
        vdf[vdf["overload_flag"] == 1]["state_level"],
        color="red",
        label="Overload"
    )

    ax.scatter(
        vdf[vdf["distress_flag"] == 1]["time"],
        vdf[vdf["distress_flag"] == 1]["state_level"],
        color="orange",
        label="Distress"
    )

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Calm", "Moderate", "Severe"])
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("State")
    ax.legend()

    st.pyplot(fig)

    st.download_button(
        "⬇ Download CSV",
        df.to_csv(index=False),
        "multimodal_output.csv"
    )
