import gradio as gr
import pandas as pd

from combined_multimodal import run_multimodal


def get_path(x):
    if x is None:
        return None
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return x.get("name") or x.get("path")
    return None


def wrapped_run(audio, video, image):
    audio_path = get_path(audio)
    video_path = get_path(video)
    image_path = get_path(image)
    return run_multimodal(audio_path, video_path, image_path)


with gr.Blocks() as demo:
    gr.Markdown("# 🎧🎥 Multimodal Sensory Environment Analyzer + 🌤️ Weather API")
    gr.Markdown("Upload audio/video, capture face image, and click **Run Full Analysis**.")

    with gr.Row():
        audio_input = gr.Audio(
            type="filepath",
            label="Audio Input"
        )
        video_input = gr.Video(
            label="Video Input"
        )
        image_input = gr.Image(
            type="filepath",
            label="Face Image"
        )

    run_btn = gr.Button("Run Full Analysis")

    summary_out = gr.Markdown(label="Summary Output")
    table_out = gr.Dataframe(label="Segment-wise Multimodal Table")
    weather_out = gr.Markdown(label="Live Weather Info")

    run_btn.click(
        fn=wrapped_run,
        inputs=[audio_input, video_input, image_input],
        outputs=[summary_out, table_out, weather_out]
    )

if __name__ == "__main__":
    demo.launch(share=True)
