import gradio as gr
from PIL import Image
import os

# Dummy prediction function (replace with your model)
def predict_betel_nut(image):
    return "Prediction: Ripeness ✅", image

# Gradio UI
with gr.Blocks(css="static/style.css") as demo:
    with gr.Column(elem_id="main-container"):
        with gr.Row():
            gr.Markdown(
                """
                <div class="header">
                    <img src="https://cdn-icons-png.flaticon.com/512/206/206865.png" class="logo">
                    <h1>Betel Nut Predictor 🌿</h1>
                    <p>Smart AI to detect ripeness and issues in betel nut images</p>
                </div>
                """,
                elem_id="topbar"
            )

        with gr.Row():
            with gr.Column(elem_id="left-panel"):
                gr.Markdown("### 📸 Upload Image")
                image_input = gr.Image(label=None, type="pil")
                submit_btn = gr.Button("🔍 Analyze Now", elem_id="analyze-btn")

                gr.Markdown("### 🔎 Prediction")
                output_text = gr.Textbox(label=None, placeholder="Prediction will appear here...", lines=2)
                output_image = gr.Image(label="Processed Image")

            with gr.Column(elem_id="right-panel"):
                gr.Markdown("### 📅 Historical Analysis")
                gr.Markdown(
                    "<div class='history-placeholder'>No history yet.<br>Upload an image to begin tracking.</div>",
                    elem_id="history-box"
                )

        submit_btn.click(fn=predict_betel_nut, inputs=image_input, outputs=[output_text, output_image])

demo.launch()
