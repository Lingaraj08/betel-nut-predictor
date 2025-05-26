import gradio as gr
from PIL import Image
import os
from utils.predictor import BetelNutPredictor

predictor = BetelNutPredictor()

def predict_betel_nut(image):
    if image is None:
        return "Please upload an image", None
    
    # Get prediction
    result = predictor.predict(image)
    
    # Format output text
    output = f"""🔍 Prediction: {result['class']}
✨ Confidence: {result['confidence']:.2f}%
⏰ Time Since Falling: {result['days_estimate']}
📝 Fall Reason: {result['fall_reason']}

📊 Class Probabilities:
""" + ''.join(f"• {name}: {prob:.2f}%\n" for name, prob in result['probabilities'].items())
    
    return output, image

# Gradio UI
with gr.Blocks(css="static/style.css") as demo:
    with gr.Column(elem_id="main-container"):
        gr.Markdown(
            """
            <div class="header">
                <img src="https://cdn-icons-png.flaticon.com/512/206/206865.png" class="logo">
                <h1>Advanced Betel Nut Analyzer 🌿</h1>
                <p>AI-powered analysis of betel nut condition, fall timing, and causes</p>
            </div>
            """,
            elem_id="topbar"
        )

        with gr.Row():
            with gr.Column(elem_id="left-panel"):
                gr.Markdown("### 📸 Upload Betel Nut Image")
                image_input = gr.Image(label=None, type="pil")
                submit_btn = gr.Button("🔍 Analyze Now", elem_id="analyze-btn")
                
                gr.Markdown("### 🎯 Analysis Results")
                output_text = gr.Textbox(
                    label=None, 
                    placeholder="Upload an image to see detailed analysis...",
                    lines=8
                )
                
            with gr.Column(elem_id="right-panel"):
                gr.Markdown("### 🖼️ Processed Image")
                output_image = gr.Image(label=None)
                
                gr.Markdown("""
                    ### 📋 Analysis Details
                    This tool analyzes:
                    * Ripeness level
                    * Pest damage detection
                    * Rain exposure signs
                    * Drying condition
                    * Days since falling
                    * Cause of falling
                """)

        submit_btn.click(
            fn=predict_betel_nut,
            inputs=image_input,
            outputs=[output_text, output_image]
        )

demo.launch()
