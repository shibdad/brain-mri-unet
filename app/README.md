# Interactive demo (in progress)

This is where the web demo lives — a small Gradio app, hosted free on Hugging Face Spaces, that runs the trained model on example brain-MRI slices so anyone can see it work without cloning the repo.

The plan: upload an MRI slice (or pick a bundled example), adjust a threshold slider, and get back the original slice with the predicted tumor contour drawn on it plus the estimated tumor area. It reuses the inference code already in the repo — no model logic gets rewritten for the front end:

```python
# app/app.py (planned)
import gradio as gr
from src.inference import load_model, predict_overlay

model = load_model("best_model.pt", "cpu")

def segment(image, threshold):
    overlay, area = predict_overlay(model, image, "cpu", threshold)
    return overlay, f"Tumor area fraction: {area:.4f}"

demo = gr.Interface(
    fn=segment,
    inputs=[gr.Image(type="numpy"), gr.Slider(0.1, 0.9, value=0.5, label="Threshold")],
    outputs=[gr.Image(label="Predicted contour"), gr.Text(label="Stats")],
    examples=[["examples/case1.png", 0.5]],
    title="Brain MRI Tumor Contouring (U-Net)",
)

if __name__ == "__main__":
    demo.launch()
```

Deploying is a matter of creating a Gradio Space, adding `app.py`, a slim `requirements.txt`, the `src/` package, the trained `best_model.pt`, and a few example slices — Spaces builds and serves it from there.

Status: the model is trained and the inference API (`src/inference.py`) is already shaped for this. Next step is wiring up `app.py` and pushing the Space.
