# Interactive demo (next iteration)

This folder will hold the **Gradio** web app that showcases the trained model on
example brain-MRI slices, deployed for free on **Hugging Face Spaces**.

## Planned UX

- Upload an MRI slice **or** pick from bundled example images.
- A threshold slider to trade off sensitivity vs. specificity at inference time.
- Output: the original slice with the predicted tumor **contour** overlaid, plus
  the estimated tumor area fraction.

## How it will be built

The app reuses the inference API already in this repo — no model code is
duplicated:

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

## Deployment plan (Hugging Face Spaces)

1. Create a new **Gradio** Space.
2. Add `app.py`, `requirements.txt` (torch + gradio + opencv + numpy), the `src/`
   package, a small `best_model.pt`, and a few bundled `examples/`.
3. Push — Spaces builds and serves the live demo automatically.

> Status: **stubbed**. Train the model first (see the repo root README), then we
> wire up `app.py` and deploy.
