import gradio as gr
from src.pipeline_ddpm_sketch2img import DDPMSketch2ImgPipeline
import numpy as np
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, DDIMScheduler
from PIL import Image

model_path = "IzumiSatoshi/sketch2img-FashionMNIST"
pipe = DDPMSketch2ImgPipeline.from_pretrained(model_path).to("cpu")
pipe.scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")


def draw(sketch):
    sketch[sketch < 250] = 0
    sketch[sketch >= 250] = 255
    sketch = Image.fromarray(sketch)
    image = pipe(sketch, num_inference_step=50)
    return sketch, image


inp = gr.inputs.Image(
    image_mode="L",
    source="canvas",
    shape=(28, 28),
    invert_colors=True,
    tool="select",
)
demo = gr.Interface(fn=draw, inputs=inp, outputs=["image", "image"])
demo.launch()
