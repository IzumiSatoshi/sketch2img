import gradio as gr
from Sketch2ImgPipeline import Sketch2ImgPipeline
import numpy as np

pipe = Sketch2ImgPipeline.from_pretrained("IzumiSatoshi/sketch2img-FashionMNIST")


def greet(input_img):
    sketches = np.expand_dims(input_img, (0, 1))
    sketches[sketches < 250] = 0
    sketches[sketches >= 250] = 255
    print(sketches.shape)
    samples = pipe(sketches, num_inference_step=10)
    out = samples[0][0]
    print(out.shape)
    return sketches[0][0], out


inp = gr.inputs.Image(
    image_mode="L",
    source="canvas",
    shape=(28, 28),
    invert_colors=True,
    tool="select",
)
demo = gr.Interface(fn=greet, inputs=inp, outputs=["image", "image"])
demo.launch()
