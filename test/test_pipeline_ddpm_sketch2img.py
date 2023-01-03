import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import random

# mine
import sys

sys.path.append("src")
from pipeline_ddpm_sketch2img import DDPMSketch2ImgPipeline

fig, axs = plt.subplots(2, 3)

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

pipe = DDPMSketch2ImgPipeline.from_pretrained("model/from_init_test").to("cuda")
dataset = torchvision.datasets.FashionMNIST("data/FashionMNIST", download=False)

image_np = dataset.data[0].numpy()
image_pil = Image.fromarray(image_np)
sketch_np = cv2.Canny(image_np, 200, 200)
sketch_pil = Image.fromarray(sketch_np)
image_tensor = torch.from_numpy(image_np).unsqueeze(0).float()
sketch_tensor = torch.from_numpy(sketch_np).unsqueeze(0).float()

round_trip = pipe.denormalize(pipe.normalize(image_tensor))

axs[0, 0].set_title("original")
axs[0, 0].imshow(image_tensor[0], "gray")

axs[0, 1].set_title("transformed then invert transformed")
axs[0, 1].imshow(round_trip[0], "gray")

# Why are the numbers slightly different?
print(torch.equal(pipe.denormalize(pipe.normalize(image_tensor)), image_tensor))

generated_image_pil = pipe(sketch_pil)

axs[1, 0].set_title("image")
axs[1, 0].imshow(image_pil, "gray")

axs[1, 1].set_title("sketch")
axs[1, 1].imshow(sketch_pil, "gray")

axs[1, 2].set_title("generated")
axs[1, 2].imshow(generated_image_pil, "gray")

plt.show()
