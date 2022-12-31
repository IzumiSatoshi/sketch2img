import torchvision
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import cv2
from diffusers import UNet2DModel
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, DDIMScheduler
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers import DDPMPipeline

device = "cpu"
base_model_path = ""
save_path = ""

base_model = torch.load(torch)

state_dict = base_model.state_dict()
original_weight = state_dict["conv_in.weight"]
zero_weight = torch.zeros_like(original_weight)
new_weight = torch.concat([original_weight, zero_weight])

state_dict["conv_in.weight"] = new_weight

new_model = UNet2DModel(
    sample_size=28,  # the target image resolution
    in_channels=2,  # the number of input channels, 3 for RGB images
    out_channels=1,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(32, 64),  # More channels -> more parameters
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
    ),
    up_block_types=(
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",  # a regular ResNet upsampling block
    ),
)

new_model.load_state_dict(state_dict)

torch.save(new_model, save_path)
