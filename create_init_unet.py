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
from diffusers import DDPMPipeline, UNet2DModel, UNet2DConditionModel
import json
import copy


def create_init_sketch2img_pipe(pipe, save_pipe_path, add_channels_num):

    unet_class = pipe.config["unet"][1]

    # modify conv_in.weight
    state_dict = pipe.unet.state_dcict()
    base_weight = state_dict["conv_in.weight"]
    s = base_weight.shape
    zero_weight = torch.zeros((s[0], add_channels_num, s[2], s[3]))

    new_weight = torch.concat([base_weight, zero_weight], dim=1)
    state_dict["conv_in.weight"] = new_weight

    # create new unet from config
    new_unet_config = copy.deepcopy(pipe.unet.config)
    new_unet_config["in_channels"] = pipe.unet.config["in_channels"] + add_channels_num

    new_unet = None
    if unet_class == "UNet2DModel":
        new_unet = UNet2DModel.from_config(new_unet_config)
    if unet_class == "UNet2DConditionModel":
        new_unet = UNet2DConditionModel.from_config(new_unet_config)

    # load state dict
    new_unet.load_state_dict(state_dict)

    # replace unet
    pipe.unet = new_unet

    # save
    pipe.save_pretrained(save_pipe_path)


def from_torch_model():
    device = "cpu"
    base_model_path = ""
    save_path = ""

    base_model = torch.load(torch)

    state_dict = base_model.state_dict()
    original_weight = state_dict["conv_in.weight"]
    s = original_weight.shape
    zero_weight = torch.zeros((s[0], 1, s[2], s[3]))
    new_weight = torch.concat([original_weight, zero_weight], dim=1)

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
