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
    # currently this method is not creating sketch2img pipeline.
    # just replacing unet
    # TODO: implement this as Sketch2Img pipline's class method

    unet_class = pipe.config["unet"][1]

    # modify conv_in.weight
    state_dict = pipe.unet.state_dict()
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
