import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from diffusers import UNet2DModel
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, DDIMScheduler
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from Sketch2ImgPipeline import Sketch2ImgPipeline


class Sketch2ImgDataset(Dataset):
    def __init__(self, data_root):
        self.fmnist_dataset = torchvision.datasets.FashionMNIST(data_root)
        self.classes = self.fmnist_dataset.classes

    def __len__(self):
        return len(self.fmnist_dataset)

    def __getitem__(self, index):
        img = self.fmnist_dataset.data[index]
        label = self.fmnist_dataset.targets[index]
        sketch = torch.from_numpy(cv2.Canny(img.numpy(), 200, 200))

        # to shape: [1, h, w]
        img = img.unsqueeze(0).float()
        sketch = sketch.unsqueeze(0).float()

        img = self.normalize(img)
        sketch = self.normalize(sketch)

        return img, sketch, label

    def normalize(self, x):
        # map x to -1 < x < 1
        # I'm doing normalization with zero understanding :o
        x /= 255
        x = torchvision.transforms.Normalize(0.5, 0.5)(x)
        return x


if __name__ == "__main__":
    batch_size = 64
    num_train_timesteps = 1000
    lr = 1e-5
    num_epochs = 2
    device = "cuda"
    train_dataset_rate = 0.1
    save_path = "./model/pipe_test.pth"

    # Create a model
    model = UNet2DModel(
        sample_size=28,  # the target image resolution
        in_channels=2,  # the number of input channels, 3 for RGB images
        out_channels=1,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(64, 128),  # More channels -> more parameters
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        ),
        up_block_types=(
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",  # a regular ResNet upsampling block
        ),
    ).to(device)

    scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
    dataset = Sketch2ImgDataset("./data/FashionMNIST")
    dataset_len = int(len(dataset) * train_dataset_rate)
    dataset, _ = torch.utils.data.random_split(
        dataset, [dataset_len, len(dataset) - dataset_len]
    )

    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for e in range(num_epochs):
        for step, (img, sketch, _) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()

            img = img.to(device)
            sketch = sketch.to(device)
            noise = torch.randn(img.shape).to(device)
            bs = img.shape[0]

            timesteps = torch.randint(
                0, scheduler.num_train_timesteps, (bs,), device=device
            ).long()
            noisy_img = scheduler.add_noise(img, noise, timesteps)

            x = torch.cat([noisy_img, sketch], dim=1).cuda()

            noise_pred = model(x, timesteps).sample

            loss = F.mse_loss(noise_pred, noise)
            # loss.backward(loss) ?
            loss.backward()
            losses.append(loss.item())

            optimizer.step()

        print(f"{e + 1} epoch : {loss.item()}")

    # shouldn't use DDPM scheduler for saved pipeline
    pipe = Sketch2ImgPipeline(unet=model, scheduler=scheduler)
    pipe.save_pretrained(save_path)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(losses)
    axs[1].plot(np.log(losses))
    plt.show()
